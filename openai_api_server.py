"""
OpenAI 格式兼容的 API 服务器
提供标准的 /v1/chat/completions 接口，桥接到现有的 CTO.NEW AI 服务
"""

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncGenerator, Union, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_500_INTERNAL_SERVER_ERROR, HTTP_502_BAD_GATEWAY

from cto_new_client import CtoNewClient, AuthError, ApiError

try:
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")
except ImportError:
    tiktoken = None
    encoding = None

# ========== 配置 ==========
COOKIES_DIR = Path(__file__).with_name("cookies")
COOKIES_FILE = COOKIES_DIR / "cookies.txt"

# 模型映射：将 OpenAI 模型名称映射到 CTO.NEW 的 adapter
MODEL_MAPPING = {
    "gpt-5": "GPT5",
    "claude-sonnet-4-5": "ClaudeSonnet4_5",
}
DEFAULT_ADAPTER = "ClaudeSonnet4_5"
PROXY_URL = os.getenv("PROXY_URL")  # 支持通过环境变量配置代理, e.g., "http://127.0.0.1:7890"


# ========== 线程安全的 Cookie 管理器 ==========
class CookieManager:
    def __init__(self, file_path: Path):
        self._file_path = file_path
        self._dir_path = file_path.parent
        self._cookies: List[str] = []
        self._index = 0
        self._mtime: Optional[float] = None
        self._lock = asyncio.Lock()
        self._load_cookies()

    def _load_cookies(self):
        try:
            if not self._dir_path.exists():
                raise FileNotFoundError(f"Cookie 目录不存在: {self._dir_path}")

            current_mtime = self._file_path.stat().st_mtime
            if self._mtime is not None and self._mtime == current_mtime:
                return

            with open(self._file_path, "r", encoding="utf-8") as f:
                raw_lines = f.readlines()

            cookies = [line.strip() for line in raw_lines if line.strip() and not line.strip().startswith("#")]
            if not cookies:
                raise ValueError(f"No cookies found in {self._file_path}")

            self._cookies = cookies
            self._index = 0
            self._mtime = current_mtime
            print(f"成功加载 {len(self._cookies)} 个 cookies")

        except FileNotFoundError as exc:
            self._cookies = []
            print(f"警告：{exc}")
        except ValueError as e:
            print(f"警告：{e}")

    async def get_cookie(self) -> str:
        async with self._lock:
            self._load_cookies()
            if not self._cookies:
                raise ValueError("Cookie 池为空")

            cookie = self._cookies[self._index]
            self._index = (self._index + 1) % len(self._cookies)
            return cookie


cookie_manager = CookieManager(COOKIES_FILE)


# ========== OpenAI 兼容辅助 ==========
def build_openai_error(
    message: str,
    type_: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"message": message, "type": type_}
    if param:
        payload["param"] = param
    if code:
        payload["code"] = code
    return {"error": payload}


def openai_http_exception(
    status_code: int,
    message: str,
    type_: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
) -> HTTPException:
    return HTTPException(status_code=status_code, detail=build_openai_error(message, type_, param, code))


# 模型映射：将 OpenAI 模型名称映射到 CTO.NEW 的 adapter
# ========== Pydantic Models ==========
class MessageContentPart(BaseModel):
    type: Literal["text"]
    text: str


class Message(BaseModel):
    role: str
    content: Union[str, List[MessageContentPart]]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


# ========== FastAPI App & Lifecycle ==========
# CtoNewClient 内部管理其网络会话，不再需要全局的 lifespan 管理器
app = FastAPI(title="OpenAI Compatible API", version="1.2.0")


# ========== 辅助函数 ==========
def count_tokens(text: str) -> int:
    """使用 tiktoken 计算 token 数量，如果库不可用则返回 0"""
    if not encoding:
        return 0
    return len(encoding.encode(text))


def format_chat_history(messages: List[Message]) -> str:
    """将聊天历史格式化为单个字符串 prompt"""
    rendered_messages: List[str] = []
    for index, msg in enumerate(messages):
        content = render_message_content(msg.content, index)
        rendered_messages.append(f"{msg.role}: {content}")
    return "\n".join(rendered_messages)


def render_message_content(content: Union[str, List[MessageContentPart]], message_index: int) -> str:
    if isinstance(content, str):
        return content

    segments: List[str] = []
    for part_index, part in enumerate(content):
        if part.type != "text":
            raise openai_http_exception(
                HTTP_400_BAD_REQUEST,
                f"Unsupported content part type '{part.type}'. Only 'text' is supported.",
                param=f"messages[{message_index}].content[{part_index}].type",
                code="unsupported_message_content_type",
            )
        segments.append(part.text)

    return "".join(segments)


# ========== 异常处理 ==========
@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    first_error = exc.errors()[0] if exc.errors() else {}
    loc = first_error.get("loc", [])
    param = ".".join(str(part) for part in loc if part != "body") or None
    message = first_error.get("msg", "Invalid request payload.")
    code = first_error.get("type")
    return JSONResponse(
        status_code=HTTP_400_BAD_REQUEST,
        content=build_openai_error(message, param=param, code=code),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        content = exc.detail
    else:
        content = build_openai_error(str(exc.detail or ""))
    return JSONResponse(status_code=exc.status_code, content=content)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    print(f"未处理异常: {exc}")
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content=build_openai_error(
            "Internal server error.",
            type_="server_error",
            code="internal_error",
        ),
    )


async def stream_ai_response(client: CtoNewClient, chat_id: str, model: str) -> AsyncGenerator[str, None]:
    """使用 CtoNewClient 流式获取并格式化 AI 响应"""
    stream_id = f"chatcmpl-{chat_id}"
    created_time = int(time.time())

    try:
        role_chunk = {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(role_chunk)}\n\n"

        async for content_chunk in client.stream_chat_response(chat_id):
            chunk = {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": content_chunk}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        # 发送结束标记
        final_chunk = {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    except ApiError as e:
        # 在流中报告错误
        error_chunk = {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": f"\n\n[ERROR: {e}]"}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"


# ========== API Routes ==========
@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "OpenAI Compatible API Server",
        "endpoints": {"chat": "/v1/chat/completions", "models": "/v1/models"},
    }


@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return {
        "object": "list",
        "data": [
            {"id": model_name, "object": "model", "created": int(time.time()), "owned_by": "cto-new"}
            for model_name in MODEL_MAPPING.keys()
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: Request, payload: ChatCompletionRequest):
    """
    OpenAI 兼容的聊天完成接口
    支持流式和非流式响应
    """
    try:
        cookie = await cookie_manager.get_cookie()
        client = CtoNewClient(cookie, proxy=PROXY_URL)

        # 认证
        await client.authenticate()

        # 确定 adapter 并格式化 prompt
        adapter = MODEL_MAPPING.get(payload.model, DEFAULT_ADAPTER)
        if not payload.messages:
            raise openai_http_exception(
                HTTP_400_BAD_REQUEST,
                "messages must be a non-empty array.",
                param="messages",
                code="missing_required_field",
            )
        if payload.messages[-1].role != "user":
            raise openai_http_exception(
                HTTP_400_BAD_REQUEST,
                "The last message must be from the user.",
                param="messages",
                code="invalid_message_role",
            )

        prompt = format_chat_history(payload.messages)
        if not prompt.strip():
            raise openai_http_exception(
                HTTP_400_BAD_REQUEST,
                "At least one user message with content is required.",
                param="messages",
                code="invalid_message_content",
            )

        # 创建聊天
        chat_id = await client.create_chat(prompt, adapter)

        # 流式响应
        if payload.stream:
            return StreamingResponse(stream_ai_response(client, chat_id, payload.model), media_type="text/event-stream")

        # 非流式响应
        response_chunks = [chunk async for chunk in client.stream_chat_response(chat_id)]
        response_content = "".join(response_chunks)

        return ChatCompletionResponse(
            id=f"chatcmpl-{chat_id}",
            created=int(time.time()),
            model=payload.model,
            choices=[
                ChatCompletionChoice(
                    index=0, message=Message(role="assistant", content=response_content), finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=count_tokens(prompt),
                completion_tokens=count_tokens(response_content),
                total_tokens=count_tokens(prompt) + count_tokens(response_content),
            ),
        )

    except HTTPException:
        raise
    except (AuthError, ApiError) as e:
        raise openai_http_exception(
            HTTP_502_BAD_GATEWAY,
            f"Upstream service error: {e}",
            type_="server_error",
            code="upstream_error",
        )
    except ValueError as e:
        raise openai_http_exception(
            HTTP_400_BAD_REQUEST,
            str(e),
            code="invalid_request",
        )
    except Exception as e:
        # 捕获所有其他意外错误
        print(f"发生意外错误：{e}")
        raise openai_http_exception(
            HTTP_500_INTERNAL_SERVER_ERROR,
            "Internal server error.",
            type_="server_error",
            code="internal_error",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
