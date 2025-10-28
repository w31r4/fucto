import asyncio
import json
import uuid
from typing import AsyncGenerator, Optional

from curl_cffi import requests as cffi_requests
import websockets
from websockets.exceptions import ConnectionClosed

from ua_utils import BrowserFingerprint


# ========== 自定义异常 ==========
class CtoNewError(Exception):
    """cto.new 客户端相关错误的基类"""

    pass


class AuthError(CtoNewError):
    """认证失败错误"""

    pass


class ApiError(CtoNewError):
    """API 调用失败错误"""

    pass


# ========== 核心客户端 ==========
class CtoNewClient:
    """
    与 cto.new 后端服务交互的异步客户端。
    使用 curl_cffi 模拟浏览器 TLS 指纹，并支持代理。
    """

    BASE_URL = "https://api.enginelabs.ai/engine-agent"
    CLERK_URL = "https://clerk.cto.new"
    CLERK_API_VERSION = "2025-04-10"
    CLERK_JS_VERSION = "5.102.0"

    def __init__(self, cookie: str, proxy: Optional[str] = None):
        self._cookie = cookie
        self._jwt: Optional[str] = None
        self._ws_user_token: Optional[str] = None
        self._session_id: Optional[str] = None
        self._fingerprint = BrowserFingerprint.create()

        proxies = {"http": proxy, "https": proxy} if proxy else None
        # 使用 impersonate 模拟真实浏览器的 TLS 指纹，这是对抗指纹识别的关键
        self._client = cffi_requests.AsyncSession(
            impersonate="chrome120",
            proxies=proxies,
            timeout=30,
        )

    async def _get_clerk_info(self):
        """获取 Clerk 会话信息"""
        url = f"{self.CLERK_URL}/v1/me/organization_memberships"
        params = {
            "paginated": "true",
            "limit": "10",
            "offset": "0",
            "__clerk_api_version": self.CLERK_API_VERSION,
            "_clerk_js_version": self.CLERK_JS_VERSION,
        }
        # 使用 ua_utils 生成动态请求头
        headers = self._fingerprint.build_headers(origin=self.CLERK_URL, referer=f"{self.CLERK_URL}/")
        headers["cookie"] = self._cookie

        try:
            r = await self._client.get(url, headers=headers, params=params)
            r.raise_for_status()
            data = r.json()

            client_data = data.get("client", {})
            sessions = client_data.get("sessions", [])
            if not sessions:
                raise AuthError("Clerk 响应中没有可用的 session")

            session = sessions[0]
            self._session_id = client_data.get("last_active_session_id") or session.get("id")
            self._ws_user_token = (
                session.get("ws_user_token") or session.get("wsToken") or session.get("user", {}).get("id")
            )

            if not self._session_id or not self._ws_user_token:
                raise AuthError("无法在 Clerk 响应中找到 session_id 或 user_id")

        except cffi_requests.errors.RequestsError as e:
            body = e.response.text[:200] if e.response else "No response"
            raise AuthError(f"获取 Clerk 信息失败: {e} - {body}") from e
        except (KeyError, IndexError) as e:
            raise AuthError(f"解析 Clerk 响应失败: {e}") from e

    async def _refresh_jwt(self):
        """刷新 JWT"""
        if not self._session_id:
            raise AuthError("刷新 JWT 前必须先获取 session_id")

        url = (
            f"{self.CLERK_URL}/v1/client/sessions/{self._session_id}/tokens"
            f"?__clerk_api_version={self.CLERK_API_VERSION}&_clerk_js_version={self.CLERK_JS_VERSION}"
        )
        headers = self._fingerprint.build_headers(origin=self.CLERK_URL, referer=f"{self.CLERK_URL}/")
        headers["cookie"] = self._cookie
        headers["content-type"] = "application/x-www-form-urlencoded"

        try:
            r = await self._client.post(url, headers=headers, data="", follow_redirects=True)
            r.raise_for_status()
            self._jwt = r.json().get("jwt")
            if not self._jwt:
                raise AuthError("JWT 为空")
        except cffi_requests.errors.RequestsError as e:
            body = e.response.text[:200] if e.response else "No response"
            raise AuthError(f"刷新 JWT 失败: {e} - {body}") from e

    async def authenticate(self):
        """执行完整的认证流程"""
        await self._get_clerk_info()
        await self._refresh_jwt()

    async def create_chat(self, prompt: str, adapter: str) -> str:
        """创建新的聊天会话"""
        if not self._jwt:
            await self.authenticate()

        chat_id = str(uuid.uuid4())
        url = f"{self.BASE_URL}/chat"
        headers = self._fingerprint.build_headers(origin="https://cto.new", referer="https://cto.new/")
        headers["authorization"] = f"Bearer {self._jwt}"
        data = {"prompt": prompt, "chatHistoryId": chat_id, "adapterName": adapter}

        try:
            r = await self._client.post(url, headers=headers, json=data, follow_redirects=True)
            r.raise_for_status()
            return chat_id
        except cffi_requests.errors.RequestsError as e:
            raise ApiError(f"创建聊天失败: {e}") from e

    async def stream_chat_response(self, chat_id: str) -> AsyncGenerator[str, None]:
        """通过 WebSocket 流式获取 AI 响应"""
        if not self._ws_user_token:
            await self.authenticate()

        ws_url = (
            f"wss://api.enginelabs.ai/engine-agent/chat-histories/{chat_id}"
            f"/buffer/stream?token={self._ws_user_token}"
        )

        try:
            # 注意：websockets 不通过 curl_cffi，所以代理需要单独配置（如果需要）
            # 但通常 WebSocket 的指纹检测没有 HTTP 严格
            ws_headers = self._fingerprint.build_ws_headers(origin="https://cto.new", referer="https://cto.new/")
            async with websockets.connect(ws_url, max_size=2**20, extra_headers=ws_headers) as ws:
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                        data = json.loads(msg)

                        if data.get("type") == "update" and data.get("buffer"):
                            inner = json.loads(data["buffer"])
                            if inner.get("type") == "chat":
                                content = inner.get("chat", {}).get("content", "")
                                if content:
                                    yield content
                        elif data.get("type") == "state" and not data["state"].get("inProgress"):
                            break
                    except (json.JSONDecodeError, KeyError):
                        continue
                    except asyncio.TimeoutError:
                        break
        except (ConnectionClosed, asyncio.TimeoutError):
            pass
        except Exception as e:
            raise ApiError(f"WebSocket 通信错误: {e}") from e
