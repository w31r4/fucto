import asyncio
from pathlib import Path

import httpx
from cto_new_client import CtoNewClient, CtoNewError

# ========== ⚙️ 配置区 ==========
# 从 cookies/cookies.txt 读取第一个 cookie
COOKIES_PATH = Path(__file__).with_name("cookies") / "cookies.txt"

try:
    with open(COOKIES_PATH, "r", encoding="utf-8") as f:
        COOKIES = f.readline().strip()
    if not COOKIES:
        raise FileNotFoundError
except FileNotFoundError:
    print(f"错误：未在 {COOKIES_PATH} 找到有效 cookie。请创建文件并添加至少一个 cookie。")
    exit(1)

ADAPTER = "GPT5"  # 或者 "ClaudeSonnet4_5"
# ==============================


async def main():
    """
    使用 CtoNewClient 与 cto.new 服务进行交互式聊天。
    """
    async with httpx.AsyncClient(timeout=30.0) as http_client:
        client = CtoNewClient(cookie=COOKIES, client=http_client)

        try:
            print("🚀 正在认证...")
            await client.authenticate()
            print("✅ 认证成功！")

            # 创建一个新的聊天会话
            initial_prompt = "你好"
            print(f"🆕 正在创建新的聊天会话 (adapter: {ADAPTER})...")
            chat_id = await client.create_chat(initial_prompt, ADAPTER)
            print(f"✅ 会话创建成功，Chat ID: {chat_id}")

            # 显示初始响应
            print("\n🤖 Assistant:")
            async for chunk in client.stream_chat_response(chat_id):
                print(chunk, end="", flush=True)
            print("\n" + "=" * 20)

            # 进入交互式聊天循环
            while True:
                try:
                    prompt = input("You: ").strip()
                    if prompt.lower() in {"exit", "quit"}:
                        print("👋 告辞！")
                        break

                    if not prompt:
                        continue

                    # 在同一个会话中发送新消息
                    # 注意：cto.new 的这个流程可能不是标准的“多轮对话”，
                    # 每次 POST 都是一次新的 prompt，但共享同一个 chat_id。
                    # 这里的实现遵循了原脚本的逻辑。
                    await client.create_chat(prompt, ADAPTER)

                    print("\n🤖 Assistant:")
                    async for chunk in client.stream_chat_response(chat_id):
                        print(chunk, end="", flush=True)
                    print("\n" + "=" * 20)

                except (KeyboardInterrupt, EOFError):
                    print("\n👋 告辞！")
                    break

        except CtoNewError as e:
            print(f"\n❌ 发生错误：{e}")
        except Exception as e:
            print(f"\n❌ 发生未知错误：{e}")


if __name__ == "__main__":
    asyncio.run(main())
