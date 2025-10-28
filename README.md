# fucto: CTO.NEW 到 OpenAI API 的逆向工程桥接

**TL;DR**: 把 CTO.NEW 的私有 API 包装成标准的 OpenAI 兼容接口。

## 架构

- `cto_new_client.py` - 核心客户端，处理所有与 CTO.NEW 的交互
- `openai_api_server.py` - FastAPI 服务器，提供 `/v1/chat/completions` 接口
- `websocket_example.py` - 命令行测试工具

## 快速部署

### 使用 uv (推荐)

```bash
git clone <repo> fucto && cd fucto
uv venv && source .venv/bin/activate
uv sync
# 把你的 cookie 放到 cookies/cookies.txt 里
uv run uvicorn openai_api_server:app --host 0.0.0.0 --port 8000
```

### 使用 Docker

```bash
docker build -t fucto .
docker run -d --name fucto -p 8000:8000 \
  -v $(pwd)/cookies:/app/cookies:ro \
  fucto
```

- 仓库已自带 `Dockerfile`，可直接使用上述命令构建镜像。
- 如果需要自定义镜像名称，将 `-t fucto` 调整为你自己的仓库名即可。

### 使用 Docker Compose

```yaml
services:
  fucto:
    build: .
    image: fucto:latest
    ports:
      - "8000:8000"
    volumes:
      - ./cookies:/app/cookies:ro
    restart: unless-stopped
```

```bash
docker compose up -d   # 构建并启动
docker compose down    # 停止并清理
```

- 记得在 `docker compose up` 之前先准备好同目录下的 `cookies/cookies.txt`。

## Cookie 获取

1. 登录 cto.new
2. 打开开发者工具，找到对 `clerk.cto.new` 的请求
3. 复制完整的 `cookie` 头部值
4. 粘贴到 `cookies/cookies.txt` 中（每行一个 cookie）

## 测试

```bash
# 测试 API
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-5", "messages": [{"role": "user", "content": "测试"}]}'

# 测试 WebSocket
uv run python websocket_example.py
```


## 支持的模型

当前通过 `/v1/models` 会返回以下 OpenAI 兼容模型名称，并映射到 CTO.NEW 的适配器：

| OpenAI 模型名            | CTO.NEW Adapter |
|-------------------------|-----------------|
| `gpt-5`                 | `GPT5`          |
| `claude-sonnet-4-5`     | `ClaudeSonnet4_5` |

如需扩展，修改 `openai_api_server.py` 中的 `MODEL_MAPPING`。

### 性能优化

- 使用 `httpx.AsyncClient` 替代 `requests`，避免阻塞事件循环
- 连接池复用，减少连接开销
- 使用 `tiktoken` 进行准确的 token 计算（如果可用）

## 故障排除

### 401 认证失败

Cookie 过期了。重新获取并更新 `cookies/cookies.txt`。

### 500 服务器错误

检查：
1. Cookie 是否有效
2. CTO.NEW 服务是否正常
3. 网络连接是否正常

### WebSocket 连接问题

可能是网络防火墙或代理问题。尝试直接连接或使用 VPN。
