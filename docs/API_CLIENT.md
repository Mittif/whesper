# Whesper 客户端 API 文档（v1）

面向 Web / iOS / Android / 桌面客户端接入的后端协议说明。

- 基础路径：`/v1`
- 内容类型：`application/json`（流式接口为 `text/event-stream`）
- 时间格式：ISO-8601 UTC，例如 `2026-04-20T10:00:00+00:00`
- 字符编码：UTF-8
- OpenAPI：`/openapi.json`，交互式文档：`/docs`

## 1. 接入前提与基地址

客户端不应假设服务运行在某一台本机、某个固定端口，或直接知道服务端内部配置。接入时应由部署方提供以下信息：

- `BASE_URL`：客户端可访问的服务根地址，例如 `https://agent.example.com`
- `API_PREFIX`：当前版本固定为 `/v1`
- `ACCESS_TOKEN`：当服务启用鉴权时，由部署方签发或下发给客户端的 Bearer Token

客户端拼接后的完整请求地址示例：

```text
https://agent.example.com/v1/health
```

本地开发时，部署方也可以临时提供 `http://127.0.0.1:8765` 这类地址，但这只是开发环境取值，不属于客户端协议的一部分。

## 2. 认证

除 `GET /v1/health` 外，其他接口在服务启用鉴权时都需要 Bearer Token：

```http
Authorization: Bearer <token>
```

对客户端而言，Token 只是一段由部署方提供的访问凭证。客户端不应依赖服务端内部是通过配置文件、环境变量还是网关转发来校验该 Token。

未携带或错误时返回 `401`：

```json
{
  "code": "UNAUTHORIZED",
  "message": "Missing or malformed Authorization header."
}
```

## 3. 通用错误结构

所有非 2xx 错误统一返回：

```json
{
  "code": "VALIDATION_ERROR",
  "message": "limit must be between 1 and 500",
  "detail": null
}
```

常见错误码：

- `UNAUTHORIZED`
- `NOT_FOUND`
- `VALIDATION_ERROR`
- `CONFLICT`
- `PROVIDER_ERROR`
- `TOOL_EXECUTION_ERROR`
- `CONFIG_ERROR`
- `GENERATION_INTERRUPTED`
- `INTERNAL_ERROR`（主要在流式事件里）

## 4. 核心数据模型

### 4.1 ChatMessage

```json
{
  "role": "assistant",
  "content": "你好，我在。",
  "created_at": "2026-04-20T10:00:00+00:00",
  "model_alias": "kimi-k2.5",
  "route_reason": "default chat mode (default chat model)",
  "reasoning_content": null,
  "name": null,
  "tool_call_id": null,
  "tool_calls": null,
  "source_profile": null
}
```

### 4.2 RouteDecision

```json
{
  "model_alias": "kimi-k2.5",
  "mode": "chat",
  "reason": "default chat mode (default chat model)"
}
```

### 4.3 AskUserAction

```json
{
  "prompt": "你希望查哪个城市？",
  "options": [
    {"label": "上海", "value": "Shanghai", "description": null},
    {"label": "北京", "value": "Beijing", "description": null}
  ],
  "allow_free_text": true,
  "field_name": "city"
}
```

客户端收到 `ask_user` 后，下一次调用 `turn`/`turn:stream` 时，将用户选择值或自由输入作为 `text` 发回即可。

### 4.4 SessionSummary

```json
{
  "session_id": "main",
  "created_at": "2026-04-20T10:00:00+00:00",
  "updated_at": "2026-04-20T10:01:00+00:00",
  "pinned_model": "auto",
  "message_count": 6,
  "has_pending_ask_user": false
}
```

## 5. 分页约定

消息列表接口使用：

- `limit`：`1..500`，默认 `50`
- `before`：ISO 时间戳，返回满足 `created_at < before` 的更早消息

返回结构：

```json
{
  "items": [],
  "next_before": "2026-04-20T09:58:00+00:00"
}
```

`next_before` 为 `null` 表示没有下一页。

## 6. HTTP 接口

### 6.1 健康检查（无需认证）

`GET /v1/health`

响应：

```json
{
  "ok": true,
  "version": "0.1.0"
}
```

### 6.2 能力说明

`GET /v1/capabilities`

响应：

```json
{
  "version": "0.1.0",
  "features": ["tools", "memory", "cup", "trace", "search"],
  "route_modes": ["auto", "chat", "reasoning", "search"],
  "websearch_providers": ["duckduckgo", "brave", "serpapi"]
}
```

### 6.3 列出会话

`GET /v1/sessions`

响应：

```json
{
  "items": [
    {
      "session_id": "main",
      "created_at": "2026-04-20T10:00:00+00:00",
      "updated_at": "2026-04-20T10:01:00+00:00",
      "pinned_model": "auto",
      "message_count": 6,
      "has_pending_ask_user": false
    }
  ],
  "next_before": null
}
```

### 6.4 创建会话

`POST /v1/sessions`

请求体：

```json
{
  "session_id": "mobile-user-001"
}
```

响应：`201 Created`，返回 `SessionSummary`。  
错误：

- `400`：`session_id` 非法
- `409`：会话已存在

### 6.5 查询会话

`GET /v1/sessions/{session_id}`

响应：`SessionSummary`。  
错误：`404 NOT_FOUND`。

### 6.6 删除会话

`DELETE /v1/sessions/{session_id}`

响应：

```json
{
  "ok": true
}
```

错误：`404 NOT_FOUND`。

### 6.7 获取消息（完整消息）

`GET /v1/sessions/{session_id}/messages?limit=50&before=...`

响应：

```json
{
  "items": [
    {
      "role": "assistant",
      "content": "你好",
      "created_at": "2026-04-20T10:00:00+00:00",
      "model_alias": "kimi-k2.5",
      "route_reason": "default chat mode (default chat model)",
      "reasoning_content": null,
      "name": null,
      "tool_call_id": null,
      "tool_calls": null,
      "source_profile": null
    }
  ],
  "next_before": null
}
```

错误：

- `404 NOT_FOUND`
- `400 VALIDATION_ERROR`（比如 `limit` 超范围）

### 6.8 获取 transcript（面向展示的精简对话）

`GET /v1/sessions/{session_id}/transcript?limit=50&before=...`

返回结构同 `messages`，但内容源是 transcript 记录（通常更适合作为聊天记录展示）。

### 6.9 非流式发起一轮对话

`POST /v1/sessions/{session_id}/turn`

请求体：

```json
{
  "text": "帮我总结今天的重点",
  "mode": "auto"
}
```

`mode` 建议值：`auto | chat | reasoning | search`。  
当前实现对未知值不会直接报错，通常会按 `auto` 路由行为处理，客户端仍建议只传受支持值。

成功响应：

```json
{
  "decision": {
    "model_alias": "kimi-k2.5",
    "mode": "reasoning",
    "reason": "matched keyword: 总结 (default chat model)"
  },
  "assistant_message": {
    "role": "assistant",
    "content": "这是总结...",
    "created_at": "2026-04-20T10:02:00+00:00",
    "model_alias": "kimi-k2.5",
    "route_reason": "matched keyword: 总结 (default chat model)",
    "reasoning_content": null,
    "name": null,
    "tool_call_id": null,
    "tool_calls": null,
    "source_profile": null
  },
  "ask_user": null
}
```

错误：

- `404 NOT_FOUND`：会话不存在
- `409 CONFLICT`：同一会话有并发请求（busy）
- `499 GENERATION_INTERRUPTED`
- `502 PROVIDER_ERROR`
- `500 TOOL_EXECUTION_ERROR | CONFIG_ERROR`

## 7. SSE 流式接口

`POST /v1/sessions/{session_id}/turn:stream`

- 请求体同 `/turn`
- 响应类型：`text/event-stream`
- 事件序列总是以 `done` 结束

服务端发送格式：

```text
event: <event_name>
data: <json>

```

### 7.1 事件类型与载荷

#### `route`

```json
{
  "decision": {
    "model_alias": "kimi-k2.5",
    "mode": "chat",
    "reason": "default chat mode (default chat model)"
  }
}
```

#### `step`

```json
{
  "step": {
    "round_index": 0,
    "kind": "tool_call",
    "tool_name": "web_search",
    "summary": "searching latest news",
    "is_error": false
  }
}
```

#### `chunk`

```json
{
  "text": "正在"
}
```

#### `ask_user`

```json
{
  "ask_user": {
    "prompt": "请选择城市",
    "options": [{"label": "上海", "value": "Shanghai", "description": null}],
    "allow_free_text": true,
    "field_name": "city"
  }
}
```

#### `final`

```json
{
  "assistant_message": {
    "role": "assistant",
    "content": "最终回答",
    "created_at": "2026-04-20T10:02:00+00:00",
    "model_alias": "kimi-k2.5",
    "route_reason": "default chat mode (default chat model)",
    "reasoning_content": null,
    "name": null,
    "tool_call_id": null,
    "tool_calls": null,
    "source_profile": null
  }
}
```

#### `error`

```json
{
  "code": "PROVIDER_ERROR",
  "message": "upstream timeout"
}
```

#### `done`

```json
{}
```

### 7.2 流式状态码说明

- 预检查失败（会话不存在、会话 busy）会直接返回 HTTP 错误（`404/409`），不会进入 SSE。
- 生成过程中的异常通过 `error` 事件上报，随后仍会收到 `done`。

## 8. 推荐客户端调用流程

1. `GET /v1/health` 探活。
2. `GET /v1/capabilities` 获取可用模式和 provider 信息。
3. `POST /v1/sessions` 创建会话（或 `GET /v1/sessions` 复用既有会话）。
4. 首选 `POST /turn:stream` 获取增量输出，实时渲染 `chunk`/`step`。
5. 收到 `ask_user` 时，让用户选择并将结果作为下一轮 `text`。
6. 通过 `/messages` 或 `/transcript` 分页拉取历史。

## 9. curl 示例

```bash
export WHESPER_BASE_URL="https://agent.example.com"
export WHESPER_ACCESS_TOKEN="replace-with-issued-token"

# 健康检查
curl "$WHESPER_BASE_URL/v1/health"

# 创建会话
curl -X POST "$WHESPER_BASE_URL/v1/sessions" \
  -H "Authorization: Bearer $WHESPER_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"mobile-user-001"}'

# 非流式对话
curl -X POST "$WHESPER_BASE_URL/v1/sessions/mobile-user-001/turn" \
  -H "Authorization: Bearer $WHESPER_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text":"你好","mode":"auto"}'

# 流式对话（终端直接看 SSE）
curl -N -X POST "$WHESPER_BASE_URL/v1/sessions/mobile-user-001/turn:stream" \
  -H "Authorization: Bearer $WHESPER_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text":"帮我查一下今天AI新闻","mode":"search"}'
```
