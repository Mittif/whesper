# Whesper Backend API

Whesper Agent 后端对外提供的 HTTP + SSE 接口规范。客户端（CLI、手机 App、Web、其他服务）均通过本文件定义的接口与后端通信。

- 基础前缀：`/v1`
- 传输：HTTP/1.1 或 HTTP/2，`application/json`
- 流式：Server-Sent Events (`text/event-stream`)
- 时间：一律 ISO-8601，UTC（形如 `2026-04-20T10:00:00+00:00`）
- 字符集：UTF-8
- 认证：`Authorization: Bearer <token>`（除 `/v1/health` 外全部必需）

版本策略：`/v1` 冻结事件名、字段名与枚举值。向后兼容原则——可新增字段和事件，不可删除或改语义。破坏性改动发布 `/v2`。

OpenAPI：后端运行时自动在 `/openapi.json` 与 `/docs` 暴露 schema，可直接据此生成手机端 SDK。

---

## 1. 通用约定

### 1.1 认证

所有请求（除 `/v1/health`）必须携带：

```
Authorization: Bearer <token>
```

Token 由后端从 `whesper.toml` 的 `[server] api_token` 或环境变量 `WHESPER_SERVER_TOKEN` 读取。未携带或错误返回 `401 UNAUTHORIZED`。

### 1.2 错误响应

HTTP 非 2xx 时返回统一结构：

```json
{
  "code": "VALIDATION_ERROR",
  "message": "session_id must be non-empty",
  "detail": { "field": "session_id" }
}
```

错误码枚举（会随版本演进，客户端应兼容未知值）：

| code | HTTP | 含义 |
|---|---|---|
| `UNAUTHORIZED` | 401 | Token 缺失或无效 |
| `NOT_FOUND` | 404 | 会话 / 记忆 / 资源不存在 |
| `VALIDATION_ERROR` | 400 | 请求体字段不合法 |
| `CONFIG_ERROR` | 500 | 后端配置问题（unknown provider 等） |
| `PROVIDER_ERROR` | 502 | 上游模型 provider 失败 |
| `TOOL_EXECUTION_ERROR` | 500 | Agent 工具执行失败 |
| `GENERATION_INTERRUPTED` | 499 | 客户端断开、生成被中断 |
| `CONFLICT` | 409 | 重命名/创建时 id 已存在 |
| `RATE_LIMITED` | 429 | 超过限流（后续版本可能启用） |

### 1.3 分页与范围

列表端点约定查询参数：
- `limit`：默认 50，最大 200
- `before`：游标，ISO 时间戳，返回 `created_at < before` 的记录
- `session_id`：按会话过滤（若接口支持）

返回结构：`{"items": [...], "next_before": "..."|null}`。

### 1.4 字段命名

所有 JSON 字段使用 `snake_case`。布尔字段不加前缀。时间字段后缀一律 `_at`。

---

## 2. 数据模型（Schema）

以下类型在多处引用。

### 2.1 `ChatMessage`

对应 `whesper.session.ChatMessage`。

```jsonc
{
  "role": "user",                          // "user" | "assistant" | "tool" | "system"
  "content": "最近天气怎么样？",
  "created_at": "2026-04-20T10:00:00+00:00",
  "model_alias": "kimi-k2.5",              // null 当 role=user
  "route_reason": "matched search keyword: 最近", // null 当 role=user
  "reasoning_content": null,               // 推理模型的思考内容
  "name": null,                            // role=tool 时为工具名
  "tool_call_id": null,                    // role=tool 时为工具调用 ID
  "tool_calls": null,                      // role=assistant 发起工具调用时填充
  "source_profile": null                   // provider profile 标记
}
```

### 2.2 `AskUserAction`

Agent 需要用户补充信息时产生。对应 `whesper.agent_types.AskUserAction`。

```jsonc
{
  "prompt": "你希望用哪个城市的天气？",
  "options": [
    {"label": "北京", "value": "Beijing", "description": null},
    {"label": "上海", "value": "Shanghai", "description": null}
  ],
  "allow_free_text": true,
  "field_name": "city"
}
```

客户端交互合约：收到 `ask_user` 后，下一次 `turn` 调用应把用户选的 `value`（或自由输入的文本）作为 `text` 字段发回。后端会用它续上之前的轮次。

### 2.3 `RouteDecision`

```jsonc
{
  "model_alias": "kimi-k2.5",
  "mode": "search",           // "auto" | "chat" | "reasoning" | "search"
  "reason": "matched search keyword: 最近 (default chat model)"
}
```

### 2.4 `AgentStep`

Agent 执行循环的一步观测事件。

```jsonc
{
  "round_index": 0,
  "kind": "tool_call",        // "tool_call" | "tool_result" | "planning_retry"
                              // | "error_recovery" | "ask_user" | "final"
  "tool_name": "get_weather_by_location",
  "summary": "calling get_weather_by_location with city=Beijing",
  "is_error": false
}
```

### 2.5 `Session`

列表与详情共享结构；列表形态不包含 `messages` / `transcript_messages`。

```jsonc
{
  "session_id": "main",
  "created_at": "2026-04-20T09:00:00+00:00",
  "updated_at": "2026-04-20T10:00:00+00:00",
  "pinned_model": "auto",
  "message_count": 42,
  "pending_ask_user": null,         // AskUserAction | null，仅详情返回
  "messages": [ChatMessage],         // 仅 GET /sessions/{id} 详情返回
  "transcript_messages": [ChatMessage] // 仅详情返回
}
```

### 2.6 `MemoryItem`

对应 `whesper.memory.MemoryItem`。

```jsonc
{
  "memory_id": "profile-3f2a4b1c0d",
  "memory_type": "profile",
  "title": "User prefers jasmine tea",
  "content": "I like jasmine tea",
  "source": "manual",            // "manual" | "auto"
  "confidence": 1.0,
  "created_at": "...",
  "updated_at": "...",
  "last_confirmed_at": null,
  "expires_at": null,
  "session_id": "main"
}
```

### 2.7 `TraceEvent`

对应 `whesper.trace.TraceEvent`。

```jsonc
{
  "timestamp": "2026-04-20T10:00:00+00:00",
  "kind": "completion_request",
  "session_id": "main",
  "model_alias": "kimi-k2.5",
  "provider_name": "kimi",
  "route_mode": "search",
  "streamed": true,
  "tools_enabled": true,
  "tool_choice": "auto",
  "note": null,
  "preview": "..."
}
```

### 2.8 `ModelInfo`

```jsonc
{
  "alias": "kimi-k2.5",
  "provider": "kimi",
  "model": "moonshot-v1-8k",
  "tags": ["reasoning"],
  "max_tokens": 4096
}
```

### 2.9 `ProviderInfo`

```jsonc
{
  "name": "kimi",
  "kind": "openai",           // "openai" | "ollama" | ...
  "base_url": "https://api.moonshot.cn/v1",
  "has_api_key": true
}
```

### 2.10 `CupStatus`

```jsonc
{
  "connected": true,
  "device": {"name": "ESP32-CUP", "firmware": "0.1.0"},
  "motor": {"enabled": true, "target_velocity": 40.0},
  "led": {"color": "#ffb36b", "blink_hz": 0.0, "blink_mode": 0}
}
```

---

## 3. 端点

### 3.1 健康与能力

#### `GET /v1/health`

无需认证。

**200**
```json
{"ok": true, "version": "0.1.0"}
```

#### `GET /v1/capabilities`

**200**
```json
{
  "version": "0.1.0",
  "features": ["tools", "memory", "cup", "trace", "search"],
  "route_modes": ["auto", "chat", "reasoning", "search"],
  "websearch_providers": ["duckduckgo", "brave", "serpapi"]
}
```

---

### 3.2 会话

#### `GET /v1/sessions`

列出已保存会话。

**查询参数**
- `limit` (int, 默认 50)

**200**
```json
{
  "items": [
    {
      "session_id": "main",
      "created_at": "...",
      "updated_at": "...",
      "pinned_model": "auto",
      "message_count": 42,
      "pending_ask_user": null
    }
  ],
  "next_before": null
}
```

#### `POST /v1/sessions`

创建新会话。若 `session_id` 已存在返回 `409 CONFLICT`。

**请求**
```json
{"session_id": "work-2026"}
```

**201** 返回 `Session`（不含 messages）。

#### `GET /v1/sessions/{session_id}`

返回会话详情，含完整 `messages` 与 `transcript_messages`。

**查询参数（可选）**
- `message_limit` (int, 默认 200)：只返回最近 N 条
- `include_transcript` (bool, 默认 true)

**200** 返回完整 `Session`。
**404** 会话不存在。

#### `PATCH /v1/sessions/{session_id}`

更新会话元数据。支持字段：`session_id`（重命名）、`pinned_model`。

**请求**
```json
{"session_id": "work-renamed", "pinned_model": "auto"}
```

**200** 返回更新后的 `Session`（不含 messages）。
**409** 新 id 已存在。

#### `DELETE /v1/sessions/{session_id}`

**200**
```json
{"ok": true}
```

#### `GET /v1/sessions/{session_id}/messages`

按时间倒序返回会话消息。

**查询参数**
- `limit` (int, 默认 50)
- `before` (ISO timestamp)

**200**
```json
{"items": [ChatMessage, ...], "next_before": "..."|null}
```

#### `GET /v1/sessions/{session_id}/transcript`

返回净化后的转录（仅 user/assistant 纯文本消息，用于展示）。

同 `messages` 接口。

---

### 3.3 对话轮次

这是核心接口。两种模式：非流式（简单）与 SSE 流式（推荐）。

#### 3.3.1 非流式 — `POST /v1/sessions/{session_id}/turn`

**请求**
```json
{
  "text": "最近北京天气怎么样？",
  "mode": "auto"
}
```

字段说明：
- `text`（string，必填）：用户输入；若会话处于 `pending_ask_user` 状态，此字段作为对 ask_user 的回答
- `mode`（string，可选，默认 `"auto"`）：路由模式覆盖，枚举 `auto|chat|reasoning|search`

**200**
```json
{
  "decision": {"model_alias": "kimi-k2.5", "mode": "search", "reason": "..."},
  "assistant_message": ChatMessage,
  "ask_user": null
}
```

若 Agent 选择追问，则：
```json
{
  "decision": RouteDecision,
  "assistant_message": ChatMessage,  // content = ask_user.prompt
  "ask_user": AskUserAction
}
```

错误：
- `404` 会话不存在
- `502 PROVIDER_ERROR` 上游失败
- `500 TOOL_EXECUTION_ERROR` 工具执行失败

#### 3.3.2 流式 — `POST /v1/sessions/{session_id}/turn:stream`

请求体同 3.3.1。响应 `Content-Type: text/event-stream`。

**SSE 事件类型（v1 冻结）**

| event | data | 说明 |
|---|---|---|
| `route` | `{"decision": RouteDecision}` | 首个事件，路由决策就位 |
| `step` | `{"step": AgentStep}` | Agent 执行一步（工具调用、重规划等） |
| `chunk` | `{"text": "token"}` | 流式文本片段，按顺序拼接 |
| `ask_user` | `{"ask_user": AskUserAction}` | 需要用户补充；本轮到此结束 |
| `final` | `{"assistant_message": ChatMessage}` | 完整最终消息（权威值） |
| `error` | `{"code": "...", "message": "...", "detail": {}}` | 错误，流结束 |
| `done` | `{}` | 终止标记，客户端关闭连接 |

**顺序保证**
1. 第一条必定是 `route`
2. 多个 `step` / `chunk` 可以任意穿插
3. 以 `ask_user` 或 `final` 或 `error` 之一作为业务结束
4. 最后必定有一条 `done`

**示例流**
```
event: route
data: {"decision":{"model_alias":"kimi-k2.5","mode":"search","reason":"matched keyword"}}

event: step
data: {"step":{"round_index":0,"kind":"tool_call","tool_name":"get_weather_by_location","summary":"...","is_error":false}}

event: step
data: {"step":{"round_index":0,"kind":"tool_result","tool_name":"get_weather_by_location","summary":"ok","is_error":false}}

event: chunk
data: {"text":"北京今天"}

event: chunk
data: {"text":"晴转多云，"}

event: final
data: {"assistant_message":{"role":"assistant","content":"北京今天晴转多云，...","created_at":"..."}}

event: done
data: {}
```

**客户端合约**
- `chunk` 顺序拼接 = 流式文本，可用于低延迟 UI 渲染
- 但**权威内容以 `final.assistant_message.content` 为准**。SSE 可能丢帧；UI 在收到 `final` 后应以它覆盖累积文本
- 收到 `ask_user` 后应停止累积，提示用户从 `options` 选一项（或自由输入），下一次 `turn` 将该 `value` 作为 `text` 发回
- 客户端主动断开连接等同于取消本轮生成，后端会尽力保存 partial 结果

#### 3.3.3 取消 — `POST /v1/sessions/{session_id}/turn:cancel`

主动取消正在进行中的轮次。幂等。

**200**
```json
{"ok": true, "cancelled": true}
```

如无进行中的轮次返回 `{"ok": true, "cancelled": false}`。

#### 3.3.4 重试 — `POST /v1/sessions/{session_id}/retry` / `retry:stream`

重新生成上一条 assistant 回复。无请求体（沿用会话里最后一条 user 消息）。

**请求（可选）**
```json
{"mode": "reasoning"}
```

响应结构同 `turn` / `turn:stream`。

若没有可重试的 user 消息返回 `400 VALIDATION_ERROR`。

---

### 3.4 模式覆盖（按会话）

Mode override 是会话级状态，持久化在 `SessionStore`。

#### `GET /v1/sessions/{session_id}/mode`

**200**
```json
{"mode": "auto"}
```

#### `PUT /v1/sessions/{session_id}/mode`

**请求**
```json
{"mode": "reasoning"}
```

**200**
```json
{"mode": "reasoning"}
```

---

### 3.5 模型与 Provider

#### `GET /v1/models`

列出配置文件中定义的所有模型别名。

**200**
```json
{
  "items": [
    {"alias":"local_chat","provider":"ollama","model":"qwen3:14b","tags":[],"max_tokens":null},
    {"alias":"kimi-k2.5","provider":"kimi","model":"moonshot-v1-8k","tags":["reasoning"],"max_tokens":4096}
  ]
}
```

#### `GET /v1/providers`

**200**
```json
{
  "items": [
    {"name":"ollama","kind":"ollama","base_url":"http://localhost:11434","has_api_key":false},
    {"name":"kimi","kind":"openai","base_url":"https://api.moonshot.cn/v1","has_api_key":true}
  ]
}
```

#### `GET /v1/providers/{provider_name}/models`

从 provider 动态查询可用模型列表（相当于 `/models?provider=...`）。

**200**
```json
{"provider": "kimi", "model_ids": ["moonshot-v1-8k", "moonshot-v1-32k"]}
```

错误：`404` provider 不存在；`502 PROVIDER_ERROR` 查询失败。

---

### 3.6 记忆

#### `GET /v1/memories`

**查询参数**
- `limit` (int, 默认 50)
- `include_expired` (bool, 默认 false)
- `session_id` (可选，按会话过滤)

**200**
```json
{"items": [MemoryItem, ...]}
```

#### `POST /v1/memories`

手动记忆（对应 `/remember`）。

**请求**
```json
{"content": "我喜欢茉莉花茶", "session_id": "main"}
```

**201** 返回 `MemoryItem`。

#### `DELETE /v1/memories/{memory_id}`

**200**
```json
{"ok": true}
```

**404** 记忆不存在。

---

### 3.7 Trace

#### `GET /v1/trace`

**查询参数**
- `session_id` (可选)
- `limit` (int, 默认 20, 最大 500)

**200**
```json
{"items": [TraceEvent, ...]}
```

---

### 3.8 Web 搜索 Provider

#### `GET /v1/websearch`

**200**
```json
{"provider": "duckduckgo", "available": ["duckduckgo", "brave", "serpapi"]}
```

#### `PUT /v1/websearch`

**请求**
```json
{"provider": "brave"}
```

**200**
```json
{"provider": "brave"}
```

**400** provider 不在允许列表。

---

### 3.9 CUP 硬件

硬件命名空间独立于对话，供客户端直接读状态或下指令。

#### `GET /v1/cup/status`

**200** 返回 `CupStatus`。若硬件未连接：
```json
{"connected": false, "device": null, "motor": null, "led": null}
```

#### `POST /v1/cup/control`

统一入口，由 `action` 分派。

**请求**（按 action 取字段）

| action | 必填字段 | 可选字段 |
|---|---|---|
| `speed` | `velocity` (float) | — |
| `stop` | — | — |
| `scene` | `scene` (`gentle\|steady\|intense\|cooldown`) | — |
| `intensity` | `direction` (`up\|down`) | `step` (int) |
| `led` | `color` (hex `#rrggbb` 或预设名) | `blink_hz` (float), `blink_mode` (int) |

示例：
```json
{"action": "scene", "scene": "gentle"}
```

**200**
```json
{"ok": true, "result": {"applied": "gentle", "motor_target": 40.0}}
```

**错误**
- `400 VALIDATION_ERROR` action 字段缺失或非法
- `500 TOOL_EXECUTION_ERROR` 硬件通信失败（含离线）

---

### 3.10 服务器事件流（可选，推送类通知）

#### `GET /v1/events:stream`

长连接 SSE，后端主动推送与具体请求无关的事件（如 CUP 状态变更）。

**事件类型**

| event | data | 说明 |
|---|---|---|
| `cup_status` | `CupStatus` | CUP 状态快照（连接/断开/参数变更时触发） |
| `ping` | `{"ts": "..."}` | 心跳，客户端忽略即可 |

客户端可自行决定是否订阅。若不订阅，改为轮询 `/v1/cup/status`。

---

## 4. 并发与一致性

1. **会话级互斥**：同一个 `session_id` 同一时刻只允许一个正在进行的 `turn` / `retry`。第二个请求会返回 `409 CONFLICT`，`detail.reason="session_busy"`。客户端应先 `turn:cancel` 再发新请求。
2. **跨会话并发**：不同 `session_id` 之间完全独立，可并发。
3. **消息持久化**：一次 `turn` 成功后（包括进入 `ask_user`），会话状态必定已落盘。客户端可立即重载。
4. **取消与 partial**：SSE 过程中客户端主动断开，后端收到 `CancelledError`；若已产生部分 assistant 文本，该条会以 partial 内容保存（`route_reason` 追加 `interrupted` 备注）。客户端下一次 `GET /sessions/{id}` 能看到。

---

## 5. 示例流程

### 5.1 最小会话

```
POST /v1/sessions {"session_id":"main"}                -> 201
POST /v1/sessions/main/turn {"text":"你好","mode":"auto"}
  -> 200 {"decision":...,"assistant_message":{...,"content":"你好！..."}}
```

### 5.2 流式 + 追问

```
POST /v1/sessions/main/turn:stream  {"text":"最近天气怎么样？","mode":"auto"}
  event: route   data: {...}
  event: step    data: {"step":{"kind":"tool_call","tool_name":"get_local_weather",...}}
  event: step    data: {"step":{"kind":"ask_user",...}}
  event: ask_user data: {"ask_user":{"prompt":"具体哪个城市？","options":[{"label":"北京","value":"Beijing"}]}}
  event: done

# 用户选"北京"后
POST /v1/sessions/main/turn:stream  {"text":"Beijing","mode":"auto"}
  event: route   data: {...}
  event: step    data: {"step":{"kind":"tool_call","tool_name":"get_weather_by_location",...}}
  event: chunk   data: {"text":"北京今天"}
  event: chunk   data: {"text":"晴转多云..."}
  event: final   data: {"assistant_message":{...}}
  event: done
```

### 5.3 重试

```
POST /v1/sessions/main/retry:stream {"mode":"reasoning"}
  # 流式结构同 turn:stream
```

### 5.4 手动保存记忆

```
POST /v1/memories {"content":"我喜欢茉莉花茶","session_id":"main"}
  -> 201 {"memory_id":"profile-3f2a4b1c0d",...}
```

---

## 6. 版本与演进

- 新增字段不触发版本升级，客户端必须忽略未知字段
- 新增 SSE 事件类型不触发版本升级，客户端必须忽略未知事件
- 删除字段、更改字段语义、收紧枚举值、重命名事件均为破坏性改动，须发布 `/v2`
- 错误码 `code` 字段为开放枚举，新增不视为破坏性

OpenAPI schema 以 `/openapi.json` 为权威源；本文档与之出现冲突时以本文档为准（文档会同步更新）。
