# HTTP API 文档

Base URL: `http://esp32-cup.local` 或 `http://<IP>`

所有响应均为 JSON，格式：

```json
{"ok": true, "data": {...}}
```

或：

```json
{"ok": false, "error": "描述"}
```

## 认证

当设备配置了 `api_token` 时，需使用以下任一方式鉴权：

- `Authorization: Bearer <token>`
- `?token=<token>`

当 `api_token` 为空字符串时，跳过鉴权。

## `GET /api/status`

查询系统状态。

响应示例：

```json
{
  "ok": true,
  "data": {
    "uptime_s": 3600,
    "ip": "192.168.1.42",
    "wifi_state": "connected",
    "rssi": -58,
    "ota": {
      "running": "ota_0",
      "boot": "ota_0",
      "next": "ota_1",
      "version": "1.0.0",
      "date": "Mar 31 2026",
      "time": "12:00:00"
    },
    "free_heap": 186432,
    "min_free_heap": 172048
  }
}
```

## `GET /api/config`

查询所有持久化配置。密码字段会脱敏为 `"***"`。

## `DELETE /api/config`

恢复出厂设置，并触发 `esp_restart()`。

## `POST /api/led`

设置 LED 颜色和闪烁模式。

请求体：

```json
{
  "color": "#FF0000",
  "blink_hz": 1.0,
  "blink_mode": 1
}
```

所有字段均为可选，仅提供需要修改的字段。

`blink_mode` 枚举：

| 值 | 说明 |
|---|---|
| `0` | 常亮 |
| `1` | 方波慢闪 |
| `2` | 方波快闪 |
| `3` | 呼吸效果 |

## `GET /api/motor`

返回实时遥测数据。

响应示例：

```json
{
  "ok": true,
  "data": {
    "vel": 98.5,
    "target": 100.0,
    "volt": 12.3,
    "curr": 0.42,
    "ok": true
  }
}
```

## `POST /api/motor`

设置目标转速和启停状态。

请求体：

```json
{
  "target_velocity": 100.0,
  "enabled": true
}
```

## `POST /api/motor/stop`

紧急停止。等效于设置 `target_velocity=0`、`enabled=false`，同时触发 `foc_emergency_stop()`。

## `POST /api/ota`

流式 OTA 固件上传。请求体为原始 `.bin` 文件，推荐 `Content-Type: application/octet-stream`。

- 支持并发保护，同一时间只能有一个 OTA 进行
- 上传完成后自动验证 `ESP_APP_DESC_MAGIC_WORD`
- 验证通过后调用 `esp_restart()` 切换到新分区

示例：

```bash
curl -X POST http://esp32-cup.local/api/ota \
  --data-binary @build/CUPMain.bin \
  -H "Content-Type: application/octet-stream" \
  -H "Authorization: Bearer mytoken"
```

## `GET /api/ota/status`

查询 OTA 分区状态。

响应示例：

```json
{
  "ok": true,
  "data": {
    "running": "ota_0",
    "boot": "ota_0",
    "next": "ota_1",
    "version": "1.0.0",
    "img_state": 1,
    "last_ota_bytes": 1032192,
    "ota_in_progress": false
  }
}
```
