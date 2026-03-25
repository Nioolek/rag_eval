# RAG Pipeline API 文档

## 概述

RAG Pipeline 通过 LangGraph API 暴露服务，支持同步和流式调用。

## 基础信息

| 项目 | 值 |
|------|-----|
| 基础 URL | `http://127.0.0.1:8123` |
| 内容类型 | `application/json` |
| Assistant ID | `rag_agent` |

## 端点

### 1. 同步调用 `/runs/wait`

等待执行完成后返回结果。

**请求**

```http
POST /runs/wait HTTP/1.1
Content-Type: application/json

{
  "assistant_id": "rag_agent",
  "input": {
    "query": "iPhone如何截屏"
  }
}
```

**参数**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `assistant_id` | string | 是 | 固定为 `rag_agent` |
| `input.query` | string | 是 | 用户查询 |
| `input.conversation_id` | string | 否 | 会话 ID（用于多轮对话） |
| `input.enable_thinking` | boolean | 否 | 启用思考模式 |
| `input.conversation_history` | array | 否 | 历史对话 |

**响应**

```json
{
  "query": "iPhone如何截屏",
  "conversation_id": "uuid-xxx",
  "answer": "Apple iPhone 截屏方法：同时按住电源键和音量加键...",
  "is_refused": false,

  "query_rewrite": {
    "original_query": "iPhone如何截屏",
    "rewritten_query": "iPhone如何截屏",
    "rewrite_type": "clarification",
    "confidence": 0.8
  },

  "faq_match": {
    "matched": true,
    "faq_id": "faq-0001",
    "question": "iPhone 15 Pro Max如何截屏？",
    "answer": "...",
    "confidence": 0.95,
    "similarity": 0.92,
    "match_type": "semantic"
  },

  "retrieval": [
    {
      "id": "doc-001",
      "content": "相关文档内容...",
      "score": 0.85,
      "metadata": {
        "title": "iPhone 使用指南",
        "category": "使用教程"
      }
    }
  ],

  "rerank": [
    {
      "id": "doc-001",
      "content": "...",
      "original_score": 0.85,
      "rerank_score": 0.92
    }
  ],

  "llm_output": {
    "content": "生成的回答...",
    "thinking": "思考过程（如果启用）",
    "token_usage": {
      "prompt_tokens": 150,
      "completion_tokens": 80,
      "total_tokens": 230
    },
    "model": "qwen-plus"
  },

  "stage_timing": {
    "input_ms": 0.5,
    "faq_match_ms": 15.2,
    "query_rewrite_ms": 2.1,
    "vector_retrieve_ms": 45.3,
    "fulltext_retrieve_ms": 12.4,
    "graph_retrieve_ms": 8.2,
    "merge_ms": 1.2,
    "rerank_ms": 12.1,
    "build_prompt_ms": 0.8,
    "refusal_check_ms": 0.3,
    "generation_ms": 234.5,
    "total_ms": 320.1
  },

  "metadata": {
    "start_time": "2026-03-23T08:00:00",
    "end_time": "2026-03-23T08:00:01"
  },

  "errors": []
}
```

### 2. 流式调用 `/runs/stream`

实时返回执行过程中的事件。

**请求**

```http
POST /runs/stream HTTP/1.1
Content-Type: application/json

{
  "assistant_id": "rag_agent",
  "input": {
    "query": "华为手机续航"
  }
}
```

**响应格式 (SSE)**

```
event: metadata
data: {"run_id": "xxx", "thread_id": "xxx"}

event: values
data: {"query": "华为手机续航", "faq_matched": false, ...}

event: values
data: {"vector_results": [...], "fulltext_results": [...]}

event: values
data: {"answer": "生成的回答...", "is_refused": false}

event: end
data: {}
```

### 3. 健康检查 `/health`

```bash
curl http://127.0.0.1:8123/health
```

**响应**

```json
{
  "status": "healthy"
}
```

### 4. API 信息 `/info`

```bash
curl http://127.0.0.1:8123/info
```

**响应**

```json
{
  "version": "0.7.83",
  "graphs": {
    "rag_agent": {
      "id": "rag_agent",
      "description": "RAG Pipeline for knowledge base Q&A"
    }
  }
}
```

## 输入参数详解

### 基础查询

```json
{
  "assistant_id": "rag_agent",
  "input": {
    "query": "iPhone 15 如何截屏？"
  }
}
```

### 多轮对话

```json
{
  "assistant_id": "rag_agent",
  "input": {
    "query": "那华为呢？",
    "conversation_id": "conv-123",
    "conversation_history": [
      {
        "role": "user",
        "content": "iPhone 15 如何截屏？",
        "timestamp": "2026-03-23T08:00:00"
      },
      {
        "role": "assistant",
        "content": "iPhone 15 截屏方法是...",
        "timestamp": "2026-03-23T08:00:01"
      }
    ]
  }
}
```

### 启用思考模式

```json
{
  "assistant_id": "rag_agent",
  "input": {
    "query": "比较 iPhone 和华为的续航能力",
    "enable_thinking": true
  }
}
```

## 输出字段说明

### 顶层字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `query` | string | 原始查询 |
| `conversation_id` | string | 会话 ID |
| `answer` | string | 最终回答 |
| `is_refused` | boolean | 是否拒答 |
| `retrieval` | array | 检索结果 |
| `rerank` | array | 重排结果 |
| `stage_timing` | object | 各阶段耗时 |
| `errors` | array | 错误信息 |

### FAQ 匹配结果

| 字段 | 类型 | 说明 |
|------|------|------|
| `matched` | boolean | 是否匹配 |
| `faq_id` | string | FAQ ID |
| `question` | string | 匹配的问题 |
| `answer` | string | 预设答案 |
| `confidence` | float | 匹配置信度 |
| `match_type` | string | 匹配类型: exact/semantic |

### 检索结果

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 文档 ID |
| `content` | string | 文档内容 |
| `score` | float | 相关性分数 |
| `metadata` | object | 元数据 |

### 阶段耗时

| 字段 | 说明 |
|------|------|
| `input_ms` | 输入处理耗时 |
| `faq_match_ms` | FAQ 匹配耗时 |
| `query_rewrite_ms` | 查询改写耗时 |
| `vector_retrieve_ms` | 向量检索耗时 |
| `fulltext_retrieve_ms` | 全文检索耗时 |
| `graph_retrieve_ms` | 图谱检索耗时 |
| `merge_ms` | 结果合并耗时 |
| `rerank_ms` | 重排耗时 |
| `generation_ms` | 生成耗时 |
| `total_ms` | 总耗时 |

## 错误处理

### 错误响应格式

```json
{
  "detail": "Error message"
}
```

### 常见错误码

| 状态码 | 说明 |
|--------|------|
| 400 | 请求参数错误 |
| 404 | 资源不存在 |
| 422 | JSON 格式错误 |
| 500 | 服务器内部错误 |

### 检查错误

响应中的 `errors` 数组包含详细错误信息：

```json
{
  "errors": [
    {
      "stage": "faq_match",
      "type": "FAQStoreError",
      "message": "Database connection failed",
      "timestamp": "2026-03-23T08:00:00"
    }
  ]
}
```

## 使用示例

### Python (requests)

```python
import requests

response = requests.post(
    "http://127.0.0.1:8123/runs/wait",
    json={
        "assistant_id": "rag_agent",
        "input": {"query": "iPhone如何截屏"}
    }
)

result = response.json()
print(result["answer"])
print(f"Total time: {result['stage_timing']['total_ms']:.1f}ms")
```

### Python (async with aiohttp)

```python
import aiohttp
import asyncio

async def query_rag(query: str):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://127.0.0.1:8123/runs/wait",
            json={
                "assistant_id": "rag_agent",
                "input": {"query": query}
            }
        ) as response:
            return await response.json()

result = asyncio.run(query_rag("iPhone如何截屏"))
print(result["answer"])
```

### JavaScript (fetch)

```javascript
const response = await fetch('http://127.0.0.1:8123/runs/wait', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    assistant_id: 'rag_agent',
    input: { query: 'iPhone如何截屏' }
  })
});

const result = await response.json();
console.log(result.answer);
```

### cURL

```bash
# 基础查询
curl -X POST http://127.0.0.1:8123/runs/wait \
  -H "Content-Type: application/json" \
  -d '{"assistant_id":"rag_agent","input":{"query":"iPhone如何截屏"}}'

# 多轮对话
curl -X POST http://127.0.0.1:8123/runs/wait \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "rag_agent",
    "input": {
      "query": "那华为呢？",
      "conversation_id": "conv-123",
      "conversation_history": [...]
    }
  }'
```