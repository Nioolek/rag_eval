# RAG Pipeline Service

企业级 RAG (Retrieval-Augmented Generation) 服务，基于 LangGraph 构建，为知识库问答场景提供高性能、可扩展的解决方案。

## 目录

- [功能特性](#功能特性)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [配置说明](#配置说明)
- [API 使用](#api-使用)
- [数据注入](#数据注入)
- [集成评测系统](#集成评测系统)
- [开发指南](#开发指南)
- [故障排除](#故障排除)

## 功能特性

### 核心能力

- **LangGraph StateGraph**: 14 节点流水线，支持并行检索
- **多源检索**: Chroma (向量)、Whoosh (全文)、Neo4j (知识图谱)
- **FAQ 优先策略**: 常见问题直接匹配，跳过检索流程
- **查询改写**: LLM 驱动的查询扩展和澄清
- **智能重排**: 阿里云 gte-rerank，BM25 降级备选
- **拒答逻辑**: 域外问题、敏感内容、低相关性检测
- **降级策略**: 熔断器 + 回退处理器

### 性能特点

- **并行检索**: Vector/Fulltext/Graph 三路并行，降低延迟
- **热更新配置**: YAML 配置 + watchdog 自动重载
- **完整 Timing**: 各阶段耗时追踪，便于性能分析

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG Pipeline Flow                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Query ──→ Input ──→ FAQ Match ──┬──→ Answer FAQ ──┐          │
│                                    │                  │          │
│                                    └──→ Query Rewrite │          │
│                                              │       │          │
│                    ┌─────────────────────────┼───────┘          │
│                    │                         │                   │
│                    ▼           ▼             ▼                   │
│              Vector      Fulltext       Graph                    │
│              Retrieve    Retrieve       Retrieve                 │
│                    │           │             │                   │
│                    └───────────┴─────────────┘                   │
│                                │                                 │
│                                ▼                                 │
│                              Merge                               │
│                                │                                 │
│                                ▼                                 │
│                             Rerank                               │
│                                │                                 │
│                                ▼                                 │
│                         Build Prompt                             │
│                                │                                 │
│                                ▼                                 │
│                        Refusal Check ──┬──→ Refuse               │
│                                │       │                         │
│                                └──→ Generate                     │
│                                        │                         │
│                                        ▼                         │
│                                      Output                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 快速开始

### 环境要求

- Python 3.13+
- pip 或 uv 包管理器
- **阿里云 DashScope API Key** (用于向量检索和 LLM 生成)

### 检索方式说明

| 检索方式 | 是否需要 API Key | 说明 |
|----------|------------------|------|
| FAQ 匹配 | ❌ 不需要 | SQLite 存储，关键词匹配 |
| 全文检索 | ❌ 不需要 | Whoosh BM25，支持中文 |
| 向量检索 | ✅ 需要 | 阿里云 text-embedding-v3 |
| LLM 生成 | ✅ 需要 | 阿里云 Qwen |

**注意**: 如果不配置 API Key，系统仍可运行，但：
- 向量检索将返回空结果
- LLM 生成将返回占位回复
- 建议至少配置 `DASHSCOPE_API_KEY` 以获得完整功能

### 安装步骤

```bash
# 1. 进入项目目录
cd rag_rag

# 2. 安装依赖
pip install -r requirements.txt

# 3. 安装项目包
pip install -e .

# 4. 创建环境配置
cp .env.example .env

# 5. 编辑 .env 配置 API 密钥
# DASHSCOPE_API_KEY=your-api-key
```

### 启动服务

```bash
# 开发模式（带热重载）
langgraph dev --port 8123 --allow-blocking

# 或者后台运行
langgraph dev --port 8123 --allow-blocking &
```

### 验证服务

```bash
# 健康检查
curl http://127.0.0.1:8123/health

# 测试查询
curl -X POST http://127.0.0.1:8123/runs/wait \
  -H "Content-Type: application/json" \
  -d '{"assistant_id":"rag_agent","input":{"query":"iPhone如何截屏"}}'
```

## 项目结构

```
rag_rag/
├── langgraph.json           # LangGraph API 配置
├── pyproject.toml           # 包元数据
├── requirements.txt         # 依赖列表
├── .env.example            # 环境变量模板
├── config/
│   └── settings.yaml       # 业务配置
│
├── src/rag_rag/
│   ├── core/               # 基础层
│   │   ├── config.py       # 配置管理
│   │   ├── exceptions.py   # 异常体系
│   │   ├── logging.py      # 日志工具
│   │   └── constants.py    # 常量定义
│   │
│   ├── graph/              # LangGraph 核心
│   │   ├── state.py        # RAGState 状态定义
│   │   ├── graph.py        # Graph 构建与编译
│   │   ├── routers.py      # 条件路由函数
│   │   └── nodes/          # 14个节点实现
│   │       ├── input_node.py
│   │       ├── faq_match_node.py
│   │       ├── query_rewrite_node.py
│   │       ├── vector_retrieve_node.py
│   │       ├── fulltext_retrieve_node.py
│   │       ├── graph_retrieve_node.py
│   │       ├── merge_node.py
│   │       ├── rerank_node.py
│   │       ├── build_prompt_node.py
│   │       ├── refusal_check_node.py
│   │       ├── generate_node.py
│   │       ├── output_node.py
│   │       ├── answer_faq_node.py
│   │       └── refuse_node.py
│   │
│   ├── storage/            # 存储层
│   │   ├── base.py         # 抽象接口
│   │   ├── faq_store.py    # SQLite FAQ
│   │   ├── vector_store.py # Chroma 向量
│   │   ├── fulltext_store.py # Whoosh 全文
│   │   ├── graph_store.py  # Neo4j 图谱
│   │   └── session_store.py # SQLite 会话
│   │
│   ├── services/           # 外部服务
│   │   ├── llm_service.py      # 阿里云 Qwen
│   │   ├── embedding_service.py # text-embedding-v3
│   │   ├── rerank_service.py   # gte-rerank
│   │   └── sensitive_filter.py # 敏感词过滤
│   │
│   ├── degradation/        # 降级策略
│   │   ├── circuit_breaker.py
│   │   ├── fallback_handlers.py
│   │   └── degradation_manager.py
│   │
│   ├── prompts/            # 提示词
│   │   ├── template_manager.py
│   │   └── templates/
│   │       ├── default.yaml
│   │       ├── thinking.yaml
│   │       └── refusal.yaml
│   │
│   └── ingestion/          # 数据摄入
│       ├── document_chunker.py
│       ├── entity_extractor.py
│       └── ingestion_pipeline.py
│
├── scripts/                # 工具脚本
│   ├── generate_electronics_data.py  # 数据生成
│   ├── ingest_electronics_data.py    # 数据注入
│   └── test_pipeline.py              # 管道测试
│
├── data/                   # 数据目录
│   ├── faq.db             # FAQ 数据库
│   ├── chroma/            # 向量索引
│   └── whoosh/            # 全文索引
│
├── tests/                  # 测试用例
│   ├── test_storage/
│   ├── test_services/
│   ├── test_graph/
│   └── test_integration/
│
└── docs/                   # 文档
    ├── API.md
    ├── CONFIGURATION.md
    ├── DATA_INGESTION.md
    └── TROUBLESHOOTING.md
```

## 配置说明

详细配置请参考 [docs/CONFIGURATION.md](docs/CONFIGURATION.md)

### 环境变量 (.env)

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DASHSCOPE_API_KEY` | 阿里云 DashScope API Key | 必填 |
| `LLM_MODEL` | LLM 模型名称 | qwen-plus |
| `EMBEDDING_MODEL` | Embedding 模型 | text-embedding-v3 |
| `EMBEDDING_DIMENSION` | 向量维度 | 1024 |
| `RERANK_MODEL` | Rerank 模型 | gte-rerank |
| `DATA_DIR` | 数据存储目录 | ./data |

### 业务配置 (config/settings.yaml)

```yaml
llm:
  model: "qwen-plus"
  temperature: 0.7
  max_tokens: 2048

retrieval:
  vector_top_k: 20      # 向量检索数量
  fulltext_top_k: 20    # 全文检索数量
  graph_top_k: 10       # 图谱检索数量
  vector_weight: 0.5    # 向量权重
  fulltext_weight: 0.3  # 全文权重
  graph_weight: 0.2     # 图谱权重

faq:
  match_threshold: 0.85  # FAQ 匹配阈值
```

## API 使用

详细 API 文档请参考 [docs/API.md](docs/API.md)

### 同步调用

```bash
curl -X POST http://127.0.0.1:8123/runs/wait \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "rag_agent",
    "input": {
      "query": "iPhone如何截屏"
    }
  }'
```

### 流式调用

```bash
curl -X POST http://127.0.0.1:8123/runs/stream \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "rag_agent",
    "input": {
      "query": "华为手机续航怎么样"
    }
  }'
```

### 响应格式

```json
{
  "query": "iPhone如何截屏",
  "answer": "同时按住电源键和音量加键...",
  "is_refused": false,
  "faq_match": {
    "matched": true,
    "question": "iPhone 15 Pro Max如何截屏？",
    "answer": "...",
    "confidence": 0.95
  },
  "retrieval": [...],
  "rerank": [...],
  "stage_timing": {
    "faq_match_ms": 15.2,
    "retrieval_ms": 45.3,
    "rerank_ms": 12.1,
    "generation_ms": 234.5,
    "total_ms": 320.1
  }
}
```

## 数据注入

详细说明请参考 [docs/DATA_INGESTION.md](docs/DATA_INGESTION.md)

### 快速注入

```bash
# 生成并注入电子产品数据
cd rag_rag
python scripts/ingest_electronics_data.py --generate

# 仅注入已有数据
python scripts/ingest_electronics_data.py
```

### 数据格式

**FAQ 数据** (`scripts/data/generated/faqs.json`):
```json
{
  "id": "faq-001",
  "question": "iPhone 15 如何截屏？",
  "answer": "同时按住侧边按钮和音量加键...",
  "category": "手机操作",
  "keywords": ["iPhone", "截屏"]
}
```

**文档数据** (`scripts/data/generated/documents.json`):
```json
{
  "id": "doc-001",
  "title": "iPhone 15 Pro 使用指南",
  "content": "完整的产品说明...",
  "category": "使用教程",
  "product": "iPhone 15 Pro",
  "brand": "Apple"
}
```

## 集成评测系统

本 RAG 服务输出格式与 `rag_eval` 评测系统完全兼容。

### 配置连接

在 `rag_eval` 项目的 `.env` 中：

```bash
RAG_SERVICE_URL=http://localhost:8123
```

### 使用 LangGraphAdapter

```python
from src.rag.langgraph_adapter import LangGraphAdapter

adapter = LangGraphAdapter(base_url="http://localhost:8123")
response = await adapter.query("iPhone如何截屏")

print(response.final_answer)
print(response.stage_timing)
```

### 输出字段映射

| RAG 输出 | RAGResponse 字段 |
|----------|------------------|
| `answer` | `final_answer` |
| `faq_match.matched` | `faq_match.matched` |
| `faq_match.question` | `faq_match.faq_question` |
| `retrieval[]` | `retrieval_results` |
| `is_refused` | `is_refused` |
| `stage_timing` | `stage_timing` |

## 开发指南

### 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行单个测试文件
pytest tests/test_storage/test_faq_store.py -v

# 带覆盖率
pytest tests/ --cov=src/rag_rag --cov-report=html
```

### 添加新节点

1. 在 `src/rag_rag/graph/nodes/` 创建节点文件
2. 使用 `@node_decorator("node_name")` 装饰器
3. 在 `graph.py` 中注册节点和边

### 添加新存储

1. 继承 `storage/base.py` 中的抽象类
2. 实现必要的方法 (`initialize`, `add`, `search`, `close`)
3. 在节点中注入使用

## 故障排除

常见问题请参考 [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

### 常见问题

**Q: 启动时报 "No module named 'rag_rag'"**

```bash
pip install -e .
```

**Q: API 返回 "Blocking call" 警告**

```bash
langgraph dev --port 8123 --allow-blocking
```

**Q: FAQ 匹配不生效**

检查 `data/faq.db` 是否存在，运行数据注入脚本。

## License

MIT License