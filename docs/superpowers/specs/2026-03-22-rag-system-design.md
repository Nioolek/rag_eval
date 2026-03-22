# RAG系统设计文档

基于LangGraph和LangGraph API的企业级RAG系统设计，用于知识库问答场景，支持评测系统进行全流程评测。

## 1. 项目概述

### 1.1 背景

本项目为RAG评测系统提供被评测的RAG服务。系统需支持完整的RAG Pipeline，输出各阶段结果供评测系统分析。

### 1.2 核心需求

| 维度 | 需求 |
|------|------|
| **场景** | 企业知识库问答 |
| **数据** | FAQ + 长文档 + 图片视频元数据，数据量十几万 |
| **检索架构** | FAQ优先 → Query改写 → 3源并行检索(向量+全文+图谱) → 融合重排 |
| **拒答策略** | 知识库外 + 敏感问题 + 多条件组合，规则+LLM判断 |
| **思考模式** | 可开关的CoT推理 |
| **多轮对话** | SQLite持久化，固定轮数历史，支持query改写 |
| **部署方式** | LangGraph API |

### 1.3 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| 向量库 | Chroma | 纯Python、无依赖、嵌入模式 |
| 全文检索 | Whoosh | 纯Python库、无需服务 |
| 知识图谱 | Neo4j Community | Docker部署、成熟稳定 |
| LLM | 阿里云通义千问 (Qwen) | qwen-plus/qwen-turbo/qwen-max |
| Embedding | 阿里云 text-embedding-v3 | 支持中英文 |
| Rerank | 阿里云 gte-rerank | 重排序服务 |

---

## 2. 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph API Server                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   RAG Pipeline Graph                      │   │
│  │                                                           │   │
│  │   Input → FAQ Match ─┬─→ Query Rewrite                   │   │
│  │          │           │         │                          │   │
│  │          └─(命中)────┘         ├─→ Vector Retrieve        │   │
│  │                                ├─→ Fulltext Retrieve      │   │
│  │                                └─→ Graph Retrieve         │   │
│  │                                           │               │   │
│  │                                    Merge ←─┘               │   │
│  │                                       │                    │   │
│  │                                    Rerank                   │   │
│  │                                       │                    │   │
│  │                                 Build Prompt               │   │
│  │                                       │                    │   │
│  │                                 Refusal Check              │   │
│  │                                    │    │                  │   │
│  │                           Generate ←┘    └─→ Refuse        │   │
│  │                                │                           │   │
│  │                              Output                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │    Chroma    │ │    Whoosh    │ │    Neo4j     │            │
│  │  (Vector)    │ │  (Fulltext)  │ │   (Graph)    │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           Alibaba Cloud Services                          │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐                   │   │
│  │  │   LLM   │  │Embedding│  │ Rerank  │                   │   │
│  │  │ (Qwen)  │  │  (v3)   │  │  (gte)  │                   │   │
│  │  └─────────┘  └─────────┘  └─────────┘                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流图

```
用户Query
    │
    ▼
┌─────────────┐
│   Input     │ 验证输入、加载会话历史
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│  FAQ Match  │────▶│  Answer FAQ │ (命中分支)
└──────┬──────┘     └──────┬──────┘
       │ 未命中            │
       ▼                   │
┌─────────────┐            │
│Query Rewrite│            │
└──────┬──────┘            │
       │                   │
       ├───────────────────┤
       ▼                   │
┌─────────────────────────┐│
│   Parallel Retrieval    ││
│ ┌─────┐ ┌─────┐ ┌─────┐ ││
│ │ Vec │ │ FT  │ │ KG  │ ││
│ └──┬──┘ └──┬──┘ └──┬──┘ ││
└────┼───────┼───────┼────┘│
     └───────┼───────┘     │
             ▼             │
      ┌─────────────┐      │
      │    Merge    │      │
      └──────┬──────┘      │
             │             │
             ▼             │
      ┌─────────────┐      │
      │   Rerank    │      │
      └──────┬──────┘      │
             │             │
             ▼             │
      ┌─────────────┐      │
      │Build Prompt │      │
      └──────┬──────┘      │
             │             │
             ▼             │
      ┌─────────────┐      │
      │Refusal Check│      │
      └──────┬──────┘      │
         ┌───┴───┐         │
         ▼       ▼         │
    ┌────────┐ ┌───────┐   │
    │Generate│ │Refuse │   │
    └───┬────┘ └───┬───┘   │
        │          │       │
        └────┬─────┘       │
             ▼             │
      ┌─────────────┐◀─────┘
      │   Output    │
      └─────────────┘
             │
             ▼
         最终回复
```

---

## 3. State 状态定义

State是LangGraph图执行过程中传递的数据结构，定义了各阶段的输入输出。

```python
class RAGState(TypedDict):
    """RAG Pipeline 状态定义"""

    # === 输入层 ===
    query: str                              # 用户原始问题
    conversation_history: list[dict]        # 对话历史
    conversation_id: str                    # 会话ID
    agent_id: str                           # Agent标识
    enable_thinking: bool                   # 是否开启思考模式

    # === FAQ匹配阶段 ===
    faq_matched: bool
    faq_result: Optional[dict]

    # === Query改写阶段 ===
    original_query: str
    rewritten_query: str
    rewrite_type: str                       # expansion/clarification/multi_turn
    rewrite_confidence: float

    # === 检索阶段 ===
    vector_results: list[dict]
    fulltext_results: list[dict]
    graph_results: list[dict]
    merged_results: list[dict]

    # === 重排序阶段 ===
    reranked_results: list[dict]
    rerank_scores: list[float]

    # === 拒答判断 ===
    should_refuse: bool
    refusal_reason: str
    refusal_type: str                       # out_of_domain/sensitive/low_relevance

    # === 提示词组装 ===
    system_prompt: str
    context_prompt: str
    few_shot_examples: list[dict]
    final_prompt: str
    prompt_template_name: str

    # === 生成阶段 ===
    thinking_process: str
    final_answer: str
    token_usage: dict

    # === 元数据 ===
    stage_timing: dict[str, float]          # 各阶段耗时
    metadata: dict
    errors: list[str]
```

---

## 4. 节点设计

### 4.1 节点列表

| 序号 | 节点名称 | 功能 | 输入 | 输出 |
|------|----------|------|------|------|
| 1 | input_node | 验证输入、初始化、加载历史 | query, conversation_id | conversation_history |
| 2 | faq_match_node | FAQ精确/语义匹配 | query | faq_matched, faq_result |
| 3 | query_rewrite_node | 多轮改写/扩展/澄清 | query, history | rewritten_query |
| 4a | vector_retrieve_node | Chroma向量检索 | rewritten_query | vector_results |
| 4b | fulltext_retrieve_node | Whoosh全文检索 | rewritten_query | fulltext_results |
| 4c | graph_retrieve_node | Neo4j实体检索 | rewritten_query | graph_results |
| 5 | merge_node | 结果融合去重 | 三路结果 | merged_results |
| 6 | rerank_node | 重排序 | merged_results | reranked_results |
| 7 | build_prompt_node | 提示词组装 | reranked_results | final_prompt |
| 8 | refusal_check_node | 拒答判断 | query, context | should_refuse |
| 9 | generate_node | LLM生成 | final_prompt | final_answer |
| 10 | output_node | 保存历史、格式化输出 | final_answer | - |
| 11 | answer_faq_node | FAQ直接回答 | faq_result | final_answer |
| 12 | refuse_node | 拒答回复 | refusal_type | final_answer |

### 4.2 条件路由

```python
def route_after_faq(state: RAGState) -> str:
    """FAQ匹配后路由"""
    if state.get("faq_matched", False):
        return "answer_faq"
    return "query_rewrite"


def route_after_refusal(state: RAGState) -> str:
    """拒答判断后路由"""
    if state.get("should_refuse", False):
        return "refuse"
    return "generate"
```

---

## 5. 存储层设计

### 5.1 存储架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Storage Layer                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │   FAQ   │  │ Vector  │  │Fulltext │  │  Graph  │            │
│  │  Store  │  │  Store  │  │  Store  │  │  Store  │            │
│  │ (SQLite)│  │(Chroma) │  │(Whoosh) │  │(Neo4j)  │            │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘            │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Session Store (SQLite)                      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Index Manager                               │    │
│  │  - 文档摄入  - 增量更新  - 状态追踪                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 各存储组件说明

| 组件 | 技术选型 | 用途 | 关键方法 |
|------|----------|------|----------|
| FAQStore | SQLite | FAQ问答对存储 | exact_search, semantic_search, add, update |
| VectorStore | Chroma | 向量索引与检索 | add, search, delete, update |
| FulltextStore | Whoosh | BM25全文检索 | add, search, delete, update |
| GraphStore | Neo4j | 知识图谱存储 | create_entity, create_relation, query_entities |
| SessionStore | SQLite | 会话历史持久化 | create_conversation, save_message, get_history |
| IndexManager | - | 数据摄入管理 | ingest_document, ingest_batch, delete_document |

### 5.3 检索权重配置

```yaml
retrieval:
  vector_weight: 0.5
  fulltext_weight: 0.3
  graph_weight: 0.2
```

动态权重调整：当某个检索源不可用时，自动重新分配权重。

---

## 6. 外部服务层

### 6.1 阿里云服务

| 服务 | 模型 | 用途 | API端点 |
|------|------|------|---------|
| LLM | qwen-plus | 生成回答、Query改写、实体提取 | dashscope.aliyuncs.com |
| Embedding | text-embedding-v3 | 文本向量化 | dashscope.aliyuncs.com |
| Rerank | gte-rerank | 检索结果重排序 | dashscope.aliyuncs.com |

### 6.2 内部服务

| 服务 | 功能 |
|------|------|
| SensitiveFilter | 敏感词检测与过滤 |
| PromptTemplateManager | 提示词模板管理 |
| ConfigManager | 配置热更新管理 |
| DegradationManager | 服务降级管理 |

---

## 7. Graph构建

### 7.1 Graph定义

```python
def build_graph() -> StateGraph:
    workflow = StateGraph(RAGState)

    # 添加节点
    workflow.add_node("input", input_node)
    workflow.add_node("faq_match", faq_match_node)
    workflow.add_node("query_rewrite", query_rewrite_node)
    workflow.add_node("vector_retrieve", vector_retrieve_node)
    workflow.add_node("fulltext_retrieve", fulltext_retrieve_node)
    workflow.add_node("graph_retrieve", graph_retrieve_node)
    workflow.add_node("merge", merge_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("build_prompt", build_prompt_node)
    workflow.add_node("refusal_check", refusal_check_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("output", output_node)
    workflow.add_node("answer_faq", answer_faq_node)
    workflow.add_node("refuse", refuse_node)

    # 设置入口
    workflow.set_entry_point("input")

    # 定义边
    workflow.add_edge("input", "faq_match")

    workflow.add_conditional_edges(
        "faq_match", route_after_faq,
        {"answer_faq": "answer_faq", "query_rewrite": "query_rewrite"}
    )

    # 并行检索
    workflow.add_edge("query_rewrite", "vector_retrieve")
    workflow.add_edge("query_rewrite", "fulltext_retrieve")
    workflow.add_edge("query_rewrite", "graph_retrieve")

    workflow.add_edge("vector_retrieve", "merge")
    workflow.add_edge("fulltext_retrieve", "merge")
    workflow.add_edge("graph_retrieve", "merge")

    workflow.add_edge("merge", "rerank")
    workflow.add_edge("rerank", "build_prompt")
    workflow.add_edge("build_prompt", "refusal_check")

    workflow.add_conditional_edges(
        "refusal_check", route_after_refusal,
        {"generate": "generate", "refuse": "refuse"}
    )

    workflow.add_edge("generate", "output")
    workflow.add_edge("answer_faq", "output")
    workflow.add_edge("refuse", "output")

    return workflow
```

### 7.2 并行执行

LangGraph自动并行执行无依赖关系的节点。三个检索节点（vector_retrieve, fulltext_retrieve, graph_retrieve）会并行执行，merge_node等待所有输入完成后执行。

---

## 8. 配置管理

### 8.1 配置来源

支持三种配置加载方式：

1. **环境变量**：敏感信息（API Key、密码）
2. **YAML文件**：业务配置，支持热更新
3. **代码默认值**：兜底配置

### 8.2 配置热更新

使用Watchdog监听配置文件变更，变更后自动重载配置并通知相关服务。

```python
class ConfigManager:
    """配置管理器"""

    def start_watching(self) -> None:
        """启动配置文件监听"""

    def stop_watching(self) -> None:
        """停止监听"""

    def on_reload(self, callback: Callable) -> None:
        """注册重载回调"""

    def reload_now(self) -> None:
        """立即重载"""
```

### 8.3 配置结构

```yaml
llm:
  model: "qwen-plus"
  temperature: 0.7
  max_tokens: 2048
  enable_thinking: false

embedding:
  model: "text-embedding-v3"
  dimension: 1024

rerank:
  model: "gte-rerank"
  top_k: 5

storage:
  data_dir: "./data"
  neo4j_uri: "bolt://localhost:7687"

retrieval:
  vector_top_k: 20
  fulltext_top_k: 20
  vector_weight: 0.5
  fulltext_weight: 0.3
  graph_weight: 0.2

session:
  max_history_turns: 5
```

---

## 9. 降级策略

### 9.1 熔断器

每个外部服务配置熔断器，防止故障扩散：

```python
@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int = 3      # 连续失败阈值
    recovery_timeout: float = 60.0  # 恢复超时
```

### 9.2 降级策略汇总

| 服务 | 降级策略 | 说明 |
|------|----------|------|
| **Embedding** | 1. 本地缓存查找<br>2. 预计算向量<br>3. 零向量 | 零向量导致向量检索失效 |
| **Vector Store** | 返回空结果 | 依赖其他检索源 |
| **Fulltext Store** | 返回空结果 | 依赖其他检索源 |
| **Graph Store** | 返回空结果 | 依赖其他检索源 |
| **Rerank** | 1. 本地BM25重排<br>2. 保持原始顺序 | 质量下降但可用 |
| **LLM** | 1. 备用模型(qwen-turbo)<br>2. 模板回复 | 降级模型或模板 |
| **Query Rewrite** | 简单规则改写 | 拼接历史上下文 |

### 9.3 降级装饰器

```python
@with_fallback("embedding")
async def embed(text: str) -> list[float]:
    # 正常逻辑
    ...
```

当服务失败时，自动调用注册的fallback handler。

### 9.4 检索源动态权重

当部分检索源不可用时，自动调整权重：

```python
# 原始权重
vector_weight: 0.5, fulltext_weight: 0.3, graph_weight: 0.2

# Vector不可用时，重新分配
fulltext_weight: 0.6, graph_weight: 0.4
```

---

## 10. 错误处理

### 10.1 异常分类

```
RAGError (基类)
├── ConfigurationError
├── StorageError
│   ├── VectorStoreError
│   ├── FulltextStoreError
│   ├── GraphStoreError
│   └── FAQStoreError
├── ServiceError
│   ├── LLMServiceError
│   ├── EmbeddingServiceError
│   └── RerankServiceError
├── IngestionError
└── ValidationError
```

### 10.2 节点错误处理

每个节点使用装饰器捕获异常，记录到state.errors，不中断流程：

```python
@handle_node_errors("vector_retrieve")
async def vector_retrieve_node(state: RAGState) -> dict:
    ...
```

---

## 11. LangGraph API部署

### 11.1 项目结构

```
rag_rag/
├── langgraph.json                # LangGraph配置
├── pyproject.toml
├── requirements.txt
├── .env.example
│
├── src/rag_rag/
│   ├── core/                     # 配置、异常、日志
│   ├── graph/                    # StateGraph定义
│   │   ├── state.py
│   │   ├── graph.py
│   │   ├── nodes/
│   │   └── routers.py
│   ├── storage/                  # 存储层
│   ├── services/                 # 外部服务
│   ├── ingestion/                # 数据摄入
│   └── api/                      # API扩展
│
├── config/                       # 配置文件
│   ├── prompts/
│   └── settings.yaml
│
├── data/                         # 数据目录
│
├── tests/
└── scripts/
```

### 11.2 LangGraph配置

```json
{
  "python_version": "3.11",
  "dependencies": ["."],
  "graphs": {
    "rag_agent": "./src/rag_rag/graph/graph.py:graph"
  },
  "env": ".env"
}
```

### 11.3 API端点

| 端点 | 方法 | 功能 |
|------|------|------|
| /threads | POST | 创建会话 |
| /threads/{id}/runs | POST | 执行查询 |
| /threads/{id}/runs/stream | POST | 流式执行 |
| /threads/{id}/state | GET | 获取状态 |
| /health/services | GET | 服务健康状态 |
| /health/ready | GET | 就绪检查 |

### 11.4 启动命令

```bash
# 开发模式
langgraph dev

# 生产模式
langgraph up --port 8123

# Docker
docker-compose up -d
```

---

## 12. 与评测系统对接

### 12.1 评测系统调用方式

```python
from langgraph.pregel.remote import RemoteGraph

rag_client = RemoteGraph("http://localhost:8123")

result = await rag_client.ainvoke({
    "query": "如何申请年假？",
    "enable_thinking": False
})
```

### 12.2 可评测字段

| 字段 | 说明 | 评测用途 |
|------|------|----------|
| query_rewrite | Query改写结果 | 改写质量评测 |
| faq_match | FAQ匹配结果 | FAQ匹配准确率 |
| vector_results | 向量检索结果 | 检索召回评测 |
| fulltext_results | 全文检索结果 | 检索召回评测 |
| graph_results | 图谱检索结果 | 图谱检索评测 |
| reranked_results | 重排序结果 | 重排序效果评测 |
| final_prompt | 组装的提示词 | 提示词分析 |
| thinking_process | 思考过程 | 推理能力评测 |
| final_answer | 最终答案 | 生成质量评测 |
| stage_timing | 各阶段耗时 | 性能评测 |
| should_refuse | 是否拒答 | 拒答策略评测 |

---

## 13. 开发与部署

### 13.1 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 初始化存储
python scripts/init_stores.py

# 导入数据
python scripts/ingest_docs.py --source ./documents
```

### 13.2 环境变量

```bash
# .env
DASHSCOPE_API_KEY=sk-xxx
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

### 13.3 Docker部署

```yaml
# docker-compose.yml
services:
  rag-rag:
    build: .
    ports:
      - "8123:8123"
    environment:
      - DASHSCOPE_API_KEY=${DASHSCOPE_API_KEY}
    depends_on:
      - neo4j

  neo4j:
    image: neo4j:community
    ports:
      - "7474:7474"
      - "7687:7687"
```

---

## 14. 后续扩展

### 14.1 可能的扩展方向

- 多租户支持
- A/B测试框架
- 检索效果反馈学习
- 动态Prompt优化
- 多语言支持

### 14.2 性能优化方向

- 检索结果缓存
- Embedding预热
- 批量处理优化
- 异步I/O优化