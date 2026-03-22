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
| Python | 3.13 | 与评测系统保持一致 |
| 向量库 | Chroma | 纯Python、无依赖、嵌入模式 |
| 全文检索 | Whoosh | 纯Python库、无需服务 |
| 知识图谱 | Neo4j Community | Docker部署、成熟稳定 |
| LLM | 阿里云通义千问 (Qwen) | qwen-plus/qwen-turbo/qwen-max |
| Embedding | 阿里云 text-embedding-v3 | 支持中英文，维度1024/768/512/256/128 |
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

### 3.1 完整State定义

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
    errors: list[dict]
```

### 3.2 关键字段结构定义

#### conversation_history 结构

```python
# 对话历史中的每条消息
{
    "role": "user" | "assistant",   # 角色
    "content": str,                  # 消息内容
    "timestamp": str                 # ISO格式时间戳
}
```

#### faq_result 结构

```python
# FAQ匹配结果
{
    "faq_id": str,                   # FAQ ID
    "question": str,                 # FAQ问题
    "answer": str,                   # FAQ答案
    "confidence": float,             # 匹配置信度
    "match_type": "exact" | "semantic"  # 匹配类型
}
```

#### 检索结果结构 (vector_results, fulltext_results, graph_results)

```python
# 单条检索结果
{
    "document_id": str,              # 文档ID
    "content": str,                  # 文档内容
    "score": float,                  # 检索分数
    "source": str,                   # 来源标识
    "metadata": {                    # 元数据
        "title": str,
        "category": str,
        "keywords": list[str]
    }
}
```

#### stage_timing 结构

```python
# 各阶段耗时（毫秒）
{
    "input_ms": float,
    "faq_match_ms": float,
    "query_rewrite_ms": float,
    "vector_retrieve_ms": float,
    "fulltext_retrieve_ms": float,
    "graph_retrieve_ms": float,
    "merge_ms": float,
    "rerank_ms": float,
    "build_prompt_ms": float,
    "refusal_check_ms": float,
    "generation_ms": float,
    "total_ms": float
}
```

#### errors 结构

```python
# 错误信息
{
    "stage": str,                    # 发生错误的阶段
    "type": str,                     # 错误类型
    "message": str,                  # 错误消息
    "timestamp": str                 # ISO格式时间戳
}
```

#### token_usage 结构

```python
# Token使用统计
{
    "prompt_tokens": int,
    "completion_tokens": int,
    "total_tokens": int
}
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

### 5.4 知识图谱Schema

#### 实体类型

| 实体类型 | 说明 | 属性 |
|----------|------|------|
| Person | 人物 | name, department, position |
| Department | 部门 | name, description |
| Product | 产品 | name, version, category |
| Document | 文档 | title, type, url |
| Concept | 概念/术语 | name, definition |
| Event | 事件 | name, date, description |

#### 关系类型

| 关系类型 | 起点 | 终点 | 说明 |
|----------|------|------|------|
| BELONGS_TO | Person | Department | 隶属关系 |
| RESPONSIBLE_FOR | Person | Document | 负责文档 |
| RELATED_TO | Document | Concept | 涉及概念 |
| PART_OF | Document | Document | 文档层级 |
| DEPENDS_ON | Product | Product | 产品依赖 |
| MENTIONED_IN | Entity | Document | 出现于文档 |

#### Neo4j初始化

```cypher
// 创建约束
CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT dept_id IF NOT EXISTS FOR (d:Department) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE;

// 创建索引
CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name);
CREATE INDEX concept_name IF NOT EXISTS FOR (c:Concept) ON (c.name);
```

### 5.5 文档分块策略

```python
@dataclass
class ChunkingConfig:
    """分块配置"""
    chunk_size: int = 500           # 分块大小（字符）
    chunk_overlap: int = 50         # 重叠大小
    min_chunk_size: int = 100       # 最小分块大小
    respect_sentence_boundary: bool = True  # 尊重句子边界

class DocumentChunker:
    """文档分块器"""

    def chunk(self, document: Document) -> list[Chunk]:
        """
        分块策略：
        1. 按段落分割
        2. 超过chunk_size的段落进一步分割
        3. 保持句子完整性
        4. 添加重叠以保持上下文
        """
        chunks = []

        # 1. 按段落分割
        paragraphs = self._split_paragraphs(document.content)

        # 2. 处理每个段落
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) <= self.config.chunk_size:
                current_chunk += para + "\n"
            else:
                # 保存当前块
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, document))

                # 处理超长段落
                if len(para) > self.config.chunk_size:
                    sub_chunks = self._split_long_paragraph(para, document)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = para + "\n"

        # 保存最后一块
        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, document))

        return chunks
```

### 5.6 实体提取Pipeline

```python
class EntityExtractor:
    """实体提取器"""

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    async def extract(self, text: str) -> list[Entity]:
        """
        从文本中提取实体

        使用LLM进行实体识别，返回结构化实体列表
        """
        prompt = f"""请从以下文本中提取实体。

文本:
{text}

请识别以下类型的实体：
- Person: 人物姓名
- Department: 部门名称
- Product: 产品名称
- Concept: 专业术语或概念

输出JSON格式：
[
  {{"name": "实体名称", "type": "实体类型", "confidence": 0.0-1.0}}
]
"""
        response = await self.llm_service.generate(
            system_prompt="你是一个专业的实体识别助手。",
            user_prompt=prompt,
            temperature=0.1
        )

        import json
        try:
            entities = json.loads(response.content)
            return [Entity(**e) for e in entities]
        except:
            return []

    async def extract_relations(
        self,
        text: str,
        entities: list[Entity]
    ) -> list[Relation]:
        """提取实体间关系"""
        # ... 关系提取逻辑
```

---

## 6. 外部服务层

### 6.1 阿里云服务

| 服务 | 模型 | 用途 | API端点 |
|------|------|------|---------|
| LLM | qwen-plus | 生成回答、Query改写、实体提取 | dashscope.aliyuncs.com |
| Embedding | text-embedding-v3 | 文本向量化 | dashscope.aliyuncs.com |
| Rerank | gte-rerank | 检索结果重排序 | dashscope.aliyuncs.com |

### 6.2 服务超时与重试配置

```yaml
service_timeouts:
  llm:
    timeout: 60              # 秒
    max_retries: 3
    retry_delay: 1.0         # 秒，指数退避
  embedding:
    timeout: 30
    max_retries: 3
    retry_delay: 0.5
  rerank:
    timeout: 30
    max_retries: 2
    retry_delay: 0.5

retrieval_timeouts:
  vector_store:
    timeout: 10
  fulltext_store:
    timeout: 10
  graph_store:
    timeout: 15

rate_limits:
  dashscope_qpm: 60          # 每分钟请求数限制
  embedding_batch_size: 20   # Embedding批量大小
```

### 6.3 内部服务

| 服务 | 功能 |
|------|------|
| SensitiveFilter | 敏感词检测与过滤 |
| PromptTemplateManager | 提示词模板管理 |
| ConfigManager | 配置热更新管理 |
| DegradationManager | 服务降级管理 |

### 6.4 提示词模板

#### 默认模板 (default.yaml)

```yaml
name: default
system: |
  你是一个专业的企业知识库助手。
  你的职责是基于提供的知识库内容，准确、专业地回答用户问题。

  ## 回答原则
  1. 只使用提供的上下文信息回答问题
  2. 如果上下文不足以回答，请诚实说明
  3. 回答要简洁、准确、有条理
  4. 使用专业的语言风格

  ## 知识领域
  {domain}

user: |
  ## 相关上下文
  {context}

  ## 对话历史
  {conversation_history}

  ## 用户问题
  {query}

  请基于上下文回答用户问题。
```

#### 思考模式模板 (thinking.yaml)

```yaml
name: thinking
system: |
  你是一个专业的企业知识库助手，擅长深度思考和分析。

  ## 思考模式
  在回答之前，请先进行深入思考：
  1. 分析问题的核心意图
  2. 评估上下文的相关性和可靠性
  3. 构建回答的逻辑框架

  ## 输出格式
  <thinking>
  [你的思考过程]
  </thinking>
  [你的最终回答]

  ## 知识领域
  {domain}

user: |
  ## 相关上下文
  {context}

  ## 用户问题
  {query}

  请先进行思考分析，然后给出回答。
```

#### 模板加载机制

```python
class PromptTemplateManager:
    def __init__(self, template_dir: str):
        self.templates = {}
        self._load_templates(template_dir)

    def render(
        self,
        template_name: str,
        context: str,
        query: str,
        conversation_history: str = "",
        domain: str = "企业知识库"
    ) -> tuple[str, str]:
        """渲染模板，返回(system_prompt, user_prompt)"""
        template = self.templates.get(template_name, self.templates["default"])
        system_prompt = template["system"].format(domain=domain)
        user_prompt = template["user"].format(
            context=context,
            query=query,
            conversation_history=conversation_history
        )
        return system_prompt, user_prompt
```

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

### 7.2 并行执行机制

LangGraph的并行执行采用fan-out/fan-in模式：

```
                 ┌─────────────────┐
                 │  query_rewrite  │
                 └────────┬────────┘
                          │
            ┌─────────────┼─────────────┐
            │             │             │
            ▼             ▼             ▼
     ┌──────────┐  ┌──────────┐  ┌──────────┐
     │  vector  │  │ fulltext │  │  graph   │
     │ retrieve │  │ retrieve │  │ retrieve │
     └────┬─────┘  └────┬─────┘  └────┬─────┘
          │             │             │
          └─────────────┼─────────────┘
                        │
                        ▼
                 ┌──────────┐
                 │  merge   │
                 └──────────┘
```

**执行机制说明**：

1. LangGraph检测到多个节点从同一节点接收输入时，自动并行执行
2. merge_node等待所有输入节点完成后才执行
3. 每个检索节点独立运行，失败不影响其他节点

**并行执行代码实现**：

```python
# 在节点内部使用asyncio.gather进行显式并行
async def parallel_retrieve_internal(state: RAGState) -> dict:
    """内部并行检索协调"""
    query = state.get("rewritten_query") or state["query"]

    # 使用asyncio.gather并行执行
    results = await asyncio.gather(
        _vector_retrieve(query),
        _fulltext_retrieve(query),
        _graph_retrieve(query),
        return_exceptions=True
    )

    # 处理结果和异常
    vector_results = results[0] if not isinstance(results[0], Exception) else []
    fulltext_results = results[1] if not isinstance(results[1], Exception) else []
    graph_results = results[2] if not isinstance(results[2], Exception) else []

    return {
        "vector_results": vector_results,
        "fulltext_results": fulltext_results,
        "graph_results": graph_results
    }
```

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

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self._config: Optional[RAGConfig] = None
        self._lock = asyncio.Lock()  # 配置锁，确保原子更新
        self._reload_callbacks: list[Callable] = []
        self._observer: Optional[Observer] = None

    def start_watching(self) -> None:
        """启动配置文件监听"""

    def stop_watching(self) -> None:
        """停止监听"""

    def on_reload(self, callback: Callable) -> None:
        """注册重载回调"""

    async def reload_now(self) -> None:
        """立即重载配置（原子操作）"""
        async with self._lock:
            # 1. 加载新配置
            new_config = RAGConfig.from_yaml(self.config_path)

            # 2. 验证配置有效性
            self._validate_config(new_config)

            # 3. 原子替换
            old_config = self._config
            self._config = new_config

            # 4. 通知服务更新（带回滚能力）
            try:
                for callback in self._reload_callbacks:
                    await callback(new_config)
            except Exception as e:
                # 回滚到旧配置
                self._config = old_config
                logger.error(f"Config reload failed, rolled back: {e}")
                raise
```

#### 配置更新通知机制

```python
# 服务注册配置更新回调
async def on_config_change(new_config: RAGConfig):
    """服务配置变更处理"""
    # 1. 检查是否需要重连
    if llm_service.config.model != new_config.llm.model:
        await llm_service.reconnect(new_config.llm)

    # 2. 重置熔断器状态
    if degradation_manager:
        degradation_manager.reset_circuit_breaker("llm")

# 注册回调
config_manager.on_reload(on_config_change)
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

#### Rerank降级说明

Rerank降级使用本地BM25重排时，与初始全文检索的BM25不同：

1. **初始BM25**：基于Whoosh对原始文档库进行检索，返回top_k文档
2. **降级BM25重排**：对已检索到的文档（来自向量、全文、图谱三路融合结果）进行重新打分排序

```python
async def _rerank_fallback(self, query: str, documents: list[str]) -> list:
    """
    Rerank降级：本地BM25重排

    与初始BM25检索的区别：
    - 初始检索：从百万级文档中检索top_k
    - 降级重排：对已检索到的几十个文档重新排序
    """
    from rank_bm25 import BM25Okapi

    tokenized_docs = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(query.split())

    # 按分数排序
    indexed_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    return [
        RerankResult(index=idx, score=score, document=documents[idx])
        for idx, score in indexed_scores[:self.top_k]
    ]
```

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

### 9.5 检索结果融合算法

使用**加权分数融合**（Weighted Score Fusion）算法合并多路检索结果：

```python
def merge_results(
    vector_results: list[dict],
    fulltext_results: list[dict],
    graph_results: list[dict],
    weights: dict[str, float]
) -> list[dict]:
    """
    加权分数融合算法

    算法说明：
    1. 各路结果分数归一化到0-1
    2. 按document_id合并相同文档
    3. 加权求和计算综合分数
    4. 按综合分数排序
    """
    merged = {}

    # 归一化并加权
    for result in vector_results:
        doc_id = result["document_id"]
        normalized_score = _normalize_score(result["score"], "vector")
        merged[doc_id] = {
            **result,
            "vector_score": normalized_score,
            "combined_score": normalized_score * weights["vector"]
        }

    for result in fulltext_results:
        doc_id = result["document_id"]
        normalized_score = _normalize_score(result["score"], "fulltext")
        if doc_id in merged:
            merged[doc_id]["fulltext_score"] = normalized_score
            merged[doc_id]["combined_score"] += normalized_score * weights["fulltext"]
        else:
            merged[doc_id] = {
                **result,
                "fulltext_score": normalized_score,
                "combined_score": normalized_score * weights["fulltext"]
            }

    # ... 类似处理graph_results

    # 按综合分数排序
    return sorted(merged.values(), key=lambda x: x["combined_score"], reverse=True)

def _normalize_score(score: float, source: str) -> float:
    """分数归一化"""
    # 向量检索：余弦相似度已在0-1范围
    # 全文检索：BM25分数需要归一化
    if source == "fulltext":
        return min(score / 10.0, 1.0)  # 假设BM25分数上限约10
    return score
```

**为什么不使用RRF（Reciprocal Rank Fusion）？**

RRF仅考虑排名位置，忽略具体分数。对于需要精细区分相关性的场景，加权分数融合更合适。

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
  "python_version": "3.13",
  "dependencies": ["."],
  "graphs": {
    "rag_agent": "./src/rag_rag/graph/graph.py:graph"
  },
  "env": ".env"
}
```

### 11.3 API认证

LangGraph API支持多种认证方式：

```yaml
# config/auth.yaml
auth:
  enabled: true
  type: "api_key"           # api_key | jwt | oauth2

  # API Key认证
  api_key:
    header: "X-API-Key"
    keys:
      - name: "evaluator"   # 评测系统专用
        key: "${EVAL_API_KEY}"
        rate_limit: 100     # 每分钟请求数
      - name: "user"
        key: "${USER_API_KEY}"
        rate_limit: 30

  # JWT认证（可选）
  jwt:
    secret: "${JWT_SECRET}"
    issuer: "rag-system"
    expiry_hours: 24
```

#### 认证中间件

```python
from fastapi import Request, HTTPException

async def auth_middleware(request: Request, call_next):
    """API认证中间件"""
    api_key = request.headers.get("X-API-Key")

    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API Key")

    if not validate_api_key(api_key):
        raise HTTPException(status_code=403, detail="Invalid API Key")

    # 检查速率限制
    if await rate_limiter.is_exceeded(api_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    response = await call_next(request)
    return response
```

### 11.4 API端点

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

### 12.3 输出格式规范

API返回的完整JSON结构：

```json
{
  "query": "如何申请年假？",
  "conversation_id": "uuid-string",

  "faq_matched": false,
  "faq_result": null,

  "original_query": "如何申请年假？",
  "rewritten_query": "企业员工年假申请流程和条件",
  "rewrite_type": "expansion",
  "rewrite_confidence": 0.85,

  "vector_results": [
    {
      "document_id": "doc-001",
      "content": "年假申请流程...",
      "score": 0.92,
      "source": "chroma",
      "metadata": {"title": "员工手册", "category": "休假制度"}
    }
  ],
  "fulltext_results": [...],
  "graph_results": [...],

  "reranked_results": [
    {
      "document_id": "doc-001",
      "content": "年假申请流程...",
      "rerank_score": 0.95,
      "rank": 1
    }
  ],
  "rerank_scores": [0.95, 0.87, 0.82],

  "should_refuse": false,
  "refusal_reason": "",
  "refusal_type": "",

  "system_prompt": "你是一个专业的企业知识库助手...",
  "context_prompt": "## 相关上下文\n1. 年假申请流程...",
  "final_prompt": "## 相关上下文\n...\n## 用户问题\n如何申请年假？",
  "prompt_template_name": "default",

  "thinking_process": "",
  "final_answer": "根据员工手册，年假申请流程如下...",
  "token_usage": {
    "prompt_tokens": 450,
    "completion_tokens": 120,
    "total_tokens": 570
  },

  "stage_timing": {
    "input_ms": 2.5,
    "faq_match_ms": 15.3,
    "query_rewrite_ms": 320.5,
    "vector_retrieve_ms": 45.2,
    "fulltext_retrieve_ms": 23.1,
    "graph_retrieve_ms": 38.7,
    "merge_ms": 1.2,
    "rerank_ms": 120.5,
    "build_prompt_ms": 0.8,
    "refusal_check_ms": 85.3,
    "generation_ms": 1250.6,
    "total_ms": 1903.7
  },

  "metadata": {
    "model": "qwen-plus",
    "degraded_sources": []
  },

  "errors": []
}
```

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