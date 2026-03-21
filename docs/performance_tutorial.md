# RAG 各环节性能评测教程

本教程介绍如何使用 RAG 评测系统的性能分析功能，追踪各处理阶段（查询改写、FAQ匹配、检索、重排序、生成）的耗时。

## 目录

1. [功能概述](#功能概述)
2. [快速开始](#快速开始)
3. [Timing 提取策略](#timing-提取策略)
4. [配置方式](#配置方式)
5. [预设配置模板](#预设配置模板)
6. [自定义配置示例](#自定义配置示例)
7. [性能指标说明](#性能指标说明)
8. [UI 查看结果](#ui-查看结果)
9. [API 使用示例](#api-使用示例)
10. [常见问题](#常见问题)

---

## 功能概述

### 问题背景

RAG 系统由多个处理阶段组成：
- **查询改写** (Query Rewrite)
- **FAQ 匹配** (FAQ Match)
- **检索** (Retrieval)
- **重排序** (Rerank)
- **生成** (Generation)

传统评测只记录整体延迟 `latency_ms`，无法定位性能瓶颈。

### 解决方案

本系统提供：
1. **StageTiming 模型**：记录各阶段耗时
2. **灵活的提取系统**：支持多种数据来源
3. **性能指标**：自动计算性能相关评测指标
4. **UI 展示**：可视化各阶段耗时

### 数据来源多样性

不同 RAG 服务返回 timing 数据的方式不同：

| 来源 | 说明 | 适用策略 |
|------|------|----------|
| LangGraph state | timing 在 state 对象中 | `STATE` |
| 响应 metadata | timing 在 metadata 字段 | `METADATA` |
| 自定义字段 | timing 在特定字段路径 | `FIELD_PATH` |
| 需要计算 | 根据返回内容估算 | `CALCULATED` |
| 自定义逻辑 | 需要特殊处理 | `CUSTOM` |
| 无 timing 数据 | 按比例估算 | `FALLBACK` |

---

## 快速开始

### 1. 使用默认配置

最简单的方式是使用预设配置：

```python
from src.rag.langgraph_adapter import LangGraphAdapter
from src.rag.timing_config import get_langgraph_config

# 使用 LangGraph 预设配置
adapter = LangGraphAdapter(
    config=rag_config,
    timing_config=get_langgraph_config(),
)

# 查询时自动提取 timing
response = await adapter.query("什么是 RAG?")

# 查看各阶段耗时
if response.stage_timing:
    print(f"总耗时: {response.stage_timing.total_ms:.2f}ms")
    for stage, ms in response.stage_timing.get_stage_timings().items():
        print(f"  {stage}: {ms:.2f}ms")
```

### 2. 使用 Mock 适配器测试

```python
from src.rag.mock_adapter import MockRAGAdapter

# Mock 适配器内置模拟 timing
adapter = MockRAGAdapter(simulate_latency=True)
response = await adapter.query("测试查询")

# 自动生成模拟的 stage_timing
print(response.stage_timing)
```

---

## Timing 提取策略

系统支持 7 种提取策略：

### STATE - 从 LangGraph State 提取

适用于 timing 数据在 LangGraph state 中的服务。

```python
StageTimingExtractor(
    stage="query_rewrite",
    strategy=TimingExtractionStrategy.STATE,
    field_path="query_rewrite.timing_ms",  # 主路径
    fallback_paths=["timing.query_rewrite_ms"],  # 备用路径
    unit="ms",
)
```

**假设响应结构**：
```json
{
  "query_rewrite": {
    "rewritten_query": "...",
    "timing_ms": 45.2
  }
}
```

### METADATA - 从 Metadata 提取

适用于 timing 数据在 metadata 字段中的服务。

```python
StageTimingExtractor(
    stage="retrieval",
    strategy=TimingExtractionStrategy.METADATA,
    field_path="timing.retrieval_ms",
    unit="ms",
)
```

**假设响应结构**：
```json
{
  "retrieval": [...],
  "metadata": {
    "timing": {
      "retrieval_ms": 120.5
    }
  }
}
```

### FIELD_PATH - 按字段路径提取

适用于 timing 在任意字段路径的情况。

```python
StageTimingExtractor(
    stage="generation",
    strategy=TimingExtractionStrategy.FIELD_PATH,
    field_path="llm_output.generation_time_ms",
    fallback_paths=["timing.generation_ms"],
    unit="ms",
)
```

### CALCULATED - 基于内容计算

适用于没有 timing 数据，需要根据返回内容估算的场景。

```python
StageTimingExtractor(
    stage="retrieval",
    strategy=TimingExtractionStrategy.CALCULATED,
    calculate_expression="len(retrieval) * 15 + 50",  # 每文档15ms + 基础50ms
)
```

**支持的表达式变量**：

| 变量 | 说明 | 类型 |
|------|------|------|
| `retrieval` | 检索结果列表 | list |
| `rerank` | 重排结果列表 | list |
| `query_rewrite` | 查询改写结果 | dict |
| `faq_match` | FAQ 匹配结果 | dict |
| `llm_output` | LLM 输出 | dict |
| `token_count` | Token 总数 | int |
| `overall_ms` | 总延迟 | float |

**表达式示例**：
```python
# 按检索文档数估算
"len(retrieval) * 15 + 50"

# 按重排文档数估算
"len(rerank) * 10 + 30"

# 按 token 数估算生成耗时
"token_count * 2 + 200"

# 条件表达式
"50 if query_rewrite else 0"

# FAQ 匹配时估算
"30 if faq_match and faq_match.get('matched') else 0"
```

### CUSTOM - 自定义函数

适用于需要复杂逻辑的场景。

```python
def custom_retrieval_timing(data: dict) -> float:
    """自定义检索耗时计算"""
    docs = data.get("retrieval", [])
    base_time = 50
    per_doc_time = 15
    return base_time + len(docs) * per_doc_time

StageTimingExtractor(
    stage="retrieval",
    strategy=TimingExtractionStrategy.CUSTOM,
    custom_extractor=custom_retrieval_timing,
)
```

### STREAMING - 流式测量

适用于流式输出场景，timing 在流式过程中测量。

```python
StageTimingExtractor(
    stage="generation",
    strategy=TimingExtractionStrategy.STREAMING,
    default_ms=0,  # 流式未完成时的默认值
)
```

### FALLBACK - 按比例估算

当无法提取 timing 数据时，按预设比例估算各阶段占比。

```python
TimingExtractionConfig(
    fallback_proportions={
        "query_rewrite": 0.05,  # 5%
        "faq_match": 0.05,      # 5%
        "retrieval": 0.20,      # 20%
        "rerank": 0.10,         # 10%
        "generation": 0.60,     # 60%
    },
)
```

---

## 配置方式

### 方式一：代码配置（推荐）

```python
from src.rag.timing_config import (
    TimingExtractionConfig,
    StageTimingExtractor,
    TimingExtractionStrategy,
)

# 创建完整配置
timing_config = TimingExtractionConfig(
    stages={
        "query_rewrite": StageTimingExtractor(
            stage="query_rewrite",
            strategy=TimingExtractionStrategy.STATE,
            field_path="query_rewrite.timing_ms",
        ),
        "faq_match": StageTimingExtractor(
            stage="faq_match",
            strategy=TimingExtractionStrategy.METADATA,
            field_path="timing.faq_match_ms",
        ),
        "retrieval": StageTimingExtractor(
            stage="retrieval",
            strategy=TimingExtractionStrategy.CALCULATED,
            calculate_expression="len(retrieval) * 15 + 50",
        ),
        "rerank": StageTimingExtractor(
            stage="rerank",
            strategy=TimingExtractionStrategy.CUSTOM,
            custom_extractor=lambda d: len(d.get("rerank", [])) * 10,
        ),
        "generation": StageTimingExtractor(
            stage="generation",
            strategy=TimingExtractionStrategy.FIELD_PATH,
            field_path="llm_output.timing_ms",
        ),
    },
    fallback_proportions={
        "query_rewrite": 0.05,
        "faq_match": 0.05,
        "retrieval": 0.20,
        "rerank": 0.10,
        "generation": 0.60,
    },
)

# 使用配置
adapter = LangGraphAdapter(config=rag_config, timing_config=timing_config)
```

### 方式二：字典配置

```python
timing_config = TimingExtractionConfig.from_dict({
    "stages": {
        "query_rewrite": {
            "strategy": "state",
            "field_path": "query_rewrite.timing_ms",
        },
        "retrieval": {
            "strategy": "calculated",
            "calculate_expression": "len(retrieval) * 15 + 50",
        },
    },
    "fallback_proportions": {
        "query_rewrite": 0.05,
        "faq_match": 0.05,
        "retrieval": 0.20,
        "rerank": 0.10,
        "generation": 0.60,
    },
})
```

---

## 预设配置模板

系统提供以下预设配置：

### 1. get_langgraph_config()

适用于标准 LangGraph RAG 服务，timing 在 state 中。

```python
from src.rag.timing_config import get_langgraph_config

config = get_langgraph_config()
```

**提取路径**：
- `query_rewrite.timing_ms`
- `faq_match.timing_ms`
- `retrieval.timing_ms`
- `rerank.timing_ms`
- `llm_output.timing_ms`

### 2. get_metadata_config()

适用于 timing 在 metadata 中的服务。

```python
from src.rag.timing_config import get_metadata_config

config = get_metadata_config()
```

**提取路径**：
- `metadata.timing.query_rewrite_ms`
- `metadata.timing.faq_match_ms`
- `metadata.timing.retrieval_ms`
- `metadata.timing.rerank_ms`
- `metadata.timing.generation_ms`

### 3. get_calculated_config()

适用于无 timing 数据，需要根据内容计算的场景。

```python
from src.rag.timing_config import get_calculated_config

config = get_calculated_config()
```

**计算规则**：
- 查询改写：固定 50ms（如果存在）
- FAQ 匹配：固定 30ms（如果匹配成功）
- 检索：`len(retrieval) * 15 + 50`
- 重排：`len(rerank) * 10 + 30`
- 生成：`token_count * 2 + 200`

### 4. get_mock_config()

适用于 Mock 适配器，生成随机模拟数据。

```python
from src.rag.timing_config import get_mock_config

config = get_mock_config()
```

### 5. get_default_config()

纯 fallback 配置，按比例估算。

```python
from src.rag.timing_config import get_default_config

config = get_default_config()
```

---

## 自定义配置示例

### 示例 1：混合策略

根据实际响应结构，不同阶段使用不同策略：

```python
timing_config = TimingExtractionConfig(
    stages={
        # 查询改写：从 state 获取
        "query_rewrite": StageTimingExtractor(
            stage="query_rewrite",
            strategy=TimingExtractionStrategy.STATE,
            field_path="query_rewrite.timing_ms",
        ),

        # FAQ：从 metadata 获取
        "faq_match": StageTimingExtractor(
            stage="faq_match",
            strategy=TimingExtractionStrategy.METADATA,
            field_path="timing.faq",
        ),

        # 检索：从特定字段获取，有备用路径
        "retrieval": StageTimingExtractor(
            stage="retrieval",
            strategy=TimingExtractionStrategy.FIELD_PATH,
            field_path="retrieval_stats.duration_ms",
            fallback_paths=["timing.retrieval", "metadata.retrieval_time"],
        ),

        # 重排：根据文档数计算
        "rerank": StageTimingExtractor(
            stage="rerank",
            strategy=TimingExtractionStrategy.CALCULATED,
            calculate_expression="len(rerank) * 12 + 40",
        ),

        # 生成：自定义逻辑
        "generation": StageTimingExtractor(
            stage="generation",
            strategy=TimingExtractionStrategy.CUSTOM,
            custom_extractor=lambda d: (
                d.get("llm_output", {}).get("token_usage", {}).get("total_tokens", 0) * 2.5 + 150
            ),
        ),
    },
)
```

### 示例 2：处理秒级时间戳

如果 RAG 服务返回的是秒而非毫秒：

```python
timing_config = TimingExtractionConfig(
    stages={
        "retrieval": StageTimingExtractor(
            stage="retrieval",
            strategy=TimingExtractionStrategy.FIELD_PATH,
            field_path="retrieval.timing_seconds",
            unit="s",  # 自动转换为毫秒
        ),
    },
)
```

### 示例 3：复杂自定义提取

```python
def extract_generation_timing(data: dict) -> float:
    """复杂的生成阶段耗时计算"""
    llm_output = data.get("llm_output", {})
    token_usage = llm_output.get("token_usage", {})

    prompt_tokens = token_usage.get("prompt_tokens", 0)
    completion_tokens = token_usage.get("completion_tokens", 0)
    model = llm_output.get("model", "")

    # 不同模型有不同的速度
    if "gpt-4" in model.lower():
        ms_per_token = 3.0
    elif "gpt-3.5" in model.lower():
        ms_per_token = 1.5
    else:
        ms_per_token = 2.0

    return prompt_tokens * 0.5 + completion_tokens * ms_per_token + 100

timing_config = TimingExtractionConfig(
    stages={
        "generation": StageTimingExtractor(
            stage="generation",
            strategy=TimingExtractionStrategy.CUSTOM,
            custom_extractor=extract_generation_timing,
        ),
    },
)
```

---

## 性能指标说明

系统提供 6 个性能相关指标：

### 1. StageLatencyMetric

各阶段延迟分析，直接从 `stage_timing` 读取。

**返回内容**：
```json
{
  "total_ms": 1250.5,
  "stages": {
    "query_rewrite": {"latency_ms": 45.2, "percentage": 3.6, "source": "state"},
    "retrieval": {"latency_ms": 180.3, "percentage": 14.4, "source": "calculated"},
    ...
  }
}
```

### 2. TotalLatencyMetric

整体延迟评级。

**评级标准**：
- 优秀 (≥1.0): ≤500ms
- 良好 (0.8): ≤1000ms
- 可接受 (0.6): ≤2000ms
- 较慢 (<0.4): >2000ms

### 3. LatencyDistributionMetric

延迟分布统计，用于批量评测计算 P50/P95/P99。

### 4. PerformanceComparisonMetric

双 RAG 接口性能对比。

### 5. StageEfficiencyMetric

阶段效率分析，识别性能瓶颈。

**理想占比**：
- 查询改写：5%
- FAQ 匹配：3%
- 检索：15%
- 重排：10%
- 生成：50%

### 6. ThroughputMetric

吞吐量估算（请求/秒）。

---

## UI 查看结果

### 在结果详情页查看

1. 运行评测后，进入"📈 结果查看"标签
2. 选择一个评测运行
3. 在结果列表中选择一条记录
4. 切换到"⚡ 性能分析"标签页

**显示内容**：
- 总耗时和数据来源
- 各阶段耗时表格（阶段名、耗时、占比、数据来源）

### 性能分析示例

```
总耗时: 1250.50 ms
数据来源: extracted

| 阶段     | 耗时 (ms) | 占比 (%) | 数据来源    |
|----------|-----------|----------|-------------|
| 查询改写 | 45.20     | 3.62     | state       |
| FAQ 匹配 | 0.00      | 0.00     | fallback    |
| 检索     | 180.30    | 14.42    | calculated  |
| 重排序   | 75.00     | 6.00     | custom      |
| 生成     | 950.00    | 76.00    | field_path  |
```

---

## API 使用示例

### 完整评测流程

```python
import asyncio
from src.rag.langgraph_adapter import LangGraphAdapter
from src.rag.timing_config import get_langgraph_config
from src.evaluation import create_runner
from src.annotation import get_annotation_handler

async def run_evaluation():
    # 1. 准备 RAG 适配器（带 timing 配置）
    adapter = LangGraphAdapter(
        config={"service_url": "http://localhost:8000"},
        timing_config=get_langgraph_config(),
    )

    # 2. 获取标注数据
    handler = await get_annotation_handler()
    annotations = await handler.list(page=1, page_size=100)

    # 3. 创建评测运行器
    runner = await create_runner(max_concurrent=10)
    runner.set_rag_adapter(adapter)

    # 性能指标会自动计算
    runner.set_metrics([
        "retrieval_precision",
        "answer_relevance",
        "stage_latency",      # 性能指标
        "total_latency",      # 性能指标
        "stage_efficiency",   # 性能指标
    ])

    # 4. 运行评测
    run = await runner.run(annotations)

    # 5. 查看结果
    for result in run.results:
        if result.rag_response.stage_timing:
            timing = result.rag_response.stage_timing
            print(f"Query: {result.annotation.query[:30]}...")
            print(f"  Total: {timing.total_ms:.2f}ms")
            print(f"  Retrieval: {timing.retrieval_ms:.2f}ms")
            print(f"  Generation: {timing.generation_ms:.2f}ms")

asyncio.run(run_evaluation())
```

### 单独提取 Timing

```python
from src.rag.timing_extractor import TimingExtractor
from src.rag.timing_config import TimingExtractionConfig

# 创建提取器
config = TimingExtractionConfig(...)
extractor = TimingExtractor(config)

# 从原始响应提取
raw_response = {
    "query_rewrite": {"timing_ms": 45.2},
    "retrieval": [...],  # 10 documents
    "rerank": [...],
    "llm_output": {"timing_ms": 950.0}
}

timing = extractor.extract(raw_response, overall_ms=1250.0)

print(timing.query_rewrite_ms)  # 45.2
print(timing.retrieval_ms)      # 根据配置计算
print(timing.generation_ms)     # 950.0
```

---

## 常见问题

### Q1: 我的 RAG 服务没有返回 timing 数据怎么办？

使用 `CALCULATED` 或 `FALLBACK` 策略：

```python
# 方案 1：根据内容计算
timing_config = get_calculated_config()

# 方案 2：按比例估算
timing_config = get_default_config()
```

### Q2: 如何调试 timing 提取？

查看 `stage_timing.extraction_details`：

```python
response = await adapter.query("测试")
if response.stage_timing:
    for stage, source in response.stage_timing.extraction_details.items():
        print(f"{stage}: {source}")
```

### Q3: timing 值为 0 是什么原因？

可能原因：
1. 该阶段未执行（如没有查询改写）
2. 提取路径配置错误
3. 数据确实为 0

检查 `extraction_details` 确认提取策略。

### Q4: 如何处理不同环境的 RAG 服务？

根据环境选择不同配置：

```python
import os

env = os.getenv("ENV", "dev")

if env == "production":
    timing_config = get_langgraph_config()
elif env == "staging":
    timing_config = get_metadata_config()
else:
    timing_config = get_calculated_config()

adapter = LangGraphAdapter(config=rag_config, timing_config=timing_config)
```

### Q5: 性能指标可以在评测时禁用吗？

可以，在设置指标时不包含性能指标：

```python
runner.set_metrics([
    "retrieval_precision",
    "answer_relevance",
    # 不包含 stage_latency, total_latency 等
])
```

---

## 总结

本系统提供了灵活、可配置的性能评测能力：

1. **自动集成**：RAG 调用时自动提取 timing
2. **多策略支持**：适应不同的 RAG 服务响应结构
3. **预设模板**：快速上手，无需复杂配置
4. **性能指标**：自动计算，与效果评测并行
5. **可视化**：UI 直接查看各阶段耗时

通过性能评测，可以：
- 定位 RAG 系统性能瓶颈
- 对比不同接口/配置的性能差异
- 监控性能变化趋势
- 优化资源分配