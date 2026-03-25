# 数据注入指南

## 概述

RAG Pipeline 需要预先注入数据才能正常工作。本文档说明如何准备和注入数据到各个存储后端。

## 数据存储架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Storage Backends                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│   │  FAQ Store  │  │Vector Store │  │Fulltext Store│        │
│   │  (SQLite)   │  │  (Chroma)   │  │  (Whoosh)   │        │
│   │             │  │             │  │             │        │
│   │  Q&A Pairs  │  │  Embeddings │  │  BM25 Index │        │
│   │  Keywords   │  │  Metadata   │  │  Content    │        │
│   └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                             │
│   ┌─────────────┐                                           │
│   │ Graph Store │  (可选)                                   │
│   │  (Neo4j)    │                                           │
│   │             │                                           │
│   │  Entities   │                                           │
│   │  Relations  │                                           │
│   └─────────────┘                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 快速开始

### 使用预置数据

项目包含电子产品知识库生成器：

```bash
cd rag_rag

# 生成并注入数据（推荐）
python scripts/ingest_electronics_data.py --generate

# 仅注入已有数据
python scripts/ingest_electronics_data.py
```

### 验证注入结果

```bash
# 检查数据量
python -c "
import asyncio
from pathlib import Path
import sys
sys.path.insert(0, 'src')

from rag_rag.storage.faq_store import FAQStore
from rag_rag.storage.vector_store import VectorStore
from rag_rag.storage.fulltext_store import FulltextStore

async def check():
    # FAQ Store
    faq = FAQStore(db_path=Path('data/faq.db'))
    await faq.initialize()
    keys = await faq.keys()
    print(f'FAQ Store: {len(keys)} items')
    await faq.close()

    # Vector Store
    vec = VectorStore(persist_dir=Path('data/chroma'))
    await vec.initialize()
    count = await vec.count()
    print(f'Vector Store: {count} documents')
    await vec.close()

    # Fulltext Store
    ft = FulltextStore(index_dir=Path('data/whoosh'))
    await ft.initialize()
    count = await ft.count()
    print(f'Fulltext Store: {count} documents')
    await ft.close()

asyncio.run(check())
"
```

## 数据格式规范

### FAQ 数据格式

文件路径: `scripts/data/generated/faqs.json`

```json
[
  {
    "id": "faq-0001",
    "question": "iPhone 15 Pro Max如何截屏？",
    "answer": "Apple iPhone 15 Pro Max截屏方法：同时按住电源键和音量加键，屏幕闪烁即表示截屏成功。截图会自动保存到相册的截屏文件夹中。",
    "category": "手机操作",
    "keywords": ["iPhone", "截屏", "iPhone 15 Pro Max"],
    "product": "iPhone 15 Pro Max",
    "brand": "Apple"
  }
]
```

**字段说明:**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `id` | string | 是 | 唯一标识符，格式 `faq-XXXX` |
| `question` | string | 是 | 问题文本 |
| `answer` | string | 是 | 答案文本 |
| `category` | string | 否 | 分类标签 |
| `keywords` | array | 否 | 关键词列表 |
| `product` | string | 否 | 关联产品 |
| `brand` | string | 否 | 关联品牌 |

**最佳实践:**
- 问题简洁明确，避免歧义
- 答案完整但不过长（建议 100-300 字）
- 关键词覆盖用户可能使用的搜索词

### 文档数据格式

文件路径: `scripts/data/generated/documents.json`

```json
[
  {
    "id": "doc-0001",
    "title": "iPhone 15 Pro Max快速入门指南",
    "content": "一、开箱检查\n\n打开包装盒，请确认以下物品齐全...",
    "category": "使用教程",
    "product": "iPhone 15 Pro Max",
    "brand": "Apple",
    "keywords": ["iPhone", "使用教程", "入门"]
  }
]
```

**字段说明:**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `id` | string | 是 | 唯一标识符，格式 `doc-XXXX` |
| `title` | string | 是 | 文档标题 |
| `content` | string | 是 | 文档正文 |
| `category` | string | 否 | 分类标签 |
| `product` | string | 否 | 关联产品 |
| `brand` | string | 否 | 关联品牌 |
| `keywords` | array | 否 | 关键词列表 |

**最佳实践:**
- 内容结构化，使用标题分段
- 单篇文档建议 500-2000 字
- 包含足够的上下文信息

### 知识图谱数据格式 (可选)

```json
{
  "entities": [
    {
      "id": "ent-001",
      "name": "iPhone 15 Pro Max",
      "type": "Product",
      "properties": {
        "brand": "Apple",
        "category": "智能手机"
      }
    }
  ],
  "relations": [
    {
      "from": "ent-001",
      "to": "ent-002",
      "type": "PRODUCED_BY"
    }
  ]
}
```

## 自定义数据注入

### 1. 准备数据文件

```bash
mkdir -p scripts/data/generated
```

创建 `faqs.json` 和 `documents.json`。

### 2. 编写注入脚本

```python
#!/usr/bin/env python
"""自定义数据注入脚本"""

import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, 'src')

from rag_rag.storage.faq_store import FAQStore
from rag_rag.storage.vector_store import VectorStore
from rag_rag.storage.fulltext_store import FulltextStore
from rag_rag.core.logging import get_logger

logger = get_logger("custom_ingestion")


async def ingest_custom_data():
    """注入自定义数据"""
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    with open("scripts/data/generated/faqs.json", encoding="utf-8") as f:
        faqs = json.load(f)

    with open("scripts/data/generated/documents.json", encoding="utf-8") as f:
        documents = json.load(f)

    # 1. 注入 FAQ Store
    logger.info(f"注入 {len(faqs)} 条 FAQ...")
    faq_store = FAQStore(db_path=data_dir / "faq.db")
    await faq_store.initialize()

    for faq in faqs:
        await faq_store.set(faq["id"], {
            "question": faq["question"],
            "answer": faq["answer"],
            "category": faq.get("category", ""),
            "keywords": faq.get("keywords", []),
        })

    await faq_store.close()

    # 2. 注入 Vector Store (需要 Embedding Service)
    logger.info(f"注入 {len(documents)} 篇文档到向量库...")
    vector_store = VectorStore(
        persist_dir=data_dir / "chroma",
        collection_name="custom_docs"
    )
    await vector_store.initialize()

    # 注意：需要真实的 Embedding
    # 这里仅演示结构
    for doc in documents:
        # embedding = await embedding_service.embed(doc["content"])
        # await vector_store.add([{"id": doc["id"], ...}], [embedding])
        pass

    await vector_store.close()

    # 3. 注入 Fulltext Store
    logger.info(f"注入 {len(documents)} 篇文档到全文索引...")
    fulltext_store = FulltextStore(
        index_dir=data_dir / "whoosh",
        index_name="custom_docs"
    )
    await fulltext_store.initialize()

    docs_to_add = [
        {
            "id": doc["id"],
            "content": doc["content"],
            "metadata": {
                "title": doc.get("title", ""),
                "category": doc.get("category", ""),
            }
        }
        for doc in documents
    ]
    await fulltext_store.add(docs_to_add)
    await fulltext_store.close()

    logger.info("注入完成！")


if __name__ == "__main__":
    asyncio.run(ingest_custom_data())
```

### 3. 执行注入

```bash
python scripts/ingest_custom_data.py
```

## 数据更新策略

### 增量更新

```python
# 仅添加新数据
async def incremental_update(new_faqs: list, new_docs: list):
    faq_store = FAQStore(db_path=Path("data/faq.db"))
    await faq_store.initialize()

    for faq in new_faqs:
        existing = await faq_store.get(faq["id"])
        if not existing:
            await faq_store.set(faq["id"], faq)

    await faq_store.close()
```

### 全量更新

```python
# 清空后重新注入
async def full_update(faqs: list, docs: list):
    # 清空现有数据
    faq_store = FAQStore(db_path=Path("data/faq.db"))
    await faq_store.initialize()
    await faq_store.clear()

    # 重新注入
    for faq in faqs:
        await faq_store.set(faq["id"], faq)

    await faq_store.close()
```

## 数据质量检查

### FAQ 数据检查

```python
async def check_faq_quality():
    faq_store = FAQStore(db_path=Path("data/faq.db"))
    await faq_store.initialize()

    keys = await faq_store.keys()
    issues = []

    for key in keys:
        faq = await faq_store.get(key)

        # 检查问题长度
        if len(faq.get("question", "")) < 5:
            issues.append(f"{key}: 问题太短")

        # 检查答案长度
        if len(faq.get("answer", "")) < 10:
            issues.append(f"{key}: 答案太短")

        # 检查重复
        # ...

    await faq_store.close()
    return issues
```

### 搜索效果测试

```python
async def test_search_quality():
    """测试搜索质量"""
    fulltext_store = FulltextStore(index_dir=Path("data/whoosh"))
    await fulltext_store.initialize()

    test_cases = [
        ("iPhone截屏", 3),  # 期望至少返回3条
        ("华为手机续航", 2),
        ("MacBook使用", 2),
    ]

    for query, expected_min in test_cases:
        results = await fulltext_store.search(query, top_k=10)
        actual = len(results)
        status = "✓" if actual >= expected_min else "✗"
        print(f"{status} '{query}': {actual} results (expect >= {expected_min})")

    await fulltext_store.close()
```

## 数据量建议

| 存储类型 | 最小量 | 推荐量 | 上限 |
|----------|--------|--------|------|
| FAQ | 100 条 | 500-2000 条 | 10000 条 |
| 文档 | 50 篇 | 200-1000 篇 | 50000 篇 |
| 图谱实体 | 100 个 | 500-5000 个 | 100000 个 |

**注意:**
- 数据量过小会影响检索效果
- 数据量过大需要考虑分片和性能优化

## 常见问题

**Q: 注入后搜索无结果？**

检查：
1. 数据文件是否正确加载
2. 存储是否初始化成功
3. 搜索关键词是否匹配数据内容

**Q: 向量检索不工作？**

需要配置 Embedding Service：
```bash
DASHSCOPE_API_KEY=your-key
```

**Q: 如何清空所有数据？**

```bash
rm -rf data/faq.db data/chroma data/whoosh
```

然后重新执行注入脚本。