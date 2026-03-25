# 故障排除指南

本文档整理了 RAG Pipeline 常见问题及解决方案。

## 目录

- [启动问题](#启动问题)
- [API 调用问题](#api-调用问题)
- [数据检索问题](#数据检索问题)
- [性能问题](#性能问题)
- [配置问题](#配置问题)
- [日志调试](#日志调试)

## 启动问题

### ModuleNotFoundError: No module named 'rag_rag'

**症状:**
```
ModuleNotFoundError: No module named 'rag_rag'
```

**原因:** 项目包未安装

**解决方案:**
```bash
cd rag_rag
pip install -e .
```

### Graph 'rag_agent' failed to load

**症状:**
```
Graph 'rag_agent' failed to load: ModuleNotFoundError
```

**原因:** 依赖未安装或导入错误

**解决方案:**
```bash
# 安装依赖
pip install -r requirements.txt

# 检查导入
python -c "from rag_rag.graph.graph import graph; print('OK')"
```

### Port 8123 already in use

**症状:**
```
OSError: [Errno 98] Address already in use
```

**解决方案:**
```bash
# 使用其他端口
langgraph dev --port 8124

# 或杀死占用进程
# Linux/Mac:
lsof -i :8123
kill -9 <PID>

# Windows:
netstat -ano | findstr :8123
taskkill /PID <PID> /F
```

### Blocking call warning

**症状:**
```
Blocking call to os.mkdir
```

**原因:** 同步代码在异步上下文中执行

**解决方案:**
```bash
# 开发环境使用 --allow-blocking
langgraph dev --port 8123 --allow-blocking

# 或设置环境变量
export BG_JOB_ISOLATED_LOOPS=true
```

## API 调用问题

### Invalid JSON in request body

**症状:**
```json
{"detail": "Invalid JSON in request body"}
```

**原因:** 请求体 JSON 格式错误

**解决方案:**
```bash
# 确保 JSON 格式正确
curl -X POST http://127.0.0.1:8123/runs/wait \
  -H "Content-Type: application/json" \
  -d '{"assistant_id":"rag_agent","input":{"query":"测试"}}'

# Windows CMD 注意转义
curl -X POST http://127.0.0.1:8123/runs/wait \
  -H "Content-Type: application/json" \
  -d "{\"assistant_id\":\"rag_agent\",\"input\":{\"query\":\"测试\"}}"
```

### Connection refused

**症状:**
```
curl: (7) Failed to connect to 127.0.0.1 port 8123
```

**解决方案:**
1. 确认服务已启动
2. 检查端口是否正确
3. 检查防火墙设置

### Timeout on long queries

**症状:** 请求超时，无响应

**解决方案:**
```bash
# 增加超时时间
curl --max-time 120 -X POST ...

# 或使用流式接口
curl -X POST http://127.0.0.1:8123/runs/stream ...
```

## 数据检索问题

### 向量检索不工作

**症状:** vector_results 始终为空

**原因:** 未配置阿里云 Embedding API Key

**解决方案:**
```bash
# 配置 DashScope API Key
export DASHSCOPE_API_KEY="sk-xxx"

# 或在 .env 中配置
echo 'DASHSCOPE_API_KEY=sk-xxx' >> .env
```

**验证配置:**
```bash
# 检查 API Key 是否有效
curl https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding \
  -H "Authorization: Bearer $DASHSCOPE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"text-embedding-v3","input":{"texts":["测试文本"]},"parameters":{"dimension":1024}}'
```

**重新注入向量数据:**
```bash
# 清理旧的向量数据
rm -rf data/chroma

# 使用真实向量重新注入
DASHSCOPE_API_KEY="sk-xxx" python scripts/ingest_electronics_data.py --generate
```

### FAQ 匹配不生效

**症状:** FAQ 问题返回 "知识库中没有找到相关信息"

**排查步骤:**

1. 检查 FAQ 数据是否存在
```bash
ls -la data/faq.db
```

2. 检查数据内容
```python
import sqlite3
conn = sqlite3.connect('data/faq.db')
cursor = conn.execute("SELECT COUNT(*) FROM faqs")
print(f"FAQ count: {cursor.fetchone()[0]}")
conn.close()
```

3. 检查匹配阈值
```yaml
# config/settings.yaml
faq:
  match_threshold: 0.85  # 降低阈值可提高召回
```

### 全文检索无结果

**症状:** retrieval 返回空数组

**排查步骤:**

1. 检查 Whoosh 索引
```bash
ls -la data/whoosh/
```

2. 重建索引
```bash
rm -rf data/whoosh
python scripts/ingest_electronics_data.py
```

3. 检查中文分词
```python
from rag_rag.storage.fulltext_store import ChineseTokenizer
tokenizer = ChineseTokenizer()
tokens = list(tokenizer("iPhone截屏方法"))
print([t.text for t in tokens])
```

### 向量检索不工作

**症状:** vector_results 始终为空

**原因:** 未配置 Embedding API

**解决方案:**
```bash
# 配置 DashScope API Key
export DASHSCOPE_API_KEY="sk-xxx"

# 或在 .env 中配置
echo 'DASHSCOPE_API_KEY=sk-xxx' >> .env
```

### 检索结果不相关

**解决方案:**

1. 调整检索权重
```yaml
retrieval:
  vector_weight: 0.5
  fulltext_weight: 0.3
  graph_weight: 0.2
```

2. 调整检索数量
```yaml
retrieval:
  vector_top_k: 30
  fulltext_top_k: 30
```

3. 调整 Rerank 参数
```yaml
rerank:
  top_k: 10
```

## 性能问题

### 响应时间过长

**排查:**

检查 stage_timing 定位瓶颈：
```json
{
  "stage_timing": {
    "faq_match_ms": 15.2,
    "vector_retrieve_ms": 200.5,  // 这里耗时过长
    "generation_ms": 150.3
  }
}
```

**优化建议:**

| 瓶颈阶段 | 优化方案 |
|----------|----------|
| faq_match | 添加索引，减少数据量 |
| vector_retrieve | 减少 top_k，优化索引 |
| fulltext_retrieve | 优化分词器，减少索引大小 |
| generation | 使用更快的模型，减少 max_tokens |

### 内存占用过高

**解决方案:**

1. 减少向量库缓存
2. 限制并发请求数
3. 使用更小的 embedding 维度

### 高并发失败

**症状:** 并发请求返回 500 错误

**解决方案:**
```bash
# 增加工作线程
langgraph dev --port 8123 --workers 4

# 或使用生产部署
langgraph up --port 8123
```

## 配置问题

### 配置不生效

**症状:** 修改配置后行为未变化

**检查:**

1. 配置文件路径是否正确
```bash
ls config/settings.yaml
```

2. YAML 格式是否正确
```bash
python -c "import yaml; yaml.safe_load(open('config/settings.yaml'))"
```

3. 环境变量是否覆盖
```bash
env | grep -E 'LLM|EMBEDDING|RERANK'
```

### API Key 无效

**症状:**
```
dashscope.errors.AuthenticationError: Invalid API-KEY
```

**解决方案:**
```bash
# 验证 API Key
curl https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation \
  -H "Authorization: Bearer $DASHSCOPE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen-plus","input":{"prompt":"hello"}}'
```

## 日志调试

### 启用详细日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 或在代码中
from rag_rag.core.logging import get_logger
get_logger("rag_rag").setLevel(logging.DEBUG)
```

### 查看请求日志

```bash
# 启动时添加日志
RUST_LOG=debug langgraph dev --port 8123
```

### 分析错误响应

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

### 常用诊断命令

```bash
# 检查服务状态
curl http://127.0.0.1:8123/health

# 检查 API 信息
curl http://127.0.0.1:8123/info

# 检查图配置
curl http://127.0.0.1:8123/threads

# 简单测试
curl -X POST http://127.0.0.1:8123/runs/wait \
  -H "Content-Type: application/json" \
  -d '{"assistant_id":"rag_agent","input":{"query":"test"}}'
```

## 重置与清理

### 完全重置

```bash
# 停止服务
pkill -f langgraph

# 清理数据
rm -rf data/faq.db data/chroma data/whoosh data/sessions.db

# 重新安装
pip install -e . --force-reinstall

# 重新注入数据
python scripts/ingest_electronics_data.py --generate

# 重启服务
langgraph dev --port 8123 --allow-blocking
```

### 清理缓存

```bash
# Python 缓存
find . -type d -name __pycache__ -exec rm -rf {} +

# LangGraph 缓存
rm -rf .langgraph/
```

## 获取帮助

1. 查看日志输出
2. 检查 `errors` 字段
3. 使用调试模式
4. 提交 Issue 并附上错误信息和环境信息