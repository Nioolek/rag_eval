# 配置说明

## 配置文件

RAG Pipeline 使用两层配置：

1. **环境变量** (`.env`) - 敏感信息、部署相关
2. **业务配置** (`config/settings.yaml`) - 业务逻辑参数

## 环境变量 (.env)

### 必填变量

| 变量 | 说明 | 示例 |
|------|------|------|
| `DASHSCOPE_API_KEY` | 阿里云 DashScope API Key | `sk-xxx` |

### LLM 配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_MODEL` | LLM 模型名称 | `qwen-plus` |
| `LLM_TEMPERATURE` | 生成温度 | `0.7` |
| `LLM_MAX_TOKENS` | 最大生成 Token | `2048` |

### Embedding 配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `EMBEDDING_MODEL` | Embedding 模型 | `text-embedding-v3` |
| `EMBEDDING_DIMENSION` | 向量维度 | `1024` |

### Rerank 配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `RERANK_MODEL` | Rerank 模型 | `gte-rerank` |

### 存储配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DATA_DIR` | 数据存储目录 | `./data` |
| `FAQ_DB_PATH` | FAQ 数据库路径 | `./data/faq.db` |
| `SESSION_DB_PATH` | 会话数据库路径 | `./data/sessions.db` |

### Neo4j 配置 (可选)

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `NEO4J_URI` | Neo4j 连接地址 | `bolt://localhost:7687` |
| `NEO4J_USER` | 用户名 | `neo4j` |
| `NEO4J_PASSWORD` | 密码 | `password` |

### API 认证

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `EVAL_API_KEY` | 评测系统 API Key | - |
| `USER_API_KEY` | 用户 API Key | - |

### 速率限制

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DASHSCOPE_QPM` | 每分钟请求数限制 | `60` |

## 业务配置 (config/settings.yaml)

### LLM 配置

```yaml
llm:
  model: "qwen-plus"        # 模型名称
  temperature: 0.7          # 生成温度 (0-1)
  max_tokens: 2048          # 最大生成 Token
  enable_thinking: false    # 是否启用思考模式
  timeout: 60               # 超时时间 (秒)
```

**可用模型:**
- `qwen-plus` - 推荐用于生产
- `qwen-turbo` - 快速响应，用于降级
- `qwen-max` - 最强能力，高成本

### Embedding 配置

```yaml
embedding:
  model: "text-embedding-v3"  # Embedding 模型
  dimension: 1024             # 向量维度
  batch_size: 20              # 批处理大小
  timeout: 30                 # 超时时间 (秒)
```

**可用模型:**
- `text-embedding-v3` - 推荐，1024 维
- `text-embedding-v2` - 旧版，1536 维

### Rerank 配置

```yaml
rerank:
  model: "gte-rerank"  # Rerank 模型
  top_k: 5             # 返回 Top K 结果
  timeout: 30          # 超时时间 (秒)
```

### 存储配置

```yaml
storage:
  data_dir: "./data"
  neo4j_uri: "bolt://localhost:7687"
  neo4j_user: "neo4j"
  neo4j_password: "password"
```

### 检索配置

```yaml
retrieval:
  # 各数据源检索数量
  vector_top_k: 20      # 向量检索数量
  fulltext_top_k: 20    # 全文检索数量
  graph_top_k: 10       # 图谱检索数量

  # 融合权重 (总和应为 1.0)
  vector_weight: 0.5    # 向量权重
  fulltext_weight: 0.3  # 全文权重
  graph_weight: 0.2     # 图谱权重
```

**权重说明:**
- 向量检索擅长语义相似
- 全文检索擅长关键词匹配
- 图谱检索擅长实体关系

### 会话配置

```yaml
session:
  max_history_turns: 5  # 最大历史轮数
```

### FAQ 配置

```yaml
faq:
  match_threshold: 0.85         # 精确匹配阈值
  enable_semantic_match: true   # 启用语义匹配
```

**阈值说明:**
- `0.9+` - 高精度，少量匹配
- `0.85` - 推荐，平衡精度和召回
- `0.7` - 高召回，可能误匹配

### 拒答配置

```yaml
refusal:
  out_of_domain_threshold: 0.3   # 域外判断阈值
  sensitive_words_enabled: true  # 启用敏感词检测
```

### 降级配置

```yaml
degradation:
  circuit_breaker:
    failure_threshold: 3   # 失败次数阈值
    recovery_timeout: 60   # 恢复超时 (秒)
```

## 配置热更新

`config/settings.yaml` 支持热更新，修改后自动生效，无需重启服务。

**支持的参数:**
- LLM temperature, max_tokens
- 检索 top_k 和权重
- FAQ 匹配阈值
- 拒答阈值

**不支持的参数 (需要重启):**
- 模型名称
- 向量维度
- 数据库连接

## 配置优先级

1. 环境变量 (最高)
2. `config/settings.yaml`
3. 代码默认值 (最低)

示例：环境变量覆盖配置文件：

```bash
# .env
LLM_TEMPERATURE=0.5

# config/settings.yaml
llm:
  temperature: 0.7  # 会被环境变量覆盖
```

## 不同环境配置

### 开发环境

```yaml
llm:
  temperature: 0.9  # 更随机，便于测试
  model: "qwen-turbo"  # 快速响应

faq:
  match_threshold: 0.7  # 宽松匹配
```

### 生产环境

```yaml
llm:
  temperature: 0.7  # 稳定输出
  model: "qwen-plus"

faq:
  match_threshold: 0.85  # 精确匹配

degradation:
  circuit_breaker:
    failure_threshold: 3
    recovery_timeout: 60
```

### 高精度场景

```yaml
retrieval:
  vector_top_k: 30
  fulltext_top_k: 30

rerank:
  top_k: 10

refusal:
  out_of_domain_threshold: 0.5  # 更严格
```

## 配置验证

启动时自动验证配置：

```bash
cd rag_rag
python -c "
from rag_rag.core.config import get_config
config = get_config()
print(f'LLM Model: {config.llm.model}')
print(f'Embedding Dimension: {config.embedding.dimension}')
print(f'Vector Top K: {config.retrieval.vector_top_k}')
"
```

## 敏感信息管理

**不要**将敏感信息提交到版本控制：

```bash
# .gitignore
.env
*.pem
*_key.*
```

使用环境变量或密钥管理服务：

```bash
# 从环境变量读取
export DASHSCOPE_API_KEY="sk-xxx"

# 或从文件读取
export DASHSCOPE_API_KEY=$(cat /path/to/key.txt)
```