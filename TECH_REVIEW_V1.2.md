# RAG 评测系统 v1.2.0 技术评审报告

**评审人**: 技术专家组  
**评审日期**: 2026-03-19  
**评审版本**: v1.2.0 需求文档  
**评审结论**: ✅ 原则通过，需调整部分技术方案

---

## 📊 评审总览

| 维度 | 评分 | 状态 |
|------|------|------|
| 架构设计合理性 | 8.5/10 | ✅ 良好 |
| 技术选型 | 8.0/10 | ⚠️ 需调整 |
| 性能与可扩展性 | 7.5/10 | ⚠️ 有风险 |
| 安全性 | 8.0/10 | ✅ 良好 |
| 实施可行性 | 8.5/10 | ✅ 良好 |

**综合评分**: **8.1/10** - 原则通过，建议按评审意见调整后实施

---

## 一、架构设计评审

### 1.1 整体架构评价 ✅

**优点**:
```
┌─────────────────────────────────────────────────────────────┐
│                      UI Layer (Gradio)                       │
├─────────────────────────────────────────────────────────────┤
│                   Business Logic Layer                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │ Annotation  │ │ Evaluation  │ │ Scheduler (新增)    │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                      Storage Layer                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │ LocalStorage│ │SQLiteStorage│ │ TaskQueue (新增)    │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

- ✅ 分层清晰，职责明确
- ✅ 符合现有设计模式体系
- ✅ 新增模块与现有架构融合良好

**改进建议**:
```
⚠️ 问题：缺少服务层抽象，UI 直接调用业务逻辑

建议增加 Service 层：
src/
├── services/           # 新增
│   ├── annotation_service.py
│   ├── evaluation_service.py
│   ├── task_service.py
│   └── export_service.py
```

**理由**:
- 便于未来 REST API 扩展
- 统一事务管理
- 便于单元测试

---

### 1.2 任务队列架构评审 ⚠️

**需求方案**:
```python
# 需求文档中的方案
SQLite 表存储任务 → APScheduler 调度 → Worker 执行
```

**技术风险**:

| 风险 | 严重性 | 说明 |
|------|--------|------|
| SQLite 并发写限制 | 🔴 高 | WAL 模式下仍有锁竞争 |
| 任务状态一致性 | 🟡 中 | 无分布式锁，多 Worker 可能重复执行 |
| 任务丢失风险 | 🟡 中 | 进程崩溃时任务状态可能不一致 |
| 内存队列不可靠 | 🟡 中 | 重启后内存中任务丢失 |

**改进方案**:

```python
# 方案 A: 使用成熟消息队列（推荐）
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  APScheduler │ ──→ │   Redis      │ ──→ │   Worker    │
│  (调度触发)  │     │  (任务队列)   │     │  (执行器)   │
└─────────────┘     └──────────────┘     └─────────────┘

优点:
- 高可靠，支持持久化
- 支持多 Worker 水平扩展
- 成熟的消息确认机制

缺点:
- 增加 Redis 依赖
- 部署复杂度增加


# 方案 B: SQLite + 事务锁（次选，适合单机）
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  APScheduler │ ──→ │  SQLite +    │ ──→ │   Worker    │
│  (调度触发)  │     │  行级锁      │     │  (执行器)   │
└─────────────┘     └──────────────┘     └─────────────┘

实现要点:
- 使用 SELECT ... FOR UPDATE 锁定任务
- 任务状态机：pending → claimed → running → completed
- 心跳检测 + 超时释放
```

**推荐决策**: 
- **短期 (v1.2.0)**: 方案 B（SQLite + 行级锁），满足单机场景
- **长期 (v2.0)**: 方案 A（Redis），支持分布式部署

---

### 1.3 数据库 Schema 评审 ⚠️

**需求文档中的 Schema**:
```sql
CREATE TABLE evaluation_tasks (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    config TEXT NOT NULL,  -- ⚠️ JSON 存储
    ...
);
```

**问题**:
1. ⚠️ `config` 字段存 JSON，查询困难
2. ⚠️ 缺少外键约束
3. ⚠️ 缺少索引定义
4. ⚠️ 缺少软删除字段

**改进 Schema**:
```sql
-- 任务队列表（改进版）
CREATE TABLE IF NOT EXISTS evaluation_tasks (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending', 'claimed', 'running', 'completed', 'failed', 'cancelled')),
    priority INTEGER DEFAULT 2 CHECK(priority >= 0 AND priority <= 3),
    
    -- 结构化配置字段（便于查询）
    annotation_source TEXT NOT NULL,  -- all / new_only / custom
    annotation_filter TEXT,           -- JSON 过滤条件
    metrics_config TEXT NOT NULL,     -- JSON 指标配置
    rag_interface TEXT,               -- 接口名称
    
    -- 调度相关
    schedule_id TEXT REFERENCES scheduled_tasks(id),
    claimed_by TEXT,                  -- Worker ID
    claimed_at TIMESTAMP,
    heartbeat_at TIMESTAMP,           -- ⚠️ 心跳检测
    
    -- 执行结果
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    
    -- 元数据
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_deleted INTEGER DEFAULT 0
);

-- 索引（关键！）
CREATE INDEX idx_tasks_status_priority ON evaluation_tasks(status, priority);
CREATE INDEX idx_tasks_schedule ON evaluation_tasks(schedule_id) WHERE schedule_id IS NOT NULL;
CREATE INDEX idx_tasks_claimed ON evaluation_tasks(claimed_at) WHERE claimed_by IS NOT NULL;
CREATE INDEX idx_tasks_created ON evaluation_tasks(created_at);

-- 定时任务表
CREATE TABLE IF NOT EXISTS scheduled_tasks (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    cron_expression TEXT NOT NULL,
    timezone TEXT DEFAULT 'Asia/Shanghai',
    enabled INTEGER DEFAULT 1,
    
    -- 任务配置
    task_config TEXT NOT NULL,  -- JSON
    
    -- 调度状态
    last_run_at TIMESTAMP,
    last_run_status TEXT,
    last_run_duration_ms INTEGER,
    next_run_at TIMESTAMP,
    
    -- 元数据
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 性能基准表
CREATE TABLE IF NOT EXISTS performance_benchmarks (
    id TEXT PRIMARY KEY,
    test_name TEXT NOT NULL,
    test_type TEXT NOT NULL CHECK(test_type IN ('benchmark', 'stress', 'load')),
    version TEXT NOT NULL,
    
    -- 指标数据（结构化）
    avg_response_time_ms REAL,
    p95_response_time_ms REAL,
    p99_response_time_ms REAL,
    throughput_qps REAL,
    error_rate REAL,
    memory_peak_mb REAL,
    cpu_peak_percent REAL,
    
    -- 环境信息
    environment_config TEXT NOT NULL,  -- JSON
    hardware_info TEXT NOT NULL,       -- JSON
    
    -- 元数据
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT
);

CREATE INDEX idx_benchmarks_version ON performance_benchmarks(version);
CREATE INDEX idx_benchmarks_test ON performance_benchmarks(test_name, test_type);
```

---

## 二、技术选型评审

### 2.1 可视化库选型 ✅

| 选项 | 评分 | 评价 |
|------|------|------|
| Plotly | ✅ 推荐 | Gradio 原生支持，交互性好 |
| Matplotlib | ⚠️ 备选 | 静态图，需额外处理交互 |
| ECharts | ❌ 不推荐 | 需自定义集成，工作量大 |

**结论**: Plotly 选择正确

**注意事项**:
```python
# ⚠️ 性能注意：大数据集时采样
import plotly.express as px

# ❌ 避免：10000+ 数据点直接渲染
fig = px.scatter(large_dataset, x='x', y='y')

# ✅ 推荐：采样或聚合
sampled = large_dataset.sample(n=1000) if len(large_dataset) > 1000 else large_dataset
fig = px.scatter(sampled, x='x', y='y')

# 或使用聚合图表
fig = px.histogram(large_dataset, x='score', nbins=50)
```

---

### 2.2 Excel 处理库选型 ⚠️

**需求文档**: `openpyxl + xlsxwriter`

**问题**:
| 库 | 问题 |
|------|------|
| openpyxl | 内存占用高，万行级数据可能 OOM |
| xlsxwriter | 只写，不能读取 |

**改进建议**:
```python
# 方案 A: pandas + openpyxl（中小数据量）
import pandas as pd
df = pd.read_excel(file, nrows=1000)  # 限制读取量

# 方案 B: openpyxl 流式读取（大数据量）
from openpyxl import load_workbook

wb = load_workbook(file, read_only=True, data_only=True)
ws = wb.active
for row in ws.iter_rows(values_only=True):
    process(row)  # 逐行处理，不占内存

# 方案 C: csvkit（CSV 专用）
import csv
with open(file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        process(row)
```

**推荐**:
- < 10000 行：pandas + openpyxl
- ≥ 10000 行：openpyxl 流式读取
- CSV 文件：csv 标准库

---

### 2.3 任务调度库选型 ✅

**APScheduler 评价**:

| 维度 | 评分 | 说明 |
|------|------|------|
| 功能完整性 | 9/10 | Cron/Interval/Date 全支持 |
| 持久化 | 8/10 | 支持 SQLite/Redis 存储 |
| 分布式 | 5/10 | 需配合 Redis 实现 |
| 易用性 | 9/10 | API 简洁 |

**使用建议**:
```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlite import SQLiteJobStore

# ✅ 推荐：使用持久化 JobStore
jobstores = {
    'default': SQLiteJobStore(db_path='./data/scheduler.db')
}
scheduler = AsyncIOScheduler(jobstores=jobstores)

# 添加任务
scheduler.add_job(
    run_evaluation_task,
    trigger='cron',
    hour=2,
    minute=0,
    id='daily_evaluation',
    replace_existing=True,
    misfire_grace_time=3600  # ⚠️ 错过执行的容忍时间
)
```

---

### 2.4 性能测试工具选型 ✅

| 工具 | 用途 | 推荐度 |
|------|------|--------|
| pytest-benchmark | 单元测试级基准 | ✅ 推荐 |
| locust | 负载/压力测试 | ✅ 推荐 |
| memory-profiler | 内存分析 | ✅ 推荐 |
| py-spy | 生产环境性能分析 | ⚠️ 可选 |

**配置建议**:
```python
# pytest.ini
[pytest]
addopts = --benchmark-autosave --benchmark-compare
benchmark_min_rounds = 5
benchmark_min_time = 0.1

# conftest.py
@pytest.fixture
def benchmark_env():
    """标准化测试环境"""
    return {
        "cpu_count": os.cpu_count(),
        "memory_gb": psutil.virtual_memory().total / 1024**3,
        "python_version": sys.version,
    }
```

---

## 三、性能与可扩展性评审

### 3.1 批量导入性能风险 🔴

**需求目标**:
- 导入 1000 条 < 30 秒
- 支持 ≥ 10000 条

**当前架构风险**:
```python
# ❌ 风险代码模式
async def import_annotations(data_list):
    for item in data_list:
        await handler.create(item)  # 逐条写入，N 次 DB 操作
```

**性能分析**:
```
1000 条 × 100ms/条 = 100 秒 ❌ 不达标
```

**优化方案**:
```python
# ✅ 批量写入（事务）
async def import_annotations_batch(data_list, batch_size=100):
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i+batch_size]
        async with storage.transaction():  # 事务包裹
            await storage.save_batch("annotations", batch)
        # 进度回调
        progress_callback(i + len(batch), len(data_list))

# 性能对比
1000 条 ÷ 100 条/批 × 50ms/批 = 500ms ✅ 达标
```

**SQLite 批量写入优化**:
```python
# 关键配置
PRAGMA journal_mode=WAL;      # WAL 模式
PRAGMA synchronous=NORMAL;    # 降低同步级别
PRAGMA cache_size=-64000;     # 64MB 缓存
PRAGMA temp_store=MEMORY;     # 临时表存内存
```

---

### 3.2 结果筛选查询性能风险 🟡

**需求**: 筛选响应 < 1 秒（1000 条数据）

**当前实现问题**:
```python
# sqlite_storage.py 现有代码
async def get_all(self, collection, filters, limit, offset):
    # ⚠️ 问题：先取数据，再 Python 内存过滤
    rows = await self._db.execute(query, params)
    results = []
    for row in rows:
        data = json.loads(row["data"])
        if filters:
            # ❌ 内存过滤，效率低
            if not match:
                continue
```

**优化方案**:
```python
# 方案 A: SQLite JSON 扩展（SQLite 3.38+）
async def get_all_with_json_filter(self, collection, filters, limit, offset):
    conditions = ["collection = ?", "is_deleted = 0"]
    params = [collection]
    
    for key, value in filters.items():
        conditions.append("json_extract(data, ?) = ?")
        params.extend([f"$.{key}", value])
    
    query = f"SELECT data FROM records WHERE {' AND '.join(conditions)} LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    cursor = await self._db.execute(query, params)
    return [json.loads(row["data"]) for row in await cursor.fetchall()]

# 方案 B: 添加虚拟列 + 索引
ALTER TABLE records ADD COLUMN query_text TEXT 
    GENERATED ALWAYS AS (json_extract(data, '$.query')) STORED;
CREATE INDEX idx_query_text ON records(query_text);
```

---

### 3.3 并发评测扩展性 🟡

**当前架构**:
```python
# runner.py 现有代码
semaphore = asyncio.Semaphore(self.max_concurrent)
tasks = []
for annotation in annotations:
    task = self._evaluate_single(..., semaphore=semaphore)
    tasks.append(task)
results = await asyncio.gather(*tasks)
```

**问题**:
1. ⚠️ 大量 Task 对象一次性创建，内存占用高
2. ⚠️ 无任务优先级
3. ⚠️ 失败任务无重试

**优化方案**:
```python
# 使用 asyncio.Queue 实现任务池
class EvaluationTaskQueue:
    def __init__(self, max_concurrent=10):
        self.queue = asyncio.Queue()
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.results = []
        
    async def worker(self, worker_id):
        while True:
            try:
                annotation = await self.queue.get()
                result = await self._evaluate_single(annotation)
                self.results.append(result)
            finally:
                self.queue.task_done()
    
    async def run(self, annotations):
        # 启动 Worker
        workers = [
            asyncio.create_task(self.worker(i))
            for i in range(self.max_concurrent)
        ]
        
        # 入队任务
        for ann in annotations:
            await self.queue.put(ann)
        
        # 等待完成
        await self.queue.join()
        
        # 停止 Worker
        for _ in range(self.max_concurrent):
            await self.queue.put(None)
        await asyncio.gather(*workers)
```

---

### 3.4 内存泄漏风险 🟡

**风险点**:
```python
# ❌ 潜在泄漏：进程池未正确关闭
self._process_pool = ProcessPoolExecutor(max_workers=4)
# 如果异常退出，可能未关闭

# ❌ 潜在泄漏：异步任务引用未释放
self._tasks.append(asyncio.create_task(...))
```

**改进建议**:
```python
# 使用 async context manager
class EvaluationRunner:
    async def __aenter__(self):
        self._process_pool = ProcessPoolExecutor(max_workers=4)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._process_pool:
            self._process_pool.shutdown(wait=True)
        return False

# 使用方式
async with EvaluationRunner() as runner:
    run = await runner.run(annotations)
```

---

## 四、安全性评审

### 4.1 文件导入安全 🟡

**风险**:
```python
# ❌ 路径遍历风险
file_path = f"./uploads/{filename}"  # 用户可传入 ../../etc/passwd

# ❌ 文件类型风险
if file_type == "xlsx":  # 仅检查扩展名，可伪造
```

**改进**:
```python
import secrets
from pathlib import Path
import magic  # python-magic 库

# ✅ 安全文件路径
def safe_upload_path(filename: str) -> Path:
    # 生成随机文件名
    safe_name = f"{secrets.token_hex(16)}_{Path(filename).name}"
    upload_dir = Path("./data/uploads").resolve()
    return (upload_dir / safe_name).resolve()

# ✅ 文件类型验证
def validate_file_type(file_path: Path, allowed_types: list[str]) -> bool:
    mime = magic.from_file(str(file_path), mime=True)
    allowed_mimes = {
        "xlsx": ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"],
        "csv": ["text/csv", "application/csv"],
        "json": ["application/json"],
    }
    return mime in allowed_mimes.get(file_path.suffix[1:], [])
```

---

### 4.2 SQL 注入风险 ✅

**当前实现**:
```python
# ✅ 参数化查询，安全
await self._db.execute(
    "SELECT data FROM records WHERE id = ? AND collection = ?",
    (record_id, collection)
)
```

**注意**: 避免字符串拼接 SQL
```python
# ❌ 危险
query = f"SELECT * FROM records WHERE id = '{user_input}'"

# ✅ 安全
await self._db.execute("SELECT * FROM records WHERE id = ?", (user_input,))
```

---

### 4.3 敏感信息保护 🟡

**风险**:
```python
# ❌ 日志可能包含敏感信息
logger.info(f"Evaluation config: {config}")  # 可能包含 API Key
```

**改进**:
```python
from ..core.logging import sanitize_for_log

# ✅ 脱敏日志
logger.info(f"Evaluation config: {sanitize_for_log(config)}")

# 脱敏函数
def sanitize_for_log(data: dict) -> dict:
    sensitive_keys = ['api_key', 'secret', 'password', 'token']
    sanitized = data.copy()
    for key in sensitive_keys:
        if key in sanitized:
            sanitized[key] = '***REDACTED***'
    return sanitized
```

---

## 五、实施建议

### 5.1 开发优先级调整

**原计划**:
```
Week 1: P0-1 可视化 → P0-2 导入导出 → P0-3 筛选 → P0-4 测试
Week 2: P1-1 任务调度 → P1-2 性能基准
```

**调整后建议**:
```
Week 1: 
  Day 1-2: 数据库 Schema 变更 + 基础模型 ⚠️ 必须先做
  Day 3-4: P0-2 批量导入导出（含安全加固）
  Day 5: P0-3 结果筛选（含查询优化）

Week 2:
  Day 1-2: P0-1 数据可视化
  Day 3-4: P1-1 任务队列（简化版）
  Day 5: P0-4 集成测试

Week 3:
  Day 1-2: P1-2 性能基准测试
  Day 3: 联调测试
  Day 4-5: 文档与发布
```

**调整理由**:
1. 数据库 Schema 是基础，必须先定义
2. 导入导出依赖存储层优化
3. 可视化可独立开发，放后
4. 任务调度复杂度高，预留缓冲

---

### 5.2 关键技术债务

| 债务 | 影响 | 偿还计划 |
|------|------|---------|
| SQLite 并发限制 | 中 | v1.2 接受，v2.0 迁移 Redis |
| JSON 字段查询 | 中 | v1.2 用虚拟列，v2.0 结构化 |
| 无 Service 层 | 低 | v1.3 重构添加 |
| 进程池管理 | 中 | v1.2 增加 context manager |

---

### 5.3 测试策略建议

**测试金字塔**:
```
           /\
          /  \       E2E 测试 (10%)
         /────\      - UI 流程测试
        /      \     - 关键用户旅程
       /────────\
      /          \   集成测试 (30%)
     /────────────\  - 模块间协作
    /              \ - API 契约测试
   /────────────────\
  /                  \ 单元测试 (60%)
 /────────────────────\ - 业务逻辑
                       - 工具函数
```

**覆盖率目标**:
```
目标：总体 70%+
- src/core/*: 90%+ (核心逻辑)
- src/storage/*: 85%+ (数据层)
- src/evaluation/*: 80%+ (评测引擎)
- src/ui/*: 50%+ (UI 组件，难测试)
- src/scheduler/*: 75%+ (新增模块)
```

---

## 六、风险清单

### 🔴 高风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| SQLite 并发写冲突 | 中 | 高 | WAL 模式 + 行级锁 + 重试 |
| 大数据导入 OOM | 中 | 高 | 流式处理 + 分批事务 |
| 定时任务重复执行 | 低 | 高 | 任务状态机 + 唯一锁 |

### 🟡 中风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 图表渲染性能 | 中 | 中 | 数据采样 + 懒加载 |
| 内存泄漏 | 低 | 中 | context manager + 监控 |
| 文件上传安全 | 中 | 中 | MIME 验证 + 随机文件名 |

### 🟢 低风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 依赖库兼容 | 低 | 低 | 锁定版本 + 测试 |
| 文档滞后 | 中 | 低 | 文档即代码 |

---

## 七、评审结论

### ✅ 通过项

1. 整体架构设计合理
2. 技术选型基本正确
3. 功能需求清晰可实施
4. 性能目标可达成

### ⚠️ 需调整项

1. **任务队列架构**: 增加行级锁机制，防止重复执行
2. **数据库 Schema**: 按评审意见补充索引和约束
3. **批量导入**: 实现流式处理，避免 OOM
4. **查询优化**: 使用 SQLite JSON 扩展或虚拟列
5. **安全加固**: 文件上传 MIME 验证 + 路径 sanitization

### ❌ 否决项

无

---

## 八、后续行动

| 行动项 | 负责人 | 截止日期 |
|--------|--------|---------|
| 更新数据库 Schema | 开发团队 | 2026-03-20 |
| 实现任务行级锁 | 开发团队 | 2026-03-21 |
| 添加文件安全验证 | 开发团队 | 2026-03-21 |
| 优化批量导入逻辑 | 开发团队 | 2026-03-22 |
| 补充集成测试框架 | 测试团队 | 2026-03-24 |

---

**评审结论**: ✅ **原则通过，按评审意见调整后实施**

*下一步：技术方案修订 → 任务拆解 → 开发实施*

---

**附录 A: 关键代码模板**

```python
# 任务行级锁实现
async def claim_task(task_id: str, worker_id: str) -> Optional[Task]:
    async with db.transaction():
        cursor = await db.execute(
            """
            UPDATE evaluation_tasks 
            SET status = 'claimed', claimed_by = ?, claimed_at = ?
            WHERE id = ? AND status = 'pending'
            """,
            (worker_id, datetime.now().isoformat(), task_id)
        )
        if cursor.rowcount == 0:
            return None  # 已被其他 Worker 抢占
        
        return await get_task(task_id)

# 心跳检测
async def task_heartbeat(task_id: str):
    await db.execute(
        "UPDATE evaluation_tasks SET heartbeat_at = ? WHERE id = ?",
        (datetime.now().isoformat(), task_id)
    )

# 超时任务释放
async def release_stale_tasks(timeout_seconds: int = 300):
    cutoff = (datetime.now() - timedelta(seconds=timeout_seconds)).isoformat()
    await db.execute(
        """
        UPDATE evaluation_tasks 
        SET status = 'pending', claimed_by = NULL, claimed_at = NULL
        WHERE status = 'claimed' AND heartbeat_at < ?
        """,
        (cutoff,)
    )
```

---

*报告结束*
