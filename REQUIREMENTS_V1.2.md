# RAG 评测系统 v1.2.0 迭代需求文档

**文档版本**: v1.0  
**创建日期**: 2026-03-19  
**优先级**: P0 + P1  
**预计周期**: 2-3 周

---

## 📋 文档修订记录

| 版本 | 日期 | 作者 | 变更说明 |
|------|------|------|----------|
| v1.0 | 2026-03-19 | AI PM | 初始版本，基于产品评审报告 P0+P1 需求 |

---

## 🎯 迭代目标

基于 v1.1.0 版本的产品评审报告，本次迭代聚焦以下核心能力增强：

1. **数据可视化** - 让评测结果更直观易懂
2. **批量处理** - 提升数据导入导出效率
3. **结果筛选** - 增强数据分析能力
4. **测试保障** - 补充集成测试确保质量
5. **任务调度** - 支持定时/批量评测任务
6. **性能基准** - 建立性能基线，持续优化

---

## 📊 P0-1 数据可视化图表

### 需求描述

在结果查看模块增加可视化图表，让用户直观理解评测结果分布和趋势。

### 功能详情

#### 1.1 评测概览仪表盘

**位置**: 结果查看 Tab 顶部

**图表类型**:

| 图表 | 展示内容 | 实现方式 |
|------|---------|---------|
| 总分卡片 | 平均分数、成功率、总耗时 | 大数字卡片 + 趋势箭头 |
| 分数分布直方图 | 评测分数段分布 (0-0.2, 0.2-0.4...) | 柱状图 |
| 指标雷达图 | 各维度指标得分对比 | 雷达图 |
| 接口对比柱状图 | 双接口各指标得分对比 | 分组柱状图 |

**交互要求**:
- 点击图表可下钻查看明细
- 支持图表类型切换
- 支持导出为 PNG 图片

**技术实现**:
```python
# 推荐使用 plotly 或 matplotlib
# Gradio 原生支持 plotly 图表
import plotly.graph_objects as go
import plotly.express as px

# 示例：分数分布直方图
fig = px.histogram(data, x="score", nbins=10, 
                   title="分数分布", 
                   color_discrete_sequence=['#3B82F6'])
gr.Plot(fig)
```

#### 1.2 单条结果可视化

**位置**: 单条详情查看区域

**展示内容**:
- 各指标得分环形图
- 与平均分对比标识
- 历史趋势（同 query 多次评测）

#### 1.3 趋势分析图

**位置**: 结果查看 Tab 新增子页面

**展示内容**:
- 多次评测的平均分趋势（折线图）
- 各指标得分变化趋势
- 支持按时间范围筛选

### 验收标准

- [ ] 概览仪表盘加载时间 < 2 秒
- [ ] 图表支持响应式缩放
- [ ] 支持导出为 PNG（分辨率≥1920x1080）
- [ ] 暗色模式下图表颜色自适应

---

## 📥 P0-2 批量导入/导出功能

### 需求描述

支持批量导入标注数据和批量导出评测结果，提升数据处理效率。

### 功能详情

#### 2.1 批量导入标注

**入口**: 标注管理 Tab → 批量导入按钮

**支持格式**:

| 格式 | 说明 | 模板 |
|------|------|------|
| Excel (.xlsx) | 推荐格式，支持多列映射 | 提供下载模板 |
| CSV (.csv) | 通用格式，UTF-8 编码 | 提供下载模板 |
| JSON (.json) | 程序生成格式 | Schema 定义 |

**导入流程**:
```
1. 用户上传文件
2. 系统解析并预览前 5 行
3. 用户确认字段映射关系
4. 系统校验数据（去重、必填项）
5. 显示导入预览（成功/失败条数）
6. 用户确认执行导入
7. 显示导入结果报告
```

**字段映射配置**:
```
Excel 列名          →  系统字段
─────────────────────────────────
用户问题            →  query
对话历史            →  conversation_history
标准答案            →  standard_answers
GT 文档              →  gt_documents
FAQ 匹配             →  faq_matched
...
```

**数据校验规则**:
| 规则 | 说明 | 处理方式 |
|------|------|---------|
| query 为空 | 必填项 | 跳过并记录错误 |
| query 重复 | 检测已存在 | 可选：跳过/覆盖/追加版本 |
| 格式错误 | 类型不匹配 | 跳过并记录错误 |
| 编码问题 | 非 UTF-8 | 自动检测转换 |

**技术实现**:
```python
# 使用 pandas 读取文件
import pandas as pd

async def parse_import_file(file_path: str, file_type: str) -> ImportPreview:
    if file_type == "xlsx":
        df = pd.read_excel(file_path)
    elif file_type == "csv":
        df = pd.read_csv(file_path, encoding='utf-8')
    elif file_type == "json":
        df = pd.read_json(file_path)
    
    # 验证必填字段
    required = ['query']
    missing = [col for col in required if col not in df.columns]
    
    # 生成预览
    return ImportPreview(
        total_rows=len(df),
        valid_rows=count_valid,
        invalid_rows=count_invalid,
        preview_data=df.head(5).to_dict(),
        field_mapping=auto_detect_mapping(df.columns)
    )
```

#### 2.2 批量导出标注

**入口**: 标注管理 Tab → 批量导出按钮

**导出选项**:
- [ ] 选择导出范围（全部/筛选后/手动选择）
- [ ] 选择导出格式（Excel/CSV/JSON）
- [ ] 选择导出字段（全选/自定义）
- [ ] 包含元数据（创建时间、版本号等）

#### 2.3 批量导出评测结果

**入口**: 结果查看 Tab → 导出按钮增强

**导出内容**:
- 评测结果明细
- 汇总统计
- 可视化图表（嵌入 Excel）

**Excel 多 Sheet 结构**:
```
📄 export_result.xlsx
├── 概览统计          # 总体指标
├── 结果明细          # 每条评测详情
├── 指标得分          # 各指标分析
├── 问题归因          # bad case 分类
└── 可视化图表        # 嵌入图表
```

### 验收标准

- [ ] 支持导入≥10000 条数据
- [ ] 导入 1000 条数据耗时 < 30 秒
- [ ] 导出 1000 条数据耗时 < 10 秒
- [ ] 导入失败有详细错误报告
- [ ] 提供标准模板下载

---

## 🔍 P0-3 结果筛选/排序增强

### 需求描述

增强评测结果的筛选和排序能力，支持多维度数据分析。

### 功能详情

#### 3.1 多条件筛选

**筛选维度**:

| 维度 | 选项类型 | 示例 |
|------|---------|------|
| 评测时间 | 日期范围选择器 | 2026-03-01 ~ 2026-03-19 |
| 评测状态 | 单选 | 成功/失败/全部 |
| 分数区间 | 滑块 | 0.0 ~ 1.0 |
| RAG 接口 | 多选 | 接口 1/接口 2 |
| 指标得分 | 条件组合 | 事实一致性 < 0.6 |
| 查询内容 | 文本搜索 | 包含关键词 |
| 标注 ID | 文本搜索 | 精确/模糊匹配 |

**筛选 UI 设计**:
```
┌─────────────────────────────────────────────────────────┐
│ 🔍 筛选条件                              [重置] [应用] │
├─────────────────────────────────────────────────────────┤
│ 评测时间：[2026-03-01] 至 [2026-03-19]                 │
│ 评测状态：○ 全部  ● 成功  ○ 失败                       │
│ 分数区间：[====●════════] 0.6 ~ 1.0                    │
│ RAG 接口： ☑ 接口 1  ☑ 接口 2                          │
│ 指标条件：[事实一致性 ▼] [小于 ▼] [0.6 ▼]              │
│ 查询内容：[输入关键词搜索...]                          │
└─────────────────────────────────────────────────────────┘
```

**组合逻辑**:
- 同维度内：OR 关系
- 不同维度间：AND 关系

#### 3.2 多列排序

**支持排序列**:
- 评测分数（升/降）
- 评测时间（升/降）
- 耗时（升/降）
- 各指标得分（升/降）

**交互**:
- 点击表头切换排序方向
- 支持多列排序（Shift+ 点击）
- 当前排序列高亮显示

#### 3.3 筛选结果操作

**批量操作**:
- [ ] 导出筛选结果
- [ ] 加入错题本
- [ ] 重新评测
- [ ] 标记为关注

**保存筛选条件**:
- 支持保存常用筛选组合
- 支持一键加载保存的筛选

### 验收标准

- [ ] 筛选响应时间 < 1 秒（1000 条数据内）
- [ ] 支持≥10 个筛选条件组合
- [ ] 筛选条件可保存和分享
- [ ] URL 可携带筛选参数（便于分享）

---

## 🧪 P0-4 补充集成测试

### 需求描述

补充端到端集成测试，确保各模块协同工作正常。

### 测试范围

#### 4.1 标注管理集成测试

```python
# tests/integration/test_annotation_flow.py

async def test_annotation_crud_flow():
    """测试标注完整 CRUD 流程"""
    # 1. 创建标注
    # 2. 查询标注列表
    # 3. 更新标注
    # 4. 删除标注
    # 5. 验证数据一致性

async def test_annotation_import_export():
    """测试标注导入导出流程"""
    # 1. 准备测试数据
    # 2. 导出为 Excel
    # 3. 删除原数据
    # 4. 从 Excel 导入
    # 5. 验证数据完整性

async def test_annotation_batch_operations():
    """测试批量操作"""
    # 1. 批量创建 100 条标注
    # 2. 批量更新
    # 3. 批量删除
    # 4. 验证性能和数据
```

#### 4.2 评测执行集成测试

```python
# tests/integration/test_evaluation_flow.py

async def test_evaluation_end_to_end():
    """测试评测端到端流程"""
    # 1. 准备标注数据
    # 2. 配置评测指标
    # 3. 启动评测任务
    # 4. 监控进度
    # 5. 验证结果

async def test_concurrent_evaluation():
    """测试并发评测"""
    # 1. 配置高并发（50 并发）
    # 2. 执行评测
    # 3. 验证无任务丢失
    # 4. 验证内存无泄漏

async def test_dual_interface_comparison():
    """测试双接口对比评测"""
    # 1. 配置两个 RAG 接口
    # 2. 执行对比评测
    # 3. 验证结果包含两个接口数据
    # 4. 验证对比分析正确
```

#### 4.3 结果管理集成测试

```python
# tests/integration/test_result_management.py

async def test_result_query_and_filter():
    """测试结果查询和筛选"""
    # 1. 准备评测结果
    # 2. 应用多种筛选条件
    # 3. 验证筛选结果正确

async def test_result_export():
    """测试结果导出"""
    # 1. 执行评测
    # 2. 导出为 JSON/CSV/Excel
    # 3. 验证导出文件内容

async def test_result_visualization():
    """测试结果可视化"""
    # 1. 准备评测结果
    # 2. 生成图表数据
    # 3. 验证图表数据正确
```

#### 4.4 UI 集成测试

```python
# tests/integration/test_ui_flow.py

async def test_ui_annotation_flow():
    """测试 UI 标注流程"""
    # 使用 Playwright 测试 Gradio UI
    # 1. 打开标注管理页面
    # 2. 创建标注
    # 3. 验证显示

async def test_ui_evaluation_flow():
    """测试 UI 评测流程"""
    # 1. 打开评测执行页面
    # 2. 配置评测参数
    # 3. 启动评测
    # 4. 验证进度显示
```

### 测试基础设施

#### 测试数据工厂

```python
# tests/factories.py

class AnnotationFactory:
    @staticmethod
    def create_batch(count: int, **kwargs) -> list[Annotation]:
        """批量创建测试标注"""
        ...

class EvaluationResultFactory:
    @staticmethod
    def create_run_results(count: int) -> EvaluationRun:
        """创建测试评测结果"""
        ...
```

#### Mock 服务

```python
# tests/mocks.py

class MockRAGService:
    """Mock RAG 服务，用于测试"""
    async def query(self, query: str) -> RAGResponse:
        # 返回预定义的响应
        ...

class MockLLMService:
    """Mock LLM 服务，用于测试"""
    async def evaluate(self, context) -> MetricResult:
        # 返回预定义的评测结果
        ...
```

### 验收标准

- [ ] 集成测试覆盖率 ≥ 70%
- [ ] 所有集成测试通过
- [ ] 测试执行时间 < 10 分钟
- [ ] CI/CD 流水线集成测试

---

## 📅 P1-1 评测任务队列/调度

### 需求描述

支持评测任务的队列管理和定时调度，实现自动化评测。

### 功能详情

#### 5.1 任务队列管理

**任务状态**:
```
待执行 (pending) → 执行中 (running) → 完成 (completed)
                              ↓
                        失败 (failed)
                              ↓
                        取消 (cancelled)
```

**队列操作**:
| 操作 | 说明 |
|------|------|
| 创建任务 | 配置评测参数，加入队列 |
| 查看队列 | 显示所有任务状态 |
| 暂停任务 | 暂停执行中的任务 |
| 恢复任务 | 恢复暂停的任务 |
| 取消任务 | 取消待执行/执行中的任务 |
| 重试任务 | 重新执行失败的任务 |
| 删除任务 | 删除已完成任务记录 |

**优先级队列**:
```
优先级：P0(紧急) > P1(高) > P2(中) > P3(低)
默认：P2(中)
```

#### 5.2 定时调度

**调度配置**:
```yaml
schedule:
  type: cron  # 或 interval
  cron: "0 2 * * *"  # 每天凌晨 2 点
  # 或
  interval: 3600  # 每 3600 秒
  
task:
  name: "每日评测"
  annotations:
    source: "all"  # 或 "new_only", "custom"
    filter: {}
  metrics:
    - retrieval_precision
    - answer_relevance
  rag_interface: "default"
  notification:
    on_complete: true
    on_failure: true
    channel: "email"  # 或 "webhook"
```

**调度 UI**:
```
┌─────────────────────────────────────────────────────────┐
│ 📅 创建定时任务                              [保存]    │
├─────────────────────────────────────────────────────────┤
│ 任务名称：[每日 RAG 评测________________]               │
│                                                         │
│ 执行频率：○ 一次性  ● 每天  ○ 每周  ○ 自定义            │
│                                                         │
│ 执行时间：[02:00 ▼]                                     │
│                                                         │
│ 评测范围：○ 全部标注  ● 新增标注  ○ 自定义筛选         │
│                                                         │
│ 评测指标：[☑ 检索精确率] [☑ 答案相关性]...             │
│                                                         │
│ 通知设置：☑ 完成后通知  ☑ 失败时通知                   │
│            通知方式：[邮件 ▼] [Webhook ▼]              │
└─────────────────────────────────────────────────────────┘
```

#### 5.3 任务执行器

**架构设计**:
```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  任务队列   │ ──→ │  任务执行器   │ ──→ │  结果存储   │
│  (SQLite)   │     │  (Worker)    │     │             │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                           ↓
                    ┌──────────────┐
                    │  通知服务     │
                    └──────────────┘
```

**并发控制**:
- 最大并发任务数：可配置（默认 3）
- 单任务最大并发评测数：可配置（默认 10）

#### 5.4 通知服务

**通知渠道**:
| 渠道 | 配置项 |
|------|--------|
| 邮件 | SMTP 配置、收件人列表 |
| Webhook | URL、Secret、事件类型 |
| 钉钉 | Robot URL、签名 |
| 飞书 | Robot URL、签名 |

**通知模板**:
```markdown
## 评测任务完成通知

**任务名称**: 每日 RAG 评测
**执行时间**: 2026-03-19 02:00:00
**耗时**: 15 分 32 秒

**评测结果**:
- 总标注数：1000
- 成功：998
- 失败：2

**平均分**: 0.85 (↑ 0.02)

[查看详细报告](链接)
```

### 验收标准

- [ ] 任务队列持久化（重启不丢失）
- [ ] 定时任务准时执行（误差 < 1 分钟）
- [ ] 支持≥100 个待执行任务
- [ ] 通知发送成功率 ≥ 99%
- [ ] 任务失败有详细日志

---

## 📈 P1-2 性能基准测试功能

### 需求描述

建立性能基准测试体系，持续监控系统性能表现。

### 功能详情

#### 6.1 基准测试指标

**系统性能指标**:

| 指标 | 说明 | 目标值 |
|------|------|--------|
| 标注创建耗时 | 单条标注创建时间 | < 100ms |
| 标注查询耗时 | 列表查询响应时间 | < 500ms |
| 评测启动耗时 | 从点击到任务开始 | < 2s |
| 单条评测耗时 | 单条标注评测时间 | < 5s |
| 并发评测吞吐 | 10 并发下的 QPS | ≥ 2 QPS |
| 结果查询耗时 | 结果列表加载时间 | < 1s |
| 导出耗时 | 1000 条导出时间 | < 10s |
| 导入耗时 | 1000 条导入时间 | < 30s |

**资源使用指标**:

| 指标 | 说明 | 目标值 |
|------|------|--------|
| 内存占用 | 空闲/负载时内存 | < 500MB / < 2GB |
| CPU 使用率 | 负载时 CPU 占用 | < 80% |
| 磁盘 I/O | 读写速度 | 正常范围 |
| 连接数 | 最大并发连接 | ≥ 100 |

#### 6.2 基准测试场景

**标准测试场景**:

```python
# tests/benchmark/test_benchmarks.py

class TestBenchmarks:
    """性能基准测试"""
    
    async def test_annotation_create_latency(self):
        """标注创建延迟测试"""
        # 创建 100 条标注，计算 P50/P95/P99 延迟
        
    async def test_annotation_query_throughput(self):
        """标注查询吞吐测试"""
        # 并发查询，计算 QPS
        
    async def test_evaluation_concurrent_load(self):
        """评测并发负载测试"""
        # 10/50/100 并发评测，验证稳定性
        
    async def test_large_dataset_performance(self):
        """大数据集性能测试"""
        # 10000 条标注下的性能表现
        
    async def test_memory_leak_detection(self):
        """内存泄漏检测"""
        # 长时间运行，监控内存增长
```

**压力测试场景**:

```python
# tests/benchmark/test_stress.py

class StressTest:
    """压力测试"""
    
    async def test_sustained_load(self):
        """持续负载测试 - 24 小时"""
        # 持续 50% 负载运行 24 小时
        
    async def test_spike_load(self):
        """峰值负载测试"""
        # 突然增加到 200% 负载
        
    async def test_recovery_test(self):
        """恢复测试"""
        # 故障后恢复能力测试
```

#### 6.3 基准测试报告

**报告内容**:
```markdown
# 性能基准测试报告

**测试时间**: 2026-03-19 10:00:00
**测试版本**: v1.2.0
**测试环境**: 
- CPU: 8 核
- 内存：16GB
- 存储：SSD

## 测试结果摘要

| 指标 | 当前值 | 基准值 | 状态 |
|------|--------|--------|------|
| 标注创建耗时 | 85ms | 100ms | ✅ |
| 标注查询耗时 | 320ms | 500ms | ✅ |
| 单条评测耗时 | 3.2s | 5s | ✅ |
| 并发吞吐 | 2.5 QPS | 2 QPS | ✅ |

## 详细结果

### 标注创建延迟分布
P50: 75ms
P95: 120ms
P99: 180ms

### 并发评测性能
10 并发：2.5 QPS
50 并发：8.2 QPS
100 并发：12.1 QPS

## 资源使用

内存峰值：1.2GB
CPU 峰值：65%

## 问题与建议

无严重性能问题。
建议优化大数据集下的查询性能。
```

#### 6.4 性能监控仪表板

**实时监控**:
- 当前 QPS
- 平均响应时间
- 错误率
- 资源使用率

**历史趋势**:
- 性能指标趋势图
- 版本对比

**告警配置**:
```yaml
alerts:
  - name: "高延迟告警"
    metric: "avg_response_time"
    threshold: 1000  # ms
    duration: 5m  # 持续 5 分钟
    
  - name: "高错误率告警"
    metric: "error_rate"
    threshold: 0.05  # 5%
    duration: 2m
```

### 验收标准

- [ ] 基准测试可一键执行
- [ ] 生成详细测试报告
- [ ] 性能数据持久化存储
- [ ] 支持版本间性能对比
- [ ] 性能回归测试集成到 CI/CD

---

## 🛠️ 技术实现方案

### 依赖库新增

```txt
# requirements.txt 新增

# 数据可视化
plotly>=5.18.0
matplotlib>=3.8.0

# Excel 处理
openpyxl>=3.1.0
xlsxwriter>=3.1.0

# 任务调度
apscheduler>=3.10.0

# 性能测试
pytest-benchmark>=4.0.0
locust>=2.20.0

# 内存分析
memory-profiler>=0.61.0
```

### 数据库 Schema 变更

```sql
-- 任务队列表
CREATE TABLE IF NOT EXISTS evaluation_tasks (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL,  -- pending, running, completed, failed, cancelled
    priority INTEGER DEFAULT 2,  -- 0-3
    config TEXT NOT NULL,  -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    schedule_id TEXT  -- 关联的定时任务 ID
);

-- 定时任务表
CREATE TABLE IF NOT EXISTS scheduled_tasks (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    cron_expression TEXT NOT NULL,
    enabled BOOLEAN DEFAULT TRUE,
    last_run_at TIMESTAMP,
    next_run_at TIMESTAMP,
    config TEXT NOT NULL  -- JSON
);

-- 性能基准表
CREATE TABLE IF NOT EXISTS performance_benchmarks (
    id TEXT PRIMARY KEY,
    test_name TEXT NOT NULL,
    test_type TEXT NOT NULL,  -- benchmark, stress, load
    version TEXT NOT NULL,
    metrics TEXT NOT NULL,  -- JSON
    environment TEXT NOT NULL,  -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 核心模块设计

#### 任务队列模块

```
src/
└── scheduler/
    ├── __init__.py
    ├── task_queue.py       # 任务队列管理
    ├── scheduler.py        # 定时调度器
    ├── worker.py           # 任务执行器
    ├── notification.py     # 通知服务
    └── models.py           # 数据模型
```

#### 可视化模块

```
src/
└── ui/
    └── components/
        └── visualization.py  # 图表组件
```

#### 基准测试模块

```
tests/
└── benchmark/
    ├── __init__.py
    ├── benchmarks.py       # 基准测试
    ├── stress.py           # 压力测试
    ├── reporter.py         # 报告生成
    └── monitor.py          # 性能监控
```

---

## 📅 迭代计划

### 第一周：P0 功能

| 天 | 任务 | 产出 |
|----|------|------|
| 1-2 | 数据可视化图表 | visualization.py, 图表组件 |
| 3-4 | 批量导入/导出 | import/export 模块，UI 组件 |
| 5 | 结果筛选/排序 | 筛选组件，后端查询优化 |
| 6-7 | 集成测试 | 测试用例，CI 集成 |

### 第二周：P1 功能

| 天 | 任务 | 产出 |
|----|------|------|
| 1-3 | 任务队列/调度 | scheduler 模块，UI 组件 |
| 4-5 | 性能基准测试 | benchmark 测试套件 |
| 6 | 联调测试 | 端到端验证 |
| 7 | 文档与发布 | 更新文档，发布 v1.2.0 |

---

## ✅ 验收清单

### P0-1 数据可视化
- [ ] 概览仪表盘实现
- [ ] 分数分布直方图
- [ ] 指标雷达图
- [ ] 接口对比柱状图
- [ ] 图表导出功能

### P0-2 批量导入/导出
- [ ] Excel 导入
- [ ] CSV 导入
- [ ] JSON 导入
- [ ] 字段映射配置
- [ ] 导入预览
- [ ] 批量导出
- [ ] 模板下载

### P0-3 结果筛选/排序
- [ ] 多条件筛选
- [ ] 多列排序
- [ ] 筛选条件保存
- [ ] URL 参数分享

### P0-4 集成测试
- [ ] 标注管理集成测试
- [ ] 评测执行集成测试
- [ ] 结果管理集成测试
- [ ] UI 集成测试
- [ ] 覆盖率 ≥ 70%

### P1-1 任务队列/调度
- [ ] 任务队列管理
- [ ] 定时调度
- [ ] 任务执行器
- [ ] 通知服务
- [ ] 调度 UI

### P1-2 性能基准测试
- [ ] 基准测试指标
- [ ] 标准测试场景
- [ ] 压力测试场景
- [ ] 测试报告生成
- [ ] 性能监控仪表板

---

## 📎 附录

### A. 参考文档
- [Plotly Gradio 集成](https://www.gradio.app/docs/gradio/plot)
- [APScheduler 文档](https://apscheduler.readthedocs.io/)
- [pytest-benchmark 文档](https://pytest-benchmark.readthedocs.io/)

### B. 相关文件
- `src/ui/components/visualization.py` (新建)
- `src/scheduler/` (新建目录)
- `tests/integration/` (新建目录)
- `tests/benchmark/` (新建目录)

### C. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 图表库兼容性问题 | 中 | 提前验证，准备备选方案 |
| 大数据导入性能 | 中 | 分块处理，进度显示 |
| 定时任务可靠性 | 高 | 持久化存储，重试机制 |
| 性能测试环境差异 | 低 | 标准化测试环境配置 |

---

**文档结束**

*下一步：技术评审 → 任务拆解 → 开发实施*
