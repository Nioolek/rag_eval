# RAG 评测系统

企业级 RAG (Retrieval-Augmented Generation) 评测框架，基于 Python Gradio 构建交互式前端。

## 功能特性

### 标注管理
- 标注数据的增删改查
- 支持多轮对话历史
- 可扩展的自定义字段
- 数据版本管理

### 评测功能
- 并发评测执行
- 支持单/双 RAG 接口对比
- 15+ 评测指标：
  - 检索指标：精确率、召回率、MRR、Hit Rate
  - 生成指标：事实一致性、相关性、完整性、流畅度
  - FAQ 指标：匹配准确率、答案一致性
  - 综合指标：风格匹配、多轮一致性、幻觉检测

### 结果展示
- 评测结果概览
- 单条详情查看
- 流式重跑展示
- 结果导出（JSON/CSV）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，设置必要的配置
```

必须配置的环境变量：
- `OPENAI_API_KEY`: OpenAI API 密钥（用于 LLM 评测）
- `RAG_SERVICE_URL`: RAG 服务地址

### 3. 运行应用

```bash
python -m src.main
```

或使用命令行参数：

```bash
python -m src.main --host 0.0.0.0 --port 7860 --debug
```

## 项目结构

```
rag_eval/
├── src/
│   ├── core/           # 配置、异常、日志
│   ├── models/         # 数据模型
│   ├── storage/        # 存储层
│   ├── annotation/     # 标注模块
│   ├── rag/            # RAG 接口适配
│   ├── evaluation/     # 评测引擎和指标
│   ├── ui/             # Gradio 前端
│   └── utils/          # 工具函数
├── tests/              # 测试用例
├── data/               # 数据存储
├── requirements.txt
├── .env.example
└── README.md
```

## 设计模式

项目严格遵循以下设计模式：

- **策略模式**: 评测指标的统一管理
- **工厂模式**: 指标实例化、存储创建
- **适配器模式**: RAG 接口适配
- **模板方法模式**: 评测流程标准化
- **单例模式**: 全局配置、数据库连接
- **迭代器模式**: 数据遍历

## 扩展评测指标

新增评测指标只需：

1. 在 `src/evaluation/metrics/` 创建新文件
2. 继承 `BaseMetric` 类
3. 实现 `calculate()` 方法
4. 在 `__init__.py` 中注册

```python
from .base import BaseMetric, MetricContext
from ...models.metric_result import MetricCategory, MetricResult

class MyCustomMetric(BaseMetric):
    name = "my_custom_metric"
    category = MetricCategory.COMPREHENSIVE
    description = "我的自定义指标"

    async def calculate(self, context: MetricContext) -> MetricResult:
        # 实现评测逻辑
        score = 0.8
        return self._create_result(score=score)
```

## 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行带覆盖率
pytest tests/ --cov=src --cov-report=html

# 运行单个测试
pytest tests/test_annotation.py -v
```

## API 文档

### 标注 API

```python
from src.annotation import get_annotation_handler

handler = await get_annotation_handler()

# 创建标注
annotation = Annotation(query="什么是 RAG?")
await handler.create(annotation)

# 列表查询
results = await handler.list(page=1, page_size=20)

# 更新
await handler.update(annotation_id, {"faq_matched": True})
```

### 评测 API

```python
from src.evaluation import create_runner
from src.rag import MockRAGAdapter

runner = await create_runner(max_concurrent=10)
runner.set_metrics(["retrieval_precision", "answer_relevance"])
runner.set_rag_adapter(MockRAGAdapter())

run = await runner.run(annotations)
```

## 安全注意事项

- 所有 API Key 从环境变量读取，禁止硬编码
- 敏感信息不记录日志
- 用户输入经过验证和清理
- 文件路径防止遍历攻击

## 性能优化

- 异步 I/O 操作
- 进程池处理 CPU 密集任务
- 大文件分块读写
- 数据库索引优化
- 内存缓存

## 许可证

MIT License