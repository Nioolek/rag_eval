"""Tests for evaluation metrics."""

import pytest

from src.models.annotation import Annotation
from src.models.rag_response import RAGResponse, RetrievalResult
from src.evaluation.metrics.base import MetricContext
from src.evaluation.metrics.retrieval import (
    RetrievalPrecisionMetric,
    RetrievalRecallMetric,
    MRRMetric,
    HitRateMetric,
)
from src.evaluation.metrics.generation import (
    AnswerRelevanceMetric,
    RefusalAccuracyMetric,
)


class TestRetrievalMetrics:
    """Tests for retrieval metrics."""

    @pytest.fixture
    def context(self):
        """Create metric context for testing."""
        annotation = Annotation(
            query="什么是机器学习？",
            gt_documents=[
                "机器学习是人工智能的一个分支",
                "机器学习算法可以从数据中学习",
            ],
        )

        rag_response = RAGResponse(
            query="什么是机器学习？",
            retrieval_results=[
                RetrievalResult(
                    document_id="1",
                    content="机器学习是人工智能的一个分支",
                    score=0.95,
                    rank=1,
                ),
                RetrievalResult(
                    document_id="2",
                    content="深度学习是机器学习的子集",
                    score=0.85,
                    rank=2,
                ),
            ],
            final_answer="机器学习是一种使计算机能够从数据中学习的技术。",
        )

        return MetricContext(annotation=annotation, rag_response=rag_response)

    @pytest.mark.asyncio
    async def test_retrieval_precision(self, context):
        """Test retrieval precision metric."""
        metric = RetrievalPrecisionMetric()
        result = await metric.evaluate(context)

        assert result.score >= 0 and result.score <= 1
        assert result.metric_name == "retrieval_precision"
        assert "retrieved_count" in result.details

    @pytest.mark.asyncio
    async def test_retrieval_recall(self, context):
        """Test retrieval recall metric."""
        metric = RetrievalRecallMetric()
        result = await metric.evaluate(context)

        assert result.score >= 0 and result.score <= 1
        assert "gt_count" in result.details

    @pytest.mark.asyncio
    async def test_mrr(self, context):
        """Test MRR metric."""
        metric = MRRMetric()
        result = await metric.evaluate(context)

        assert result.score >= 0 and result.score <= 1
        assert result.score > 0  # Should find a match

    @pytest.mark.asyncio
    async def test_hit_rate(self, context):
        """Test hit rate metric."""
        metric = HitRateMetric(config={"k": 5})
        result = await metric.evaluate(context)

        assert result.score in [0, 1]
        assert "k" in result.details

    @pytest.mark.asyncio
    async def test_empty_retrieval(self):
        """Test metrics with empty retrieval."""
        annotation = Annotation(query="测试查询")
        rag_response = RAGResponse(query="测试查询")

        context = MetricContext(annotation=annotation, rag_response=rag_response)

        metric = RetrievalPrecisionMetric()
        result = await metric.evaluate(context)

        assert result.score == 0


class TestGenerationMetrics:
    """Tests for generation metrics."""

    @pytest.fixture
    def context(self):
        """Create metric context for generation tests."""
        annotation = Annotation(
            query="什么是RAG？",
            should_refuse=False,
        )

        rag_response = RAGResponse(
            query="什么是RAG？",
            final_answer="RAG是检索增强生成技术的缩写，结合了检索和生成模型。",
            is_refused=False,
        )

        return MetricContext(annotation=annotation, rag_response=rag_response)

    @pytest.mark.asyncio
    async def test_answer_relevance(self, context):
        """Test answer relevance metric."""
        metric = AnswerRelevanceMetric()
        result = await metric.evaluate(context)

        assert result.score >= 0 and result.score <= 1
        assert result.metric_name == "answer_relevance"

    @pytest.mark.asyncio
    async def test_refusal_accuracy_correct(self, context):
        """Test refusal accuracy when correct."""
        metric = RefusalAccuracyMetric()
        result = await metric.evaluate(context)

        assert result.score == 1.0  # Correct: not refused, should not refuse
        assert result.details["outcome"] == "true_negative"

    @pytest.mark.asyncio
    async def test_refusal_accuracy_incorrect(self):
        """Test refusal accuracy when incorrect."""
        annotation = Annotation(query="测试", should_refuse=True)
        rag_response = RAGResponse(query="测试", is_refused=False)

        context = MetricContext(annotation=annotation, rag_response=rag_response)

        metric = RefusalAccuracyMetric()
        result = await metric.evaluate(context)

        assert result.score == 0  # Incorrect: should refuse but didn't
        assert result.details["outcome"] == "false_negative"


class TestMetricRegistry:
    """Tests for metric registry."""

    def test_registry_initialization(self):
        """Test registry initialization."""
        from src.evaluation.metrics.metric_registry import get_registry

        registry = get_registry()

        assert len(registry) > 0

    def test_list_metrics(self):
        """Test listing metrics."""
        from src.evaluation.metrics.metric_registry import get_registry

        registry = get_registry()
        metrics = registry.list_all()

        assert "retrieval_precision" in metrics
        assert "answer_relevance" in metrics

    def test_get_metric_info(self):
        """Test getting metric info."""
        from src.evaluation.metrics.metric_registry import get_registry

        registry = get_registry()
        info = registry.get_info("retrieval_precision")

        assert info is not None
        assert info["name"] == "retrieval_precision"
        assert info["category"] == "retrieval"


class TestMetricFactory:
    """Tests for metric factory."""

    def test_create_metric(self):
        """Test creating a metric."""
        from src.evaluation.metrics.metric_factory import MetricFactory

        metric = MetricFactory.create("retrieval_precision")

        assert metric.name == "retrieval_precision"

    def test_create_all_metrics(self):
        """Test creating all metrics."""
        from src.evaluation.metrics.metric_factory import MetricFactory

        metrics = MetricFactory.create_all()

        assert len(metrics) > 0

    def test_create_unknown_metric(self):
        """Test creating unknown metric."""
        from src.evaluation.metrics.metric_factory import MetricFactory
        from src.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            MetricFactory.create("unknown_metric")