"""Tests for RAG adapters."""

import pytest
import pytest_asyncio

from src.rag.mock_adapter import MockRAGAdapter
from src.rag.response_parser import RAGResponseParser
from src.models.annotation import Annotation
from src.models.rag_response import RAGResponse


class TestMockRAGAdapter:
    """Tests for Mock RAG Adapter."""

    @pytest.fixture
    def adapter(self):
        """Create mock adapter."""
        return MockRAGAdapter(simulate_latency=False)

    @pytest.mark.asyncio
    async def test_query(self, adapter):
        """Test basic query."""
        response = await adapter.query("什么是机器学习？")

        assert response.success is True
        assert response.query == "什么是机器学习？"
        assert len(response.retrieval_results) > 0
        assert response.final_answer != ""

    @pytest.mark.asyncio
    async def test_query_from_annotation(self, adapter):
        """Test query from annotation."""
        annotation = Annotation(
            query="测试查询",
            conversation_history=["历史消息"],
        )

        response = await adapter.query_from_annotation(annotation)

        assert response.success is True
        assert response.query == "测试查询"

    @pytest.mark.asyncio
    async def test_health_check(self, adapter):
        """Test health check."""
        is_healthy = await adapter.health_check()

        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_faq_match_probability(self, adapter):
        """Test FAQ match probability (should match ~30% of the time)."""
        matches = 0
        for _ in range(100):
            response = await adapter.query("测试问题")
            if response.faq_match and response.faq_match.matched:
                matches += 1

        # Should be roughly 30% with some variance
        assert 10 < matches < 50


class TestRAGResponseParser:
    """Tests for RAG Response Parser."""

    def test_parse_simple_format(self):
        """Test parsing simple format."""
        data = {
            "answer": "这是答案",
            "contexts": ["文档1", "文档2"],
        }

        response = RAGResponseParser.parse(data, "测试查询")

        assert response.query == "测试查询"
        assert response.final_answer == "这是答案"
        assert len(response.retrieval_results) == 2

    def test_parse_langgraph_format(self):
        """Test parsing LangGraph format."""
        data = {
            "query_rewrite": {
                "rewritten": "改写后的查询",
                "type": "expansion",
            },
            "faq": {
                "matched": True,
                "id": "faq_1",
                "answer": "FAQ答案",
            },
            "retrieval": [
                {"id": "doc1", "content": "检索文档", "score": 0.9},
            ],
            "generation": {
                "content": "生成的答案",
            },
        }

        response = RAGResponseParser.parse(data, "原始查询")

        assert response.query_rewrite is not None
        assert response.faq_match.matched is True
        assert len(response.retrieval_results) == 1
        assert response.final_answer == "生成的答案"

    def test_parse_json_string(self):
        """Test parsing JSON string."""
        import json

        data = {"answer": "JSON答案"}
        json_str = json.dumps(data)

        response = RAGResponseParser.parse_json(json_str, "查询")

        assert response.final_answer == "JSON答案"

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON."""
        response = RAGResponseParser.parse_json("invalid json", "查询")

        assert response.success is False
        assert "JSON parse error" in response.error_message


class TestRAGResponse:
    """Tests for RAG Response model."""

    def test_response_creation(self):
        """Test creating RAG response."""
        response = RAGResponse(
            query="测试",
            final_answer="答案",
        )

        assert response.query == "测试"
        assert response.final_answer == "答案"

    def test_get_retrieved_contents(self):
        """Test getting retrieved contents."""
        from src.models.rag_response import RetrievalResult

        response = RAGResponse(
            query="测试",
            retrieval_results=[
                RetrievalResult(document_id="1", content="内容1", rank=1),
                RetrievalResult(document_id="2", content="内容2", rank=2),
            ],
        )

        contents = response.get_retrieved_contents()

        assert len(contents) == 2
        assert contents[0] == "内容1"

    def test_get_top_k(self):
        """Test getting top-k contents."""
        from src.models.rag_response import RetrievalResult

        response = RAGResponse(
            query="测试",
            retrieval_results=[
                RetrievalResult(document_id=str(i), content=f"内容{i}", rank=i)
                for i in range(10)
            ],
        )

        top_5 = response.get_top_k_contents(5)

        assert len(top_5) == 5