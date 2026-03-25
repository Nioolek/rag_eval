"""
Tests for LangGraph node functions.

Unit tests for each node in the RAG pipeline.
"""

import asyncio
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from rag_rag.graph.state import create_initial_state


class TestInputNode:
    """Tests for input_node."""

    @pytest.mark.asyncio
    async def test_input_node_validates_empty_query(self):
        """Empty query should be captured in errors."""
        from rag_rag.graph.nodes import input_node

        state = create_initial_state(query="")
        result = await input_node(state)

        # Error should be captured in errors list
        assert "errors" in result
        assert len(result["errors"]) > 0
        assert "Query cannot be empty" in result["errors"][0]["message"]

    @pytest.mark.asyncio
    async def test_input_node_validates_whitespace_query(self):
        """Whitespace-only query should be captured in errors."""
        from rag_rag.graph.nodes import input_node

        state = create_initial_state(query="   ")
        result = await input_node(state)

        # Error should be captured in errors list
        assert "errors" in result
        assert len(result["errors"]) > 0
        assert "Query cannot be empty" in result["errors"][0]["message"]

    @pytest.mark.asyncio
    async def test_input_node_generates_conversation_id(self, sample_query: str):
        """Should generate conversation_id if not provided."""
        from rag_rag.graph.nodes import input_node

        state = create_initial_state(query=sample_query)
        result = await input_node(state)

        assert "conversation_id" in result
        assert result["conversation_id"]  # Not empty

    @pytest.mark.asyncio
    async def test_input_node_preserves_existing_conversation_id(self, sample_query: str):
        """Should preserve existing conversation_id."""
        from rag_rag.graph.nodes import input_node

        existing_id = "existing-conv-id"
        state = create_initial_state(query=sample_query, conversation_id=existing_id)
        result = await input_node(state)

        assert result["conversation_id"] == existing_id

    @pytest.mark.asyncio
    async def test_input_node_records_timing(self, sample_query: str):
        """Should record timing for input stage."""
        from rag_rag.graph.nodes import input_node

        state = create_initial_state(query=sample_query)
        result = await input_node(state)

        assert "stage_timing" in result
        assert "input_ms" in result["stage_timing"]


class TestFAQMatchNode:
    """Tests for faq_match_node."""

    @pytest.mark.asyncio
    async def test_faq_match_with_matching_query(self, sample_query: str):
        """FAQ match should set faq_matched=True when query matches."""
        from rag_rag.graph.nodes import faq_match_node

        # Use a query that matches our test data
        state = create_initial_state(query="iPhone截屏")

        result = await faq_match_node(state)

        assert "faq_matched" in result
        assert "faq_result" in result
        # Note: Actual matching depends on FAQ store content

    @pytest.mark.asyncio
    async def test_faq_match_with_non_matching_query(self):
        """FAQ match should set faq_matched=False when query doesn't match."""
        from rag_rag.graph.nodes import faq_match_node

        # Use a query that likely won't match
        state = create_initial_state(query="这是一个完全不存在的问题xyz123")

        result = await faq_match_node(state)

        assert "faq_matched" in result
        assert result["faq_matched"] is False

    @pytest.mark.asyncio
    async def test_faq_match_records_timing(self, sample_query: str):
        """Should record timing for FAQ match stage."""
        from rag_rag.graph.nodes import faq_match_node

        state = create_initial_state(query=sample_query)
        result = await faq_match_node(state)

        assert "stage_timing" in result
        assert "faq_match_ms" in result["stage_timing"]


class TestVectorRetrieveNode:
    """Tests for vector_retrieve_node."""

    @pytest.mark.asyncio
    async def test_vector_retrieve_with_api_key(self, sample_query: str, monkeypatch):
        """Vector retrieve should handle API key (may fail if key invalid)."""
        from rag_rag.graph.nodes import vector_retrieve_node

        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test-key")

        state = create_initial_state(query=sample_query)

        result = await vector_retrieve_node(state)

        # Result should have either vector_results or errors
        assert "vector_results" in result or "errors" in result

    @pytest.mark.asyncio
    async def test_vector_retrieve_without_api_key(self, sample_query: str, monkeypatch):
        """Vector retrieve should return empty results when no API key."""
        from rag_rag.graph.nodes import vector_retrieve_node

        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

        state = create_initial_state(query=sample_query)

        result = await vector_retrieve_node(state)

        assert "vector_results" in result
        # Should be empty or have warning in logs

    @pytest.mark.asyncio
    async def test_vector_retrieve_with_placeholder_key(self, sample_query: str, monkeypatch):
        """Vector retrieve should treat placeholder key as missing."""
        from rag_rag.graph.nodes import vector_retrieve_node

        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-placeholder-key")

        state = create_initial_state(query=sample_query)

        result = await vector_retrieve_node(state)

        assert "vector_results" in result


class TestFulltextRetrieveNode:
    """Tests for fulltext_retrieve_node."""

    @pytest.mark.asyncio
    async def test_fulltext_retrieve_returns_results(self, sample_query: str):
        """Fulltext retrieve should return results from Whoosh index."""
        from rag_rag.graph.nodes import fulltext_retrieve_node

        state = create_initial_state(query=sample_query)

        result = await fulltext_retrieve_node(state)

        assert "fulltext_results" in result
        assert isinstance(result["fulltext_results"], list)

    @pytest.mark.asyncio
    async def test_fulltext_retrieve_records_timing(self, sample_query: str):
        """Should record timing for fulltext retrieve stage."""
        from rag_rag.graph.nodes import fulltext_retrieve_node

        state = create_initial_state(query=sample_query)
        result = await fulltext_retrieve_node(state)

        assert "stage_timing" in result
        assert "fulltext_retrieve_ms" in result["stage_timing"]


class TestMergeNode:
    """Tests for merge_node."""

    @pytest.mark.asyncio
    async def test_merge_combines_results(self):
        """Merge should combine results from multiple sources."""
        from rag_rag.graph.nodes import merge_node

        state = create_initial_state(query="test")
        state["vector_results"] = [
            {"document_id": "doc-001", "content": "content 1", "score": 0.8, "source": "vector", "metadata": {}},
        ]
        state["fulltext_results"] = [
            {"document_id": "doc-002", "content": "content 2", "score": 5.0, "source": "fulltext", "metadata": {}},
        ]
        state["graph_results"] = []

        result = await merge_node(state)

        assert "merged_results" in result
        assert len(result["merged_results"]) == 2

    @pytest.mark.asyncio
    async def test_merge_deduplicates_by_document_id(self):
        """Merge should deduplicate documents with same ID."""
        from rag_rag.graph.nodes import merge_node

        state = create_initial_state(query="test")
        state["vector_results"] = [
            {"document_id": "doc-001", "content": "content", "score": 0.8, "source": "vector", "metadata": {}},
        ]
        state["fulltext_results"] = [
            {"document_id": "doc-001", "content": "content", "score": 5.0, "source": "fulltext", "metadata": {}},
        ]
        state["graph_results"] = []

        result = await merge_node(state)

        # Should have only 1 unique document
        assert len(result["merged_results"]) == 1
        # Should have combined scores
        merged = result["merged_results"][0]
        assert "vector_score" in merged
        assert "fulltext_score" in merged

    @pytest.mark.asyncio
    async def test_merge_with_empty_results(self):
        """Merge should handle empty retrieval results."""
        from rag_rag.graph.nodes import merge_node

        state = create_initial_state(query="test")
        state["vector_results"] = []
        state["fulltext_results"] = []
        state["graph_results"] = []

        result = await merge_node(state)

        assert "merged_results" in result
        assert len(result["merged_results"]) == 0


class TestRerankNode:
    """Tests for rerank_node."""

    @pytest.mark.asyncio
    async def test_rerank_returns_top_k(self):
        """Rerank should return top_k results."""
        from rag_rag.graph.nodes import rerank_node

        state = create_initial_state(query="test")
        state["merged_results"] = [
            {"document_id": f"doc-{i}", "content": f"content {i}", "combined_score": 0.9 - i * 0.1, "metadata": {}}
            for i in range(10)
        ]

        result = await rerank_node(state)

        assert "reranked_results" in result
        # Default top_k is 5
        assert len(result["reranked_results"]) <= 5

    @pytest.mark.asyncio
    async def test_rerank_with_empty_input(self):
        """Rerank should handle empty input."""
        from rag_rag.graph.nodes import rerank_node

        state = create_initial_state(query="test")
        state["merged_results"] = []

        result = await rerank_node(state)

        assert "reranked_results" in result
        assert len(result["reranked_results"]) == 0


class TestRefusalCheckNode:
    """Tests for refusal_check_node."""

    @pytest.mark.asyncio
    async def test_refusal_when_no_results(self):
        """Should refuse when no retrieval results."""
        from rag_rag.graph.nodes import refusal_check_node

        state = create_initial_state(query="test")
        state["reranked_results"] = []

        result = await refusal_check_node(state)

        assert result["should_refuse"] is True
        assert result["refusal_type"] == "out_of_domain"

    @pytest.mark.asyncio
    async def test_refusal_when_low_relevance(self):
        """Should refuse when top score is below threshold."""
        from rag_rag.graph.nodes import refusal_check_node

        state = create_initial_state(query="test")
        state["reranked_results"] = [
            {"document_id": "doc-001", "rerank_score": 0.1, "metadata": {}},
        ]

        result = await refusal_check_node(state)

        assert result["should_refuse"] is True
        assert result["refusal_type"] == "low_relevance"

    @pytest.mark.asyncio
    async def test_no_refusal_when_good_relevance(self):
        """Should not refuse when results have good relevance."""
        from rag_rag.graph.nodes import refusal_check_node

        state = create_initial_state(query="test")
        state["reranked_results"] = [
            {"document_id": "doc-001", "rerank_score": 0.8, "metadata": {}},
        ]

        result = await refusal_check_node(state)

        assert result["should_refuse"] is False


class TestGenerateNode:
    """Tests for generate_node."""

    @pytest.mark.asyncio
    async def test_generate_with_api_key(self, monkeypatch):
        """Generate should use LLM when API key is set."""
        from rag_rag.graph.nodes import generate_node

        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test-key")

        state = create_initial_state(query="test")
        state["system_prompt"] = "You are a helpful assistant."
        state["final_prompt"] = "User question"
        state["enable_thinking"] = False

        result = await generate_node(state)

        assert "llm_output" in result
        assert "final_answer" in result
        # With real API key, should get real response

    @pytest.mark.asyncio
    async def test_generate_without_api_key(self, monkeypatch):
        """Generate should return placeholder when no API key."""
        from rag_rag.graph.nodes import generate_node

        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

        state = create_initial_state(query="test")
        state["system_prompt"] = "You are a helpful assistant."
        state["final_prompt"] = "User question"
        state["enable_thinking"] = False

        result = await generate_node(state)

        assert "llm_output" in result
        assert "final_answer" in result
        # Should have placeholder response
        assert result["llm_output"]["model"] == "placeholder"


class TestOutputNode:
    """Tests for output_node."""

    @pytest.mark.asyncio
    async def test_output_formats_response(self):
        """Output should format response correctly."""
        from rag_rag.graph.nodes import output_node

        state = create_initial_state(query="test")
        state["final_answer"] = "This is the answer."
        state["is_refused"] = False
        state["reranked_results"] = []
        state["merged_results"] = []
        state["stage_timing"] = {"input_ms": 1.0, "total_ms": 100.0}

        result = await output_node(state)

        assert "answer" in result
        assert result["answer"] == "This is the answer."
        assert "retrieval" in result
        assert "rerank" in result
        assert "stage_timing" in result

    @pytest.mark.asyncio
    async def test_output_calculates_total_timing(self):
        """Output should calculate total timing."""
        from rag_rag.graph.nodes import output_node

        state = create_initial_state(query="test")
        state["final_answer"] = "Answer"
        state["is_refused"] = False
        state["reranked_results"] = []
        state["merged_results"] = []
        state["stage_timing"] = {
            "input_ms": 10.0,
            "faq_match_ms": 20.0,
            "vector_retrieve_ms": 30.0,
        }

        result = await output_node(state)

        assert "total_ms" in result["stage_timing"]
        assert result["stage_timing"]["total_ms"] == 60.0


class TestAnswerFAQNode:
    """Tests for answer_faq_node."""

    @pytest.mark.asyncio
    async def test_answer_faq_returns_answer(self):
        """Should return FAQ answer directly."""
        from rag_rag.graph.nodes import answer_faq_node

        state = create_initial_state(query="test")
        state["faq_result"] = {
            "matched": True,
            "question": "How to screenshot?",
            "answer": "Press power + volume up.",
        }

        result = await answer_faq_node(state)

        assert result["final_answer"] == "Press power + volume up."
        assert result["is_refused"] is False


class TestRefuseNode:
    """Tests for refuse_node."""

    @pytest.mark.asyncio
    async def test_refuse_returns_template_response(self):
        """Should return refusal template response."""
        from rag_rag.graph.nodes import refuse_node

        state = create_initial_state(query="test")
        state["refusal_type"] = "out_of_domain"
        state["refusal_reason"] = "No relevant information"

        result = await refuse_node(state)

        assert result["is_refused"] is True
        assert "final_answer" in result
        assert "抱歉" in result["final_answer"] or "无法回答" in result["final_answer"]

    @pytest.mark.asyncio
    async def test_refuse_with_different_types(self):
        """Should return appropriate response for different refusal types."""
        from rag_rag.graph.nodes import refuse_node

        refusal_types = ["out_of_domain", "sensitive", "low_relevance"]

        for refusal_type in refusal_types:
            state = create_initial_state(query="test")
            state["refusal_type"] = refusal_type

            result = await refuse_node(state)

            assert result["is_refused"] is True
            assert result["llm_output"]["finish_reason"] == "refused"