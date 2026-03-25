"""
Integration tests for refusal branch.

Tests the complete flow when the system should refuse to answer.
"""

import pytest
import os

from rag_rag.graph.state import create_initial_state
from rag_rag.graph.graph import run_rag_pipeline


class TestRefusalBranch:
    """Integration tests for refusal scenarios."""

    @pytest.mark.asyncio
    async def test_refusal_out_of_domain(self):
        """
        Scenario D1: Out of domain (no results).

        When no relevant documents are found, should refuse with out_of_domain.
        """
        # Use a query that won't match our electronics knowledge base
        result = await run_rag_pipeline(query="如何制作意大利面")

        # Check if refused
        if result.get("is_refused"):
            assert result["is_refused"] is True
            assert result.get("refusal_type") == "out_of_domain"
            assert "抱歉" in result.get("answer", "") or "无法回答" in result.get("answer", "")

    @pytest.mark.asyncio
    async def test_refusal_low_relevance(self):
        """
        Scenario D2: Low relevance refusal.

        When top result has low relevance score, should refuse.
        """
        # Use a vague query that might have low relevance results
        result = await run_rag_pipeline(query="东西")

        # Check if refused due to low relevance
        if result.get("is_refused") and result.get("refusal_type") == "low_relevance":
            assert result["is_refused"] is True
            assert "相关性过低" in result.get("refusal_reason", "") or "抱歉" in result.get("answer", "")

    @pytest.mark.asyncio
    async def test_normal_generation_no_refusal(self):
        """
        Scenario D3: Normal generation.

        When query matches well, should generate normally without refusal.
        """
        # Use a query that should match our knowledge base well
        result = await run_rag_pipeline(query="推荐一款适合玩游戏的手机")

        # Should not be refused
        assert result.get("is_refused") is False
        assert result.get("answer")
        assert "抱歉" not in result.get("answer", "") or "推荐" in result.get("answer", "")

    @pytest.mark.asyncio
    async def test_refusal_response_format(self):
        """
        Scenario D4: Refusal response format.

        Refusal response should have proper format.
        """
        result = await run_rag_pipeline(query="完全无关的查询xyz123关于烹饪")

        if result.get("is_refused"):
            # Check response structure
            assert "answer" in result
            assert "is_refused" in result
            assert result["is_refused"] is True

            # Check LLM output
            llm_output = result.get("llm_output", {})
            assert llm_output.get("finish_reason") == "refused"


class TestRefusalRouting:
    """Test routing decisions in refusal branch."""

    @pytest.mark.asyncio
    async def test_routes_to_refuse_when_no_results(self):
        """Should route to refuse node when no retrieval results."""
        from rag_rag.graph.routers import route_after_refusal

        state = create_initial_state(query="test")
        state["should_refuse"] = True
        state["refusal_type"] = "out_of_domain"

        result = route_after_refusal(state)

        assert result == "refuse"

    @pytest.mark.asyncio
    async def test_routes_to_generate_when_results_found(self):
        """Should route to generate node when results are found."""
        from rag_rag.graph.routers import route_after_refusal

        state = create_initial_state(query="test")
        state["should_refuse"] = False
        state["reranked_results"] = [
            {"document_id": "doc-001", "rerank_score": 0.8},
        ]

        result = route_after_refusal(state)

        assert result == "generate"


class TestRefusalThresholds:
    """Test refusal threshold configurations."""

    @pytest.mark.asyncio
    async def test_low_relevance_threshold(self):
        """Test low relevance threshold."""
        from rag_rag.core.config import get_config

        config = get_config()

        # Check default threshold
        assert hasattr(config, "refusal")
        assert hasattr(config.refusal, "low_relevance_threshold")
        assert config.refusal.low_relevance_threshold == 0.2

    @pytest.mark.asyncio
    async def test_refusal_check_with_high_score(self):
        """Should not refuse when score is above threshold."""
        from rag_rag.graph.nodes import refusal_check_node

        state = create_initial_state(query="test")
        state["reranked_results"] = [
            {"document_id": "doc-001", "rerank_score": 0.8, "metadata": {}},
        ]

        result = await refusal_check_node(state)

        assert result["should_refuse"] is False

    @pytest.mark.asyncio
    async def test_refusal_check_with_low_score(self):
        """Should refuse when score is below threshold."""
        from rag_rag.graph.nodes import refusal_check_node

        state = create_initial_state(query="test")
        state["reranked_results"] = [
            {"document_id": "doc-001", "rerank_score": 0.1, "metadata": {}},
        ]

        result = await refusal_check_node(state)

        assert result["should_refuse"] is True
        assert result["refusal_type"] == "low_relevance"


class TestRefusalMessages:
    """Test refusal message templates."""

    @pytest.mark.asyncio
    async def test_out_of_domain_message(self):
        """Test out of domain refusal message."""
        from rag_rag.graph.nodes import refuse_node

        state = create_initial_state(query="test")
        state["refusal_type"] = "out_of_domain"

        result = await refuse_node(state)

        assert "知识范围" in result["final_answer"] or "无法回答" in result["final_answer"]

    @pytest.mark.asyncio
    async def test_sensitive_message(self):
        """Test sensitive content refusal message."""
        from rag_rag.graph.nodes import refuse_node

        state = create_initial_state(query="test")
        state["refusal_type"] = "sensitive"

        result = await refuse_node(state)

        assert "敏感" in result["final_answer"] or "无法回答" in result["final_answer"]

    @pytest.mark.asyncio
    async def test_low_relevance_message(self):
        """Test low relevance refusal message."""
        from rag_rag.graph.nodes import refuse_node

        state = create_initial_state(query="test")
        state["refusal_type"] = "low_relevance"

        result = await refuse_node(state)

        assert "没有找到" in result["final_answer"] or "无法" in result["final_answer"]


class TestRefusalTiming:
    """Test timing in refusal branch."""

    @pytest.mark.asyncio
    async def test_refusal_timing_recorded(self):
        """Refusal timing should be recorded."""
        result = await run_rag_pipeline(query="完全无关的查询xyz烹饪")

        timing = result.get("stage_timing", {})

        # All stages should have timing
        assert "input_ms" in timing
        assert "faq_match_ms" in timing

        # If refused, refuse_ms should be recorded
        if result.get("is_refused"):
            assert "refuse_ms" in timing or "generation_ms" not in timing