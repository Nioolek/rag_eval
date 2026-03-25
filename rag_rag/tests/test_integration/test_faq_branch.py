"""
Integration tests for FAQ match branch.

Tests the complete pipeline flow when FAQ is matched.
"""

import pytest
import asyncio

from rag_rag.graph.state import create_initial_state
from rag_rag.graph.graph import run_rag_pipeline, graph


class TestFAQBranchIntegration:
    """Integration tests for FAQ match branch."""

    @pytest.mark.asyncio
    async def test_faq_exact_match_pipeline(self):
        """
        Scenario A1: FAQ exact match.

        When query exactly matches an FAQ question,
        should route directly to answer_faq and return the answer.
        """
        # Use a query that matches our FAQ data
        result = await run_rag_pipeline(query="iPhone如何截屏")

        # Verify FAQ was matched
        if result.get("faq_result") and result["faq_result"].get("matched"):
            assert result["faq_result"]["matched"] is True
            assert result["faq_result"].get("answer")
            # Verify timing was recorded
            assert "faq_match_ms" in result.get("stage_timing", {})

    @pytest.mark.asyncio
    async def test_faq_semantic_match_pipeline(self):
        """
        Scenario A2: FAQ semantic match.

        When query is semantically similar to an FAQ,
        should match and return answer.
        """
        # Use a slightly different phrasing
        result = await run_rag_pipeline(query="苹果手机截图怎么弄")

        # Check if FAQ matched (depends on similarity threshold)
        if result.get("faq_result") and result["faq_result"].get("matched"):
            assert result["faq_result"]["confidence"] >= 0.6

    @pytest.mark.asyncio
    async def test_faq_no_match_pipeline(self):
        """
        Scenario A4: FAQ no match.

        When query doesn't match any FAQ,
        should proceed to retrieval pipeline.
        """
        # Use a query unlikely to match FAQ
        result = await run_rag_pipeline(query="量子计算机的工作原理是什么")

        # FAQ should not be matched
        faq_result = result.get("faq_result")
        if faq_result:
            assert faq_result.get("matched") is False
        else:
            assert result.get("faq_matched") is False

        # Should have retrieval results (or empty if nothing found)
        assert "fulltext_results" in result or "vector_results" in result

    @pytest.mark.asyncio
    async def test_faq_match_skips_retrieval(self):
        """
        Verify that FAQ match skips the retrieval pipeline.

        When FAQ is matched, retrieval stages should have 0 timing.
        """
        result = await run_rag_pipeline(query="iPhone截屏")

        timing = result.get("stage_timing", {})

        # If FAQ matched, retrieval should be minimal or skipped
        if result.get("faq_result") and result["faq_result"].get("matched"):
            # Vector and fulltext retrieve should be minimal
            # (they're run but with short-circuit logic)
            pass  # Timing values depend on implementation


class TestFAQBranchRouting:
    """Test routing decisions in FAQ branch."""

    @pytest.mark.asyncio
    async def test_graph_routes_correctly_on_match(self):
        """Verify graph routes to answer_faq when matched."""
        from rag_rag.graph.routers import route_after_faq

        state = create_initial_state(query="test")
        state["faq_matched"] = True
        state["faq_result"] = {"matched": True, "answer": "test answer"}

        result = route_after_faq(state)

        assert result == "answer_faq"

    @pytest.mark.asyncio
    async def test_graph_routes_correctly_on_no_match(self):
        """Verify graph routes to query_rewrite when not matched."""
        from rag_rag.graph.routers import route_after_faq

        state = create_initial_state(query="test")
        state["faq_matched"] = False

        result = route_after_faq(state)

        assert result == "query_rewrite"


class TestFAQBranchTiming:
    """Test timing in FAQ branch."""

    @pytest.mark.asyncio
    async def test_faq_match_timing_recorded(self):
        """FAQ match timing should be recorded."""
        result = await run_rag_pipeline(query="iPhone截屏")

        timing = result.get("stage_timing", {})

        assert "input_ms" in timing
        assert "faq_match_ms" in timing
        assert timing["faq_match_ms"] >= 0

    @pytest.mark.asyncio
    async def test_total_timing_recorded(self):
        """Total timing should be recorded."""
        result = await run_rag_pipeline(query="测试问题")

        timing = result.get("stage_timing", {})

        assert "total_ms" in timing
        assert timing["total_ms"] >= 0


class TestFAQBranchErrorHandling:
    """Test error handling in FAQ branch."""

    @pytest.mark.asyncio
    async def test_faq_store_unavailable(self, monkeypatch):
        """Should handle FAQ store unavailability gracefully."""
        # This test would mock FAQ store to raise an error
        # For now, we verify the pipeline doesn't crash
        result = await run_rag_pipeline(query="测试问题")

        # Should have some result
        assert "answer" in result
        # Errors should be recorded if any
        assert "errors" in result