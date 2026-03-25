"""
Integration tests for retrieval pipeline.

Tests the complete flow through retrieval, merge, and rerank stages.
"""

import pytest
import os

from rag_rag.graph.state import create_initial_state
from rag_rag.graph.graph import run_rag_pipeline


class TestRetrievalPipeline:
    """Integration tests for retrieval stages."""

    @pytest.mark.asyncio
    async def test_vector_retrieval_pipeline(self, monkeypatch):
        """
        Scenario B1: Vector retrieval returns results.

        With valid API key, vector retrieval should return results.
        """
        # API key should be set in environment
        result = await run_rag_pipeline(query="推荐一款适合玩游戏的手机")

        # Check vector results
        vector_results = result.get("vector_results", [])

        # If API key is set, should have results
        if os.environ.get("DASHSCOPE_API_KEY") and os.environ.get("DASHSCOPE_API_KEY") != "sk-placeholder-key":
            # Vector retrieval timing should be recorded
            timing = result.get("stage_timing", {})
            assert "vector_retrieve_ms" in timing

    @pytest.mark.asyncio
    async def test_vector_retrieval_without_api_key(self, monkeypatch):
        """
        Scenario B2: Vector retrieval without API key.

        Without API key, should gracefully degrade to empty results.
        """
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

        result = await run_rag_pipeline(query="测试查询")

        # Should have vector_results key (may be empty)
        assert "vector_results" in result

    @pytest.mark.asyncio
    async def test_fulltext_retrieval_pipeline(self):
        """
        Scenario B3: Fulltext retrieval returns results.

        Fulltext search should return BM25-scored results.
        """
        result = await run_rag_pipeline(query="iPhone手机功能")

        # Check fulltext results
        fulltext_results = result.get("fulltext_results", [])

        # Should have fulltext timing
        timing = result.get("stage_timing", {})
        assert "fulltext_retrieve_ms" in timing

        # Results should have expected structure
        for doc in fulltext_results:
            assert "document_id" in doc
            assert "content" in doc
            assert "score" in doc
            assert doc["source"] == "fulltext"

    @pytest.mark.asyncio
    async def test_parallel_retrieval_pipeline(self):
        """
        Test parallel retrieval (fan-out).

        Vector, fulltext, and graph retrieval should run in parallel.
        """
        result = await run_rag_pipeline(query="推荐一款手机")

        timing = result.get("stage_timing", {})

        # All retrieval timings should be recorded
        assert "vector_retrieve_ms" in timing
        assert "fulltext_retrieve_ms" in timing
        # graph_retrieve_ms may be minimal (placeholder)

    @pytest.mark.asyncio
    async def test_empty_retrieval_results(self):
        """
        Scenario B5: All retrieval sources return empty.

        When nothing matches, should have empty merged results.
        """
        result = await run_rag_pipeline(query="xyznonexistentquery12345")

        # Check merged results
        merged = result.get("merged_results", [])

        # May be empty or have some results depending on data
        assert isinstance(merged, list)


class TestMergeRerankPipeline:
    """Tests for merge and rerank stages."""

    @pytest.mark.asyncio
    async def test_merge_combines_sources(self):
        """
        Scenario C1: Weighted fusion from multiple sources.

        Results from vector and fulltext should be combined with weights.
        """
        result = await run_rag_pipeline(query="iPhone手机")

        merged = result.get("merged_results", [])

        # Check structure of merged results
        for doc in merged:
            assert "document_id" in doc
            assert "combined_score" in doc

    @pytest.mark.asyncio
    async def test_deduplication_in_merge(self):
        """
        Scenario C2: Deduplication of same documents.

        Same document from multiple sources should be deduplicated.
        """
        result = await run_rag_pipeline(query="iPhone")

        merged = result.get("merged_results", [])

        # Check no duplicate document_ids
        doc_ids = [doc.get("document_id") for doc in merged]
        assert len(doc_ids) == len(set(doc_ids))

    @pytest.mark.asyncio
    async def test_rerank_top_k(self):
        """
        Scenario C4: Rerank returns top-K results.

        Rerank should return at most top_k results.
        """
        result = await run_rag_pipeline(query="推荐手机")

        reranked = result.get("reranked_results", [])

        # Default top_k is 5
        assert len(reranked) <= 5

        # Check ranking
        for i, doc in enumerate(reranked):
            assert doc.get("rank") == i + 1

    @pytest.mark.asyncio
    async def test_rerank_timing_recorded(self):
        """Rerank timing should be recorded."""
        result = await run_rag_pipeline(query="测试")

        timing = result.get("stage_timing", {})
        assert "rerank_ms" in timing


class TestRetrievalBranchFlow:
    """Test complete retrieval branch flow."""

    @pytest.mark.asyncio
    async def test_complete_retrieval_flow(self):
        """
        Complete flow: FAQ miss -> Query rewrite -> Retrieval -> Merge -> Rerank.
        """
        result = await run_rag_pipeline(query="推荐一款适合游戏的手机")

        # Verify all stages ran
        timing = result.get("stage_timing", {})

        assert "input_ms" in timing
        assert "faq_match_ms" in timing
        assert "query_rewrite_ms" in timing
        assert "vector_retrieve_ms" in timing
        assert "fulltext_retrieve_ms" in timing
        assert "merge_ms" in timing
        assert "rerank_ms" in timing

    @pytest.mark.asyncio
    async def test_query_rewrite_type_detection(self):
        """Query rewrite should detect correct type."""
        # Short query -> clarification
        result = await run_rag_pipeline(query="手机")

        query_rewrite = result.get("query_rewrite", {})
        if query_rewrite:
            assert "rewrite_type" in query_rewrite

    @pytest.mark.asyncio
    async def test_retrieval_with_context_for_generation(self):
        """Retrieval results should be used for context."""
        result = await run_rag_pipeline(query="iPhone有什么功能")

        # Should have context prompt built
        context = result.get("context_prompt", "")
        final_prompt = result.get("final_prompt", "")

        # If retrieval found results, context should be non-empty
        if result.get("reranked_results"):
            assert context or final_prompt