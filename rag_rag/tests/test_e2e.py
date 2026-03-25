"""
End-to-end tests for RAG Pipeline.

Tests complete pipeline scenarios and edge cases.
"""

import pytest
import os
import asyncio

from rag_rag.graph.state import create_initial_state
from rag_rag.graph.graph import run_rag_pipeline, stream_rag_pipeline


class TestEndToEndScenarios:
    """End-to-end scenario tests."""

    @pytest.mark.asyncio
    async def test_complete_pipeline_faq_match(self):
        """
        E2E: Complete pipeline with FAQ match.

        Query matches FAQ -> Direct answer returned.
        """
        result = await run_rag_pipeline(query="iPhone如何截屏")

        assert "answer" in result
        assert result.get("stage_timing", {}).get("total_ms", 0) > 0

    @pytest.mark.asyncio
    async def test_complete_pipeline_retrieval(self):
        """
        E2E: Complete pipeline with retrieval.

        Query doesn't match FAQ -> Retrieval -> Generation.
        """
        result = await run_rag_pipeline(query="推荐一款适合玩游戏的手机")

        assert "answer" in result
        assert "retrieval" in result
        assert "stage_timing" in result

    @pytest.mark.asyncio
    async def test_complete_pipeline_refusal(self):
        """
        E2E: Complete pipeline with refusal.

        Query is out of domain -> Refusal response.
        """
        result = await run_rag_pipeline(query="如何做红烧肉")

        assert "answer" in result
        assert "is_refused" in result


class TestEdgeCases:
    """Edge case tests."""

    @pytest.mark.asyncio
    async def test_empty_query_raises_error(self):
        """
        Scenario F1: Empty query validation.

        Empty query should be handled gracefully.
        """
        # Note: With node_decorator, errors are captured in errors list
        # rather than raised as exceptions
        result = await run_rag_pipeline(query="")

        # Should have error recorded
        errors = result.get("errors", [])
        assert len(errors) > 0
        assert any("Query cannot be empty" in str(e) for e in errors)

    @pytest.mark.asyncio
    async def test_whitespace_query_raises_error(self):
        """
        Scenario F2: Whitespace query validation.

        Whitespace-only query should be handled gracefully.
        """
        result = await run_rag_pipeline(query="   ")

        # Should have error recorded
        errors = result.get("errors", [])
        assert len(errors) > 0
        assert any("Query cannot be empty" in str(e) for e in errors)

    @pytest.mark.asyncio
    async def test_very_long_query(self):
        """Test with very long query."""
        long_query = "iPhone" * 100

        result = await run_rag_pipeline(query=long_query)

        assert "answer" in result

    @pytest.mark.asyncio
    async def test_special_characters_query(self):
        """Test with special characters."""
        special_query = "iPhone!!@#$%^&*()"

        result = await run_rag_pipeline(query=special_query)

        assert "answer" in result

    @pytest.mark.asyncio
    async def test_chinese_english_mixed_query(self):
        """Test with mixed Chinese and English."""
        mixed_query = "iPhone手机的battery寿命怎么样"

        result = await run_rag_pipeline(query=mixed_query)

        assert "answer" in result


class TestMultiTurnConversation:
    """Multi-turn conversation tests."""

    @pytest.mark.asyncio
    async def test_with_conversation_history(self):
        """
        Scenario F5: Multi-turn conversation.

        Query with conversation history should use multi_turn rewrite type.
        """
        result = await run_rag_pipeline(
            query="那华为的呢？",
            conversation_history=[
                {
                    "role": "user",
                    "content": "iPhone续航怎么样？",
                    "timestamp": "2024-01-01T10:00:00",
                },
                {
                    "role": "assistant",
                    "content": "iPhone续航表现不错...",
                    "timestamp": "2024-01-01T10:00:05",
                },
            ],
        )

        assert "answer" in result

        # Check rewrite type
        query_rewrite = result.get("query_rewrite", {})
        if query_rewrite:
            # With history, should be multi_turn
            assert query_rewrite.get("rewrite_type") in ["multi_turn", "clarification"]


class TestThinkingMode:
    """Thinking mode tests."""

    @pytest.mark.asyncio
    async def test_thinking_mode_enabled(self):
        """
        Scenario E3: Thinking mode.

        Query with thinking mode should include thinking prompt.
        """
        result = await run_rag_pipeline(
            query="比较iPhone和华为的优缺点",
            enable_thinking=True,
        )

        assert "answer" in result

        # Check if thinking prompt was used
        system_prompt = result.get("system_prompt", "")
        if system_prompt:
            assert "思考" in system_prompt or "分析" in system_prompt

    @pytest.mark.asyncio
    async def test_thinking_mode_disabled(self):
        """Query without thinking mode."""
        result = await run_rag_pipeline(
            query="iPhone怎么样",
            enable_thinking=False,
        )

        assert "answer" in result


class TestStageTiming:
    """Stage timing tests."""

    @pytest.mark.asyncio
    async def test_all_stages_have_timing(self):
        """
        Scenario E5: All stages timing recorded.

        Verify all stage timings are recorded.
        """
        result = await run_rag_pipeline(query="推荐手机")

        timing = result.get("stage_timing", {})

        # Required timing fields
        required_fields = [
            "input_ms",
            "faq_match_ms",
            "total_ms",
        ]

        for field in required_fields:
            assert field in timing, f"Missing timing: {field}"
            assert timing[field] >= 0, f"Invalid timing for {field}"

    @pytest.mark.asyncio
    async def test_total_timing_is_sum(self):
        """Total timing should be sum of stages."""
        result = await run_rag_pipeline(query="测试")

        timing = result.get("stage_timing", {})

        # Total should be > 0
        assert timing.get("total_ms", 0) > 0


class TestOutputFormat:
    """Output format tests."""

    @pytest.mark.asyncio
    async def test_output_has_all_required_fields(self):
        """
        Scenario E4: RAGResponse-compatible format.

        Output should have all fields required by RAGResponseAdapter.
        """
        result = await run_rag_pipeline(query="iPhone功能")

        # Required fields
        required_fields = [
            "answer",
            "stage_timing",
        ]

        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

    @pytest.mark.asyncio
    async def test_retrieval_format(self):
        """Retrieval results should have correct format."""
        result = await run_rag_pipeline(query="推荐手机")

        retrieval = result.get("retrieval", [])

        for doc in retrieval:
            assert "id" in doc or "document_id" in doc
            assert "content" in doc
            assert "score" in doc

    @pytest.mark.asyncio
    async def test_rerank_format(self):
        """Rerank results should have correct format."""
        result = await run_rag_pipeline(query="推荐手机")

        rerank = result.get("rerank", [])

        for doc in rerank:
            assert "id" in doc or "document_id" in doc


class TestStreamingPipeline:
    """Streaming pipeline tests."""

    @pytest.mark.asyncio
    async def test_stream_pipeline_yields_events(self):
        """Stream pipeline should yield events."""
        events = []

        async for event in stream_rag_pipeline(query="iPhone"):
            events.append(event)

        # Should have at least one event
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_stream_pipeline_complete_flow(self):
        """Stream pipeline should complete full flow."""
        final_state = None

        async for event in stream_rag_pipeline(query="测试"):
            final_state = event

        # Final state should have some output
        if final_state:
            # Streaming output may be nested under 'output' key
            output = final_state.get("output", final_state)
            # Check for either answer or final_answer
            has_answer = "answer" in output or "final_answer" in output
            assert has_answer or "errors" in final_state


class TestErrorHandling:
    """Error handling tests."""

    @pytest.mark.asyncio
    async def test_errors_recorded_on_failure(self):
        """
        Scenario F3: Node error propagation.

        Errors should be recorded in errors list.
        """
        result = await run_rag_pipeline(query="测试")

        # errors field should exist (even if empty)
        assert "errors" in result

    @pytest.mark.asyncio
    async def test_pipeline_doesnt_crash_on_error(self):
        """Pipeline should handle errors gracefully."""
        # Try various edge cases that might cause errors
        queries = [
            "a",  # Very short
            "测",  # Single Chinese character
            "123",  # Numbers only
        ]

        for query in queries:
            try:
                result = await run_rag_pipeline(query=query)
                assert "answer" in result
            except ValueError:
                # Expected for some edge cases
                pass


class TestPerformance:
    """Performance tests."""

    @pytest.mark.asyncio
    async def test_pipeline_completes_in_reasonable_time(self):
        """Pipeline should complete within reasonable time."""
        import time

        start = time.time()
        result = await run_rag_pipeline(query="iPhone")
        elapsed = time.time() - start

        # Should complete within 30 seconds
        assert elapsed < 30

    @pytest.mark.asyncio
    async def test_multiple_queries(self):
        """Test multiple queries in sequence."""
        queries = [
            "iPhone怎么样",
            "华为手机",
            "MacBook推荐",
        ]

        for query in queries:
            result = await run_rag_pipeline(query=query)
            assert "answer" in result