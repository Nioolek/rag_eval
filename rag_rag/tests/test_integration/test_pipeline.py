"""
Integration tests for RAG Pipeline.
"""

import pytest
import asyncio

from rag_rag.graph.state import RAGState, create_initial_state
from rag_rag.graph.graph import build_rag_graph, compile_rag_graph, run_rag_pipeline, stream_rag_pipeline


class TestRAGPipeline:
    """Tests for RAG Pipeline."""

    def test_create_initial_state(self):
        """Test initial state creation."""
        state = create_initial_state(
            query="如何申请年假？",
            conversation_id="test-conv-001",
            agent_id="test-agent",
            enable_thinking=False,
        )

        assert state["query"] == "如何申请年假？"
        assert state["conversation_id"] == "test-conv-001"
        assert state["agent_id"] == "test-agent"
        assert state["enable_thinking"] is False
        assert state["faq_matched"] is False
        assert state["should_refuse"] is False
        assert isinstance(state["errors"], list)
        assert isinstance(state["stage_timing"], dict)

    def test_build_graph(self):
        """Test graph building."""
        workflow = build_rag_graph()
        assert workflow is not None

    def test_compile_graph(self):
        """Test graph compilation."""
        compiled = compile_rag_graph()
        assert compiled is not None

    @pytest.mark.asyncio
    async def test_run_pipeline_simple(self):
        """Test running the pipeline with a simple query."""
        result = await run_rag_pipeline(
            query="如何申请年假？",
            enable_thinking=False,
        )

        assert result is not None
        assert "query" in result
        assert "answer" in result
        assert "stage_timing" in result
        assert result["query"] == "如何申请年假？"

    @pytest.mark.asyncio
    async def test_run_pipeline_with_thinking(self):
        """Test running the pipeline with thinking mode."""
        result = await run_rag_pipeline(
            query="公司有哪些福利？",
            enable_thinking=True,
        )

        assert result is not None
        assert result.get("enable_thinking") is True or "thinking" in result

    @pytest.mark.asyncio
    async def test_stream_pipeline(self):
        """Test streaming the pipeline."""
        events = []
        async for event in stream_rag_pipeline(query="测试问题"):
            events.append(event)

        assert len(events) > 0


class TestRAGState:
    """Tests for RAG State."""

    def test_stage_timing_structure(self):
        """Test stage timing structure."""
        state = create_initial_state(query="test")
        timing = state["stage_timing"]

        required_keys = [
            "input_ms",
            "faq_match_ms",
            "query_rewrite_ms",
            "vector_retrieve_ms",
            "fulltext_retrieve_ms",
            "graph_retrieve_ms",
            "merge_ms",
            "rerank_ms",
            "build_prompt_ms",
            "refusal_check_ms",
            "generation_ms",
            "total_ms",
        ]

        for key in required_keys:
            assert key in timing, f"Missing timing key: {key}"
            assert timing[key] == 0.0, f"Initial timing should be 0: {key}"


class TestPromptTemplates:
    """Tests for prompt templates."""

    def test_default_template_exists(self):
        """Test that default template exists."""
        from rag_rag.prompts.template_manager import get_template_manager

        manager = get_template_manager()
        template = manager.get_template("default")

        assert template is not None
        assert "system" in template
        assert "user" in template

    def test_render_template(self):
        """Test rendering a template."""
        from rag_rag.prompts.template_manager import get_template_manager

        manager = get_template_manager()
        system, user = manager.render(
            template_name="default",
            context="这是测试上下文",
            query="这是测试问题",
        )

        assert "测试上下文" in user
        assert "测试问题" in user
        assert len(system) > 0

    def test_refusal_template(self):
        """Test refusal template."""
        from rag_rag.prompts.template_manager import get_template_manager

        manager = get_template_manager()
        refusal = manager.render_refusal("out_of_domain")

        assert "抱歉" in refusal or "无法" in refusal