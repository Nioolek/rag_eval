"""
RAG Pipeline Graph Construction.

Builds and compiles the LangGraph StateGraph.
"""

from typing import Optional

from langgraph.graph import StateGraph, END

from rag_rag.graph.state import RAGState, create_initial_state
from rag_rag.graph.routers import route_after_faq, route_after_refusal
from rag_rag.graph.nodes import (
    input_node,
    faq_match_node,
    query_rewrite_node,
    vector_retrieve_node,
    fulltext_retrieve_node,
    graph_retrieve_node,
    merge_node,
    rerank_node,
    build_prompt_node,
    refusal_check_node,
    generate_node,
    output_node,
    answer_faq_node,
    refuse_node,
)
from rag_rag.core.logging import get_logger

logger = get_logger("rag_rag.graph")


def build_rag_graph() -> StateGraph:
    """
    Build the RAG Pipeline StateGraph.

    Graph structure:
    ```
    input → faq_match ─┬→ answer_faq ──┐
                       │               │
                       └→ query_rewrite → [vector|fulltext|graph]_retrieve
                                          ↓
                                        merge → rerank → build_prompt
                                          ↓              ↓
                                       refusal_check → generate
                                          ↓
                                       refuse → output
    ```

    Returns:
        Compiled StateGraph
    """
    # Create the graph with RAGState
    workflow = StateGraph(RAGState)

    # Add all nodes
    workflow.add_node("input", input_node)
    workflow.add_node("faq_match", faq_match_node)
    workflow.add_node("query_rewrite", query_rewrite_node)
    workflow.add_node("vector_retrieve", vector_retrieve_node)
    workflow.add_node("fulltext_retrieve", fulltext_retrieve_node)
    workflow.add_node("graph_retrieve", graph_retrieve_node)
    workflow.add_node("merge", merge_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("build_prompt", build_prompt_node)
    workflow.add_node("refusal_check", refusal_check_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("output", output_node)
    workflow.add_node("answer_faq", answer_faq_node)
    workflow.add_node("refuse", refuse_node)

    # Set entry point
    workflow.set_entry_point("input")

    # Linear edge: input → faq_match
    workflow.add_edge("input", "faq_match")

    # Conditional routing after FAQ match
    workflow.add_conditional_edges(
        "faq_match",
        route_after_faq,
        {
            "answer_faq": "answer_faq",
            "query_rewrite": "query_rewrite",
        },
    )

    # Parallel retrieval (fan-out from query_rewrite)
    workflow.add_edge("query_rewrite", "vector_retrieve")
    workflow.add_edge("query_rewrite", "fulltext_retrieve")
    workflow.add_edge("query_rewrite", "graph_retrieve")

    # Fan-in to merge (wait for all retrieval nodes)
    workflow.add_edge("vector_retrieve", "merge")
    workflow.add_edge("fulltext_retrieve", "merge")
    workflow.add_edge("graph_retrieve", "merge")

    # Continue pipeline
    workflow.add_edge("merge", "rerank")
    workflow.add_edge("rerank", "build_prompt")
    workflow.add_edge("build_prompt", "refusal_check")

    # Conditional routing after refusal check
    workflow.add_conditional_edges(
        "refusal_check",
        route_after_refusal,
        {
            "generate": "generate",
            "refuse": "refuse",
        },
    )

    # Final edges to output
    workflow.add_edge("generate", "output")
    workflow.add_edge("answer_faq", "output")
    workflow.add_edge("refuse", "output")

    # Output to end
    workflow.add_edge("output", END)

    logger.info("RAG Pipeline graph built successfully")
    return workflow


def compile_rag_graph(workflow: Optional[StateGraph] = None) -> any:
    """
    Compile the RAG Pipeline graph.

    Args:
        workflow: Optional pre-built workflow

    Returns:
        Compiled graph ready for execution
    """
    if workflow is None:
        workflow = build_rag_graph()

    compiled = workflow.compile()
    logger.info("RAG Pipeline graph compiled successfully")
    return compiled


# Create the default compiled graph for LangGraph API
graph = compile_rag_graph()


async def run_rag_pipeline(
    query: str,
    conversation_id: Optional[str] = None,
    agent_id: str = "default",
    enable_thinking: bool = False,
    conversation_history: Optional[list[dict]] = None,
) -> dict:
    """
    Run the RAG pipeline for a single query.

    Args:
        query: User query
        conversation_id: Optional conversation ID
        agent_id: Agent identifier
        enable_thinking: Enable thinking mode
        conversation_history: Previous conversation turns

    Returns:
        Pipeline result as dictionary
    """
    # Create initial state
    initial_state = create_initial_state(
        query=query,
        conversation_id=conversation_id,
        agent_id=agent_id,
        enable_thinking=enable_thinking,
        conversation_history=conversation_history,
    )

    # Run the graph
    result = await graph.ainvoke(initial_state)

    return result


async def stream_rag_pipeline(
    query: str,
    conversation_id: Optional[str] = None,
    agent_id: str = "default",
    enable_thinking: bool = False,
    conversation_history: Optional[list[dict]] = None,
):
    """
    Stream the RAG pipeline execution.

    Yields intermediate states as the pipeline progresses.

    Args:
        query: User query
        conversation_id: Optional conversation ID
        agent_id: Agent identifier
        enable_thinking: Enable thinking mode
        conversation_history: Previous conversation turns

    Yields:
        Intermediate state dictionaries
    """
    # Create initial state
    initial_state = create_initial_state(
        query=query,
        conversation_id=conversation_id,
        agent_id=agent_id,
        enable_thinking=enable_thinking,
        conversation_history=conversation_history,
    )

    # Stream the graph execution
    async for event in graph.astream(initial_state):
        yield event