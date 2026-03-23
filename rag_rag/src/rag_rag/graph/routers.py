"""
Conditional Routing Functions for RAG Pipeline.

Determines which path to take based on state conditions.
"""

from typing import Literal

from rag_rag.graph.state import RAGState


def route_after_faq(state: RAGState) -> Literal["answer_faq", "query_rewrite"]:
    """
    Route after FAQ match node.

    If FAQ matched, go directly to answer_faq.
    Otherwise, proceed with query rewrite and retrieval.
    """
    if state.get("faq_matched", False):
        return "answer_faq"
    return "query_rewrite"


def route_after_refusal(state: RAGState) -> Literal["refuse", "generate"]:
    """
    Route after refusal check node.

    If should refuse, go to refuse node.
    Otherwise, proceed with generation.
    """
    if state.get("should_refuse", False):
        return "refuse"
    return "generate"