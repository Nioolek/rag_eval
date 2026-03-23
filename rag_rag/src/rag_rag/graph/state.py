"""
RAG Pipeline State Definition.

Defines the RAGState TypedDict that flows through all nodes in the LangGraph.
"""

from datetime import datetime
from typing import Annotated, Any, Optional

from typing_extensions import TypedDict


def merge_dicts(left: dict, right: dict) -> dict:
    """Merge two dictionaries, with right taking precedence."""
    if not isinstance(left, dict):
        left = {}
    if not isinstance(right, dict):
        right = {}
    result = left.copy()
    result.update(right)
    return result


def merge_stage_timing(left: dict, right: dict) -> dict:
    """Merge stage timing dictionaries."""
    if not isinstance(left, dict):
        left = {}
    if not isinstance(right, dict):
        right = {}
    result = left.copy()
    result.update(right)
    return result


def replace_value(left: Any, right: Any) -> Any:
    """Replace left value with right value."""
    return right if right is not None else left


def append_to_list(left: list, right: list) -> list:
    """Append right list to left list."""
    return left + right


class QueryRewriteResult(TypedDict):
    """Query rewrite result structure."""

    original_query: str
    rewritten_query: str
    rewrite_type: str  # expansion, clarification, multi_turn, translation
    confidence: float
    timing_ms: float


class FAQMatchResult(TypedDict):
    """FAQ match result structure."""

    matched: bool
    faq_id: str
    question: str
    answer: str
    confidence: float
    similarity: float
    match_type: str  # exact, semantic
    timing_ms: float


class RetrievalDocument(TypedDict):
    """Single retrieved document."""

    document_id: str
    content: str
    score: float
    source: str  # vector, fulltext, graph
    metadata: dict[str, Any]


class RerankDocument(TypedDict):
    """Reranked document with scores."""

    document_id: str
    content: str
    original_score: float
    rerank_score: float
    rank: int
    metadata: dict[str, Any]


class LLMOutputResult(TypedDict):
    """LLM generation output."""

    content: str
    thinking_process: str
    token_usage: dict[str, int]  # prompt_tokens, completion_tokens, total_tokens
    model: str
    finish_reason: str


class StageTiming(TypedDict):
    """Timing for each pipeline stage in milliseconds."""

    input_ms: float
    faq_match_ms: float
    query_rewrite_ms: float
    vector_retrieve_ms: float
    fulltext_retrieve_ms: float
    graph_retrieve_ms: float
    merge_ms: float
    rerank_ms: float
    build_prompt_ms: float
    refusal_check_ms: float
    generation_ms: float
    total_ms: float


class ErrorInfo(TypedDict):
    """Error information structure."""

    stage: str
    type: str
    message: str
    timestamp: str


class ConversationMessage(TypedDict):
    """Single conversation message."""

    role: str  # user, assistant
    content: str
    timestamp: str


class RAGState(TypedDict):
    """
    RAG Pipeline State Definition.

    This state flows through all nodes in the LangGraph pipeline.
    Each node receives this state and returns partial updates.

    The Annotated types define how updates from parallel nodes are merged.
    """

    # === Input Layer ===
    query: str  # User's original question
    conversation_history: list[ConversationMessage]  # Previous turns
    conversation_id: str  # Session identifier
    agent_id: str  # Agent identifier
    enable_thinking: bool  # Enable CoT reasoning mode

    # === FAQ Match Stage ===
    faq_matched: bool
    faq_result: Optional[FAQMatchResult]

    # === Query Rewrite Stage ===
    query_rewrite: Optional[QueryRewriteResult]

    # === Retrieval Stage (Each node writes to its own key) ===
    vector_results: list[RetrievalDocument]
    fulltext_results: list[RetrievalDocument]
    graph_results: list[RetrievalDocument]
    merged_results: list[RetrievalDocument]

    # === Rerank Stage ===
    reranked_results: list[RerankDocument]
    rerank_scores: list[float]

    # === Refusal Stage ===
    should_refuse: bool
    refusal_reason: str
    refusal_type: str  # out_of_domain, sensitive, low_relevance

    # === Prompt Stage ===
    system_prompt: str
    context_prompt: str
    few_shot_examples: list[dict[str, str]]
    final_prompt: str
    prompt_template_name: str

    # === Generation Stage ===
    llm_output: Optional[LLMOutputResult]
    thinking_process: str
    final_answer: str
    is_refused: bool

    # === Output ===
    answer: str  # Final answer (for RAGResponseAdapter compatibility)
    retrieval: list[dict[str, Any]]  # Retrieval results (for RAGResponseAdapter compatibility)
    rerank: list[dict[str, Any]]  # Rerank results (for RAGResponseAdapter compatibility)

    # === Metadata ===
    stage_timing: Annotated[dict[str, float], merge_stage_timing]
    metadata: dict[str, Any]
    errors: Annotated[list[ErrorInfo], append_to_list]


def create_initial_state(
    query: str,
    conversation_id: str = "",
    agent_id: str = "default",
    enable_thinking: bool = False,
    conversation_history: Optional[list[ConversationMessage]] = None,
) -> RAGState:
    """Create an initial RAG state with default values."""
    return RAGState(
        # Input
        query=query,
        conversation_history=conversation_history or [],
        conversation_id=conversation_id,
        agent_id=agent_id,
        enable_thinking=enable_thinking,
        # FAQ
        faq_matched=False,
        faq_result=None,
        # Query rewrite
        query_rewrite=None,
        # Retrieval
        vector_results=[],
        fulltext_results=[],
        graph_results=[],
        merged_results=[],
        # Rerank
        reranked_results=[],
        rerank_scores=[],
        # Refusal
        should_refuse=False,
        refusal_reason="",
        refusal_type="",
        # Prompt
        system_prompt="",
        context_prompt="",
        few_shot_examples=[],
        final_prompt="",
        prompt_template_name="default",
        # Generation
        llm_output=None,
        thinking_process="",
        final_answer="",
        is_refused=False,
        # Output
        answer="",
        retrieval=[],
        rerank=[],
        # Metadata
        stage_timing=StageTiming(
            input_ms=0.0,
            faq_match_ms=0.0,
            query_rewrite_ms=0.0,
            vector_retrieve_ms=0.0,
            fulltext_retrieve_ms=0.0,
            graph_retrieve_ms=0.0,
            merge_ms=0.0,
            rerank_ms=0.0,
            build_prompt_ms=0.0,
            refusal_check_ms=0.0,
            generation_ms=0.0,
            total_ms=0.0,
        ),
        metadata={},
        errors=[],
    )


def state_to_rag_response_format(state: RAGState) -> dict[str, Any]:
    """
    Convert RAGState to format compatible with RAGResponseAdapter.from_langgraph().

    This ensures the output matches what the evaluation system expects.
    """
    # Build query_rewrite format
    query_rewrite = None
    if state.get("query_rewrite"):
        qr = state["query_rewrite"]
        query_rewrite = {
            "rewritten_query": qr.get("rewritten_query", state["query"]),
            "type": qr.get("rewrite_type", ""),
            "confidence": qr.get("confidence", 1.0),
        }

    # Build faq_match format
    faq_match = None
    if state.get("faq_result"):
        fr = state["faq_result"]
        faq_match = {
            "matched": state.get("faq_matched", False),
            "faq_id": fr.get("faq_id", ""),
            "question": fr.get("question", ""),
            "answer": fr.get("answer", ""),
            "confidence": fr.get("confidence", 0.0),
            "similarity": fr.get("similarity", 0.0),
        }

    # Build retrieval format
    retrieval = []
    for doc in state.get("reranked_results") or state.get("merged_results") or []:
        retrieval.append({
            "id": doc.get("document_id", ""),
            "content": doc.get("content", ""),
            "score": doc.get("rerank_score") or doc.get("score", 0.0),
            "metadata": doc.get("metadata", {}),
        })

    # Build rerank format
    rerank = []
    for doc in state.get("reranked_results", []):
        rerank.append({
            "id": doc.get("document_id", ""),
            "content": doc.get("content", ""),
            "original_score": doc.get("original_score", 0.0),
            "rerank_score": doc.get("rerank_score", 0.0),
        })

    # Build llm_output format
    llm_output = None
    if state.get("llm_output"):
        lo = state["llm_output"]
        llm_output = {
            "content": lo.get("content", ""),
            "thinking": lo.get("thinking_process", ""),
            "token_usage": lo.get("token_usage", {}),
            "model": lo.get("model", ""),
        }

    result = {
        "query": state["query"],
        "retrieval": retrieval,
        "rerank": rerank,
        "answer": state.get("final_answer", ""),
        "is_refused": state.get("is_refused", False),
        "stage_timing": state.get("stage_timing", {}),
        "metadata": state.get("metadata", {}),
    }

    # Only include optional fields if they have values
    if query_rewrite is not None:
        result["query_rewrite"] = query_rewrite
    if faq_match is not None:
        result["faq_match"] = faq_match
    if llm_output is not None:
        result["llm_output"] = llm_output

    return result