"""
RAG Response model.
Captures all fields returned by RAG service for evaluation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class StageTiming(BaseModel):
    """
    RAG各阶段时间追踪模型。
    记录查询改写、FAQ匹配、检索、重排序、生成等阶段的耗时。
    """

    # 各阶段耗时（毫秒）
    query_rewrite_ms: float = 0.0
    faq_match_ms: float = 0.0
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0

    # 数据来源标记
    # "measured": 实测值, "extracted": 从响应提取, "calculated": 计算得出, "fallback": 按比例估算
    source: str = "unknown"

    # 各阶段数据来源详情
    # 例如: {"query_rewrite": "state.timing", "retrieval": "calculated.from_count"}
    extraction_details: dict[str, str] = Field(default_factory=dict)

    # 原始时间戳（可选）
    timestamps: dict[str, str] = Field(default_factory=dict)

    def get_stage_timings(self) -> dict[str, float]:
        """获取各阶段耗时字典"""
        return {
            "query_rewrite": self.query_rewrite_ms,
            "faq_match": self.faq_match_ms,
            "retrieval": self.retrieval_ms,
            "rerank": self.rerank_ms,
            "generation": self.generation_ms,
        }

    def get_percentages(self) -> dict[str, float]:
        """计算各阶段耗时占比"""
        if self.total_ms <= 0:
            return {}

        return {
            stage: (ms / self.total_ms * 100)
            for stage, ms in self.get_stage_timings().items()
        }

    def get_measured_total(self) -> float:
        """获取各阶段实测总耗时（可能与total_ms不同）"""
        return sum(self.get_stage_timings().values())


class QueryRewrite(BaseModel):
    """Query rewrite result from RAG."""
    original_query: str
    rewritten_query: str
    rewrite_type: str = ""  # e.g., "expansion", "clarification", "translation"
    confidence: float = 1.0


class FAQMatch(BaseModel):
    """FAQ matching result from RAG."""
    matched: bool = False
    faq_id: str = ""
    faq_question: str = ""
    faq_answer: str = ""
    confidence: float = 0.0
    similarity_score: float = 0.0


class RetrievalResult(BaseModel):
    """Single retrieval result."""
    document_id: str
    content: str
    score: float = 0.0
    rank: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    is_relevant: Optional[bool] = None  # Ground truth relevance


class RerankResult(BaseModel):
    """Reranking result."""
    document_id: str
    content: str
    original_score: float = 0.0
    rerank_score: float = 0.0
    rank: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMOutput(BaseModel):
    """LLM generated output."""
    content: str = ""
    thinking_process: str = ""  # For thinking mode
    token_usage: dict[str, int] = Field(default_factory=dict)
    model: str = ""
    finish_reason: str = ""


class RAGResponse(BaseModel):
    """
    Complete RAG response model.
    Captures all intermediate results for comprehensive evaluation.
    """
    # Response metadata
    id: str = Field(default_factory=lambda: str(uuid4()))
    query: str
    created_at: datetime = Field(default_factory=datetime.now)
    latency_ms: float = 0.0
    success: bool = True
    error_message: str = ""

    # Pipeline results
    query_rewrite: Optional[QueryRewrite] = None
    faq_match: Optional[FAQMatch] = None
    retrieval_results: list[RetrievalResult] = Field(default_factory=list)
    rerank_results: list[RerankResult] = Field(default_factory=list)
    llm_output: Optional[LLMOutput] = None

    # Final answer
    final_answer: str = ""
    is_refused: bool = False  # Whether the system refused to answer

    # Performance timing
    stage_timing: Optional[StageTiming] = None

    # Additional metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    def get_retrieved_contents(self) -> list[str]:
        """Get list of retrieved document contents."""
        return [r.content for r in self.retrieval_results]

    def get_reranked_contents(self) -> list[str]:
        """Get list of reranked document contents."""
        return [r.content for r in self.rerank_results]

    def get_top_k_contents(self, k: int = 5) -> list[str]:
        """Get top-k retrieved contents."""
        return self.get_retrieved_contents()[:k]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return self.model_dump(mode='json')

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RAGResponse":
        """Create from dictionary."""
        return cls(**data)


class RAGResponseAdapter:
    """Adapter for converting different RAG response formats to standard model."""

    @staticmethod
    def from_langgraph(data: dict[str, Any], query: str) -> RAGResponse:
        """Convert LangGraph response to RAGResponse."""
        response = RAGResponse(query=query)

        # Extract query rewrite if present
        if "query_rewrite" in data:
            qr = data["query_rewrite"]
            response.query_rewrite = QueryRewrite(
                original_query=query,
                rewritten_query=qr.get("rewritten_query", query),
                rewrite_type=qr.get("type", ""),
                confidence=qr.get("confidence", 1.0),
            )

        # Extract FAQ match if present
        if "faq_match" in data:
            fm = data["faq_match"]
            response.faq_match = FAQMatch(
                matched=fm.get("matched", False),
                faq_id=fm.get("faq_id", ""),
                faq_question=fm.get("question", ""),
                faq_answer=fm.get("answer", ""),
                confidence=fm.get("confidence", 0.0),
                similarity_score=fm.get("similarity", 0.0),
            )

        # Extract retrieval results
        if "retrieval" in data:
            for i, r in enumerate(data["retrieval"]):
                response.retrieval_results.append(RetrievalResult(
                    document_id=r.get("id", str(i)),
                    content=r.get("content", ""),
                    score=r.get("score", 0.0),
                    rank=i + 1,
                    metadata=r.get("metadata", {}),
                ))

        # Extract rerank results
        if "rerank" in data:
            for i, r in enumerate(data["rerank"]):
                response.rerank_results.append(RerankResult(
                    document_id=r.get("id", str(i)),
                    content=r.get("content", ""),
                    original_score=r.get("original_score", 0.0),
                    rerank_score=r.get("rerank_score", 0.0),
                    rank=i + 1,
                    metadata=r.get("metadata", {}),
                ))

        # Extract LLM output
        if "llm_output" in data or "answer" in data:
            llm_data = data.get("llm_output", {})
            response.final_answer = data.get("answer", llm_data.get("content", ""))
            response.is_refused = data.get("is_refused", False)
            response.llm_output = LLMOutput(
                content=response.final_answer,
                thinking_process=llm_data.get("thinking", ""),
                token_usage=llm_data.get("token_usage", {}),
                model=llm_data.get("model", ""),
                finish_reason=llm_data.get("finish_reason", ""),
            )

        return response