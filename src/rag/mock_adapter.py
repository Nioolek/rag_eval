"""
Mock RAG adapter for testing and development.
"""

import asyncio
import random
import time
from typing import Any, Optional

from ..models.rag_response import (
    RAGResponse,
    QueryRewrite,
    FAQMatch,
    RetrievalResult,
    RerankResult,
    LLMOutput,
)
from ..models.annotation import Annotation
from ..core.logging import logger

from .base_adapter import RAGAdapter


class MockRAGAdapter(RAGAdapter):
    """
    Mock adapter for testing without real RAG service.
    Generates realistic mock responses.
    """

    def __init__(
        self,
        name: str = "mock",
        simulate_latency: bool = True,
        min_latency_ms: float = 100,
        max_latency_ms: float = 500,
    ):
        self._name = name
        self.simulate_latency = simulate_latency
        self.min_latency_ms = min_latency_ms
        self.max_latency_ms = max_latency_ms

    @property
    def name(self) -> str:
        return self._name

    async def query(
        self,
        query: str,
        conversation_history: Optional[list[str]] = None,
        agent_id: Optional[str] = None,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> RAGResponse:
        """Generate a mock RAG response."""
        start_time = time.time()

        # Simulate latency
        if self.simulate_latency:
            latency = random.uniform(
                self.min_latency_ms / 1000,
                self.max_latency_ms / 1000,
            )
            await asyncio.sleep(latency)

        # Generate mock response
        response = self._generate_mock_response(
            query=query,
            conversation_history=conversation_history or [],
            enable_thinking=enable_thinking,
        )

        response.latency_ms = (time.time() - start_time) * 1000
        return response

    def _generate_mock_response(
        self,
        query: str,
        conversation_history: list[str],
        enable_thinking: bool,
    ) -> RAGResponse:
        """Generate a realistic mock response."""
        response = RAGResponse(query=query)

        # Mock query rewrite
        if len(query) > 20:
            response.query_rewrite = QueryRewrite(
                original_query=query,
                rewritten_query=f"Expanded: {query}",
                rewrite_type="expansion",
                confidence=0.9,
            )

        # Mock FAQ match (30% chance)
        if random.random() < 0.3:
            response.faq_match = FAQMatch(
                matched=True,
                faq_id=f"faq_{random.randint(1, 100)}",
                faq_question=query,
                faq_answer=f"这是一个关于 '{query}' 的常见问题答案。",
                confidence=random.uniform(0.8, 0.99),
                similarity_score=random.uniform(0.85, 0.99),
            )

        # Mock retrieval results
        for i in range(random.randint(3, 8)):
            response.retrieval_results.append(RetrievalResult(
                document_id=f"doc_{i}",
                content=f"这是第 {i+1} 个检索到的文档片段，与查询 '{query}' 相关。",
                score=random.uniform(0.5, 0.95),
                rank=i + 1,
                metadata={"source": f"source_{i % 3}"},
            ))

        # Mock rerank results
        reranked = sorted(
            response.retrieval_results,
            key=lambda x: random.random(),
            reverse=True
        )
        for i, r in enumerate(reranked[:5]):
            response.rerank_results.append(RerankResult(
                document_id=r.document_id,
                content=r.content,
                original_score=r.score,
                rerank_score=random.uniform(0.6, 0.99),
                rank=i + 1,
            ))

        # Mock LLM output
        thinking = ""
        if enable_thinking:
            thinking = f"让我思考一下用户的问题：{query}\n分析：这是一个关于...的问题\n..."

        answer = f"根据检索到的信息，关于 '{query}' 的回答是：这是一个模拟的RAG系统响应。"
        if response.faq_match and response.faq_match.matched:
            answer = response.faq_match.faq_answer

        response.llm_output = LLMOutput(
            content=answer,
            thinking_process=thinking,
            token_usage={
                "prompt_tokens": random.randint(100, 500),
                "completion_tokens": random.randint(50, 200),
                "total_tokens": random.randint(150, 700),
            },
            model="gpt-4-mock",
            finish_reason="stop",
        )

        response.final_answer = answer

        # Mock refusal (10% chance)
        if random.random() < 0.1:
            response.is_refused = True
            response.final_answer = "抱歉，我无法回答这个问题。"
            response.llm_output.content = response.final_answer

        return response

    async def query_from_annotation(
        self,
        annotation: Annotation
    ) -> RAGResponse:
        """Query using annotation data."""
        return await self.query(
            query=annotation.query,
            conversation_history=annotation.conversation_history,
            agent_id=annotation.agent_id,
            enable_thinking=annotation.enable_thinking,
        )

    async def health_check(self) -> bool:
        """Always healthy for mock."""
        return True

    async def close(self) -> None:
        """No resources to close."""
        pass