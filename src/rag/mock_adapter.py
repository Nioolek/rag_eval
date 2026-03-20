"""
Mock RAG adapter for testing and development.
"""

import asyncio
import random
import time
from typing import Any, Optional, AsyncGenerator

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

from .base_adapter import RAGAdapter, StreamingChunk
from .timing_config import TimingExtractionConfig, get_mock_config
from .timing_extractor import TimingExtractor


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
        timing_config: Optional[TimingExtractionConfig] = None,
    ):
        self._name = name
        self.simulate_latency = simulate_latency
        self.min_latency_ms = min_latency_ms
        self.max_latency_ms = max_latency_ms
        self.timing_config = timing_config or get_mock_config()
        self.timing_extractor = TimingExtractor(self.timing_config)

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

        # Extract stage timing
        raw_data = response.to_dict()
        response.stage_timing = self.timing_extractor.extract(
            raw_data, response.latency_ms
        )

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

    async def stream_query(
        self,
        query: str,
        conversation_history: Optional[list[str]] = None,
        agent_id: Optional[str] = None,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamingChunk, None]:
        """
        Stream mock query execution with simulated intermediate results.
        Provides realistic streaming demo for UI development.
        """
        start_time = time.time()

        # Simulate query rewrite stage
        await asyncio.sleep(random.uniform(0.1, 0.3))
        if len(query) > 20:
            yield StreamingChunk(
                stage="query_rewrite",
                content=f"**原查询**: {query}\n\n**改写后**: Expanded: {query}",
                metadata={
                    "rewrite_type": "expansion",
                    "confidence": 0.9
                }
            )

        # Simulate FAQ match stage
        await asyncio.sleep(random.uniform(0.1, 0.3))
        if random.random() < 0.3:
            yield StreamingChunk(
                stage="faq_match",
                content=f"✅ 匹配到 FAQ: 这是一个关于 '{query}' 的常见问题答案。",
                metadata={
                    "matched": True,
                    "confidence": random.uniform(0.8, 0.99),
                }
            )
        else:
            yield StreamingChunk(
                stage="faq_match",
                content="❌ 未匹配到 FAQ",
                metadata={"matched": False}
            )

        # Simulate retrieval stage
        await asyncio.sleep(random.uniform(0.2, 0.5))
        doc_count = random.randint(3, 8)
        docs_preview = "\n".join([
            f"  {i+1}. 文档片段 {i+1} (相关度: {random.uniform(0.5, 0.95):.2f})"
            for i in range(min(doc_count, 5))
        ])
        yield StreamingChunk(
            stage="retrieval",
            content=f"📚 检索到 {doc_count} 个文档:\n{docs_preview}",
            metadata={"count": doc_count}
        )

        # Simulate rerank stage
        await asyncio.sleep(random.uniform(0.1, 0.3))
        rerank_count = min(doc_count, 5)
        yield StreamingChunk(
            stage="rerank",
            content=f"🔄 重排序后保留 {rerank_count} 个最相关文档",
            metadata={"count": rerank_count}
        )

        # Simulate thinking stage (if enabled)
        if enable_thinking:
            await asyncio.sleep(random.uniform(0.2, 0.4))
            thinking = f"""💭 **思考过程**

分析用户问题: {query}

1. 识别关键信息...
2. 匹配检索文档...
3. 综合答案..."""
            yield StreamingChunk(
                stage="thinking",
                content=thinking,
                metadata={"tokens": {"thinking": random.randint(50, 150)}}
            )

        # Simulate generation stage (token by token)
        await asyncio.sleep(random.uniform(0.3, 0.6))
        answer = f"根据检索到的信息，关于 '{query}' 的回答是：这是一个模拟的RAG系统响应，用于演示流式输出功能。"

        # Yield tokens in small chunks for realistic streaming effect
        words = answer.split()
        current_content = ""
        for i, word in enumerate(words):
            current_content += word + " "
            if i % 3 == 0:  # Yield every 3 words
                await asyncio.sleep(0.05)
                yield StreamingChunk(
                    stage="generation",
                    content=current_content.strip(),
                    metadata={"streaming": True, "progress": (i + 1) / len(words)}
                )

        # Final result
        latency_ms = (time.time() - start_time) * 1000
        yield StreamingChunk(
            stage="final",
            content=answer,
            is_final=True,
            metadata={
                "success": True,
                "latency_ms": latency_ms,
                "is_refused": False
            }
        )