"""
LangGraph SSE (Server-Sent Events) adapter for RAG services.
Supports the new SSE streaming response format with metadata, values, and custom events.
"""

import asyncio
import json
import time
from typing import Any, Optional, AsyncGenerator

import aiohttp

from ..models.rag_response import (
    RAGResponse,
    QueryRewrite,
    FAQMatch,
    RetrievalResult,
    RerankResult,
    LLMOutput,
    StageTiming,
)
from ..models.annotation import Annotation
from ..core.exceptions import RAGConnectionError
from ..core.logging import logger

from .base_adapter import RAGAdapter, RAGAdapterConfig, StreamingChunk
from .sse_parser import (
    SSEEventParser,
    SSEEvent,
    SSEEventType,
    SSEStreamAccumulator,
    ContentBlock,
)


class LangGraphSSEAdapterConfig:
    """
    Configuration for LangGraph SSE adapter.

    Extends RAGAdapterConfig with SSE-specific options.
    """

    def __init__(
        self,
        service_url: str,
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        default_agent_id: str = "128",
        stream_endpoint: str = "/stream",
        health_endpoint: str = "/health",
        **kwargs: Any,
    ):
        self.service_url = service_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.default_agent_id = default_agent_id
        self.stream_endpoint = stream_endpoint
        self.health_endpoint = health_endpoint
        self.extra = kwargs


class LangGraphSSEAdapter(RAGAdapter):
    """
    Adapter for LangGraph RAG services with SSE streaming response format.

    Handles the new SSE format with:
    - event: metadata -> Run metadata (run_id, attempt)
    - event: values -> State updates (query, messages, routing_result, etc.)
    - event: custom -> UI content blocks + intermediate results
    - : (heartbeat) -> Keep-alive signals

    The adapter accumulates state across events and builds a complete RAGResponse.
    """

    def __init__(
        self,
        config: LangGraphSSEAdapterConfig,
    ):
        self.config = config
        self._parser = SSEEventParser()
        self._http_session: Optional[aiohttp.ClientSession] = None

    @property
    def name(self) -> str:
        return "langgraph_sse"

    async def _get_http_session(self) -> aiohttp.ClientSession:
        """Get or create shared HTTP session."""
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()
        return self._http_session

    async def query(
        self,
        query: str,
        conversation_history: Optional[list[str]] = None,
        agent_id: Optional[str] = None,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> RAGResponse:
        """
        Send a query to the RAG service via SSE streaming.

        Consumes the entire SSE stream and returns a complete RAGResponse.

        Args:
            query: User query
            conversation_history: Multi-turn conversation history
            agent_id: Agent identifier (defaults to configured default_agent_id)
            enable_thinking: Enable thinking mode
            **kwargs: Additional parameters

        Returns:
            RAGResponse with all pipeline results
        """
        start_time = time.time()
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                # Consume SSE stream and accumulate state
                accumulator = await self._consume_sse_stream(
                    query=query,
                    conversation_history=conversation_history or [],
                    agent_id=agent_id or self.config.default_agent_id,
                    enable_thinking=enable_thinking,
                    **kwargs,
                )

                # Build response from accumulated state
                latency_ms = (time.time() - start_time) * 1000
                response = self._build_response(accumulator, query, latency_ms)

                return response

            except asyncio.TimeoutError:
                last_error = RAGConnectionError(
                    f"RAG query timed out after {self.config.timeout}s"
                )
                logger.warning(f"RAG query attempt {attempt + 1} timed out")

            except aiohttp.ClientError as e:
                last_error = RAGConnectionError(f"RAG connection error: {e}")
                logger.warning(f"RAG query attempt {attempt + 1} failed: {e}")

            except Exception as e:
                last_error = e
                logger.warning(f"RAG query attempt {attempt + 1} failed: {e}")

            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))

        # All retries failed
        latency_ms = (time.time() - start_time) * 1000
        error_msg = str(last_error) if last_error else "Unknown error"

        return RAGResponse(
            query=query,
            success=False,
            error_message=error_msg,
            latency_ms=latency_ms,
        )

    async def _consume_sse_stream(
        self,
        query: str,
        conversation_history: list[str],
        agent_id: str,
        enable_thinking: bool,
        **kwargs: Any,
    ) -> SSEStreamAccumulator:
        """
        Consume SSE stream and accumulate state.

        Args:
            query: User query
            conversation_history: Conversation history
            agent_id: Agent identifier
            enable_thinking: Enable thinking mode

        Returns:
            SSEStreamAccumulator with accumulated state
        """
        url = f"{self.config.service_url}{self.config.stream_endpoint}"
        payload = {
            "query": query,
            "conversation_history": conversation_history,
            "agent_id": agent_id,
            "enable_thinking": enable_thinking,
            **kwargs,
        }

        session = await self._get_http_session()
        accumulator = SSEStreamAccumulator()

        timeout = aiohttp.ClientTimeout(total=self.config.timeout)

        async with session.post(
            url,
            json=payload,
            timeout=timeout,
            headers={
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
            },
        ) as response:
            if response.status != 200:
                raise RAGConnectionError(
                    f"RAG service returned status {response.status}"
                )

            # Read SSE stream line by line
            buffer = []
            async for line in response.content:
                line_str = line.decode("utf-8").strip()

                if not line_str:
                    # Empty line - process buffered event
                    if buffer:
                        events = self._parser.parse_event_batch(buffer)
                        for event in events:
                            accumulator.accumulate(event)
                        buffer = []
                else:
                    buffer.append(line_str)

            # Process any remaining buffered lines
            if buffer:
                events = self._parser.parse_event_batch(buffer)
                for event in events:
                    accumulator.accumulate(event)

        return accumulator

    def _build_response(
        self,
        accumulator: SSEStreamAccumulator,
        query: str,
        latency_ms: float,
    ) -> RAGResponse:
        """
        Build RAGResponse from accumulated SSE state.

        Args:
            accumulator: Accumulated state from SSE stream
            query: Original query
            latency_ms: Total latency in milliseconds

        Returns:
            RAGResponse with all pipeline results
        """
        state = accumulator.get_state()
        response = RAGResponse(query=query, latency_ms=latency_ms)

        # Extract query rewrite
        query_rewrite_data = state.get("query_rewrite_result")
        if query_rewrite_data:
            response.query_rewrite = self._build_query_rewrite(query_rewrite_data, query)

        # Extract FAQ match
        faq_data = state.get("faq_result")
        if faq_data:
            response.faq_match = self._build_faq_match(faq_data)

        # Extract retrieval results (combine internal and external)
        retrieve_data = state.get("retrieve_result")
        external_retrieve_data = state.get("external_retrieve_result")
        response.retrieval_results = self._build_retrieval_results(
            retrieve_data, external_retrieve_data
        )

        # Extract rerank results
        rerank_data = state.get("rerank_result")
        if rerank_data:
            response.rerank_results = self._build_rerank_results(rerank_data)

        # Extract answer
        answer_data = state.get("answer_result")
        if answer_data:
            response.final_answer = answer_data.get("answer_content", "")
            response.is_refused = answer_data.get("is_refused", False)

            # Build LLM output
            response.llm_output = LLMOutput(
                content=response.final_answer,
                thinking_process=answer_data.get("thinking_process", ""),
                token_usage=answer_data.get("token_usage", {}),
                model=answer_data.get("model", ""),
                finish_reason=answer_data.get("finish_reason", ""),
            )

            # If retrieval results not already set, use source_list from answer
            if not response.retrieval_results and "source_list" in answer_data:
                response.retrieval_results = self._build_retrieval_from_sources(
                    answer_data["source_list"]
                )

        # Build stage timing from event timestamps
        response.stage_timing = self._build_stage_timing(state, latency_ms)

        # Add metadata
        response.metadata = {
            "run_id": state.get("run_id"),
            "attempt": state.get("attempt"),
            "content_blocks_count": len(state.get("content_blocks", [])),
        }

        return response

    def _build_query_rewrite(
        self, data: dict[str, Any], original_query: str
    ) -> QueryRewrite:
        """Build QueryRewrite from SSE data."""
        return QueryRewrite(
            original_query=data.get("original_query", original_query),
            rewritten_query=data.get("rewritten_query", data.get("query", original_query)),
            rewrite_type=data.get("rewrite_type", data.get("type", "")),
            confidence=data.get("confidence", 1.0),
        )

    def _build_faq_match(self, data: dict[str, Any]) -> FAQMatch:
        """Build FAQMatch from SSE data."""
        return FAQMatch(
            matched=data.get("matched", False),
            faq_id=data.get("faq_id", data.get("id", "")),
            faq_question=data.get("question", data.get("faq_question", "")),
            faq_answer=data.get("answer", data.get("faq_answer", "")),
            confidence=data.get("confidence", 0.0),
            similarity_score=data.get("similarity", data.get("similarity_score", 0.0)),
        )

    def _build_retrieval_results(
        self,
        retrieve_data: Optional[dict[str, Any]],
        external_data: Optional[dict[str, Any]],
    ) -> list[RetrievalResult]:
        """Build RetrievalResult list from SSE data."""
        results = []

        # Internal retrieval results
        if retrieve_data:
            docs = retrieve_data.get("documents", retrieve_data.get("results", []))
            for i, doc in enumerate(docs):
                results.append(RetrievalResult(
                    document_id=doc.get("id", doc.get("document_id", str(i))),
                    content=doc.get("content", doc.get("text", "")),
                    score=doc.get("score", 0.0),
                    rank=i + 1,
                    metadata=doc.get("metadata", {}),
                ))

        # External API retrieval results
        if external_data:
            docs = external_data.get("documents", external_data.get("results", []))
            for i, doc in enumerate(docs):
                results.append(RetrievalResult(
                    document_id=doc.get("id", doc.get("document_id", f"external_{i}")),
                    content=doc.get("content", doc.get("text", "")),
                    score=doc.get("score", 0.0),
                    rank=len(results) + i + 1,
                    metadata={"source": "external_api", **doc.get("metadata", {})},
                ))

        return results

    def _build_retrieval_from_sources(
        self, sources: list[dict[str, Any]]
    ) -> list[RetrievalResult]:
        """Build RetrievalResult list from answer source_list."""
        results = []
        for i, source in enumerate(sources):
            results.append(RetrievalResult(
                document_id=source.get("id", source.get("document_id", str(i))),
                content=source.get("content", source.get("text", "")),
                score=source.get("score", 0.0),
                rank=i + 1,
                metadata=source.get("metadata", {}),
            ))
        return results

    def _build_rerank_results(
        self, data: dict[str, Any]
    ) -> list[RerankResult]:
        """Build RerankResult list from SSE data."""
        results = []
        docs = data.get("documents", data.get("results", []))

        for i, doc in enumerate(docs):
            results.append(RerankResult(
                document_id=doc.get("id", doc.get("document_id", str(i))),
                content=doc.get("content", doc.get("text", "")),
                original_score=doc.get("original_score", doc.get("score", 0.0)),
                rerank_score=doc.get("rerank_score", 0.0),
                rank=i + 1,
                metadata=doc.get("metadata", {}),
            ))

        return results

    def _build_stage_timing(
        self, state: dict[str, Any], total_ms: float
    ) -> StageTiming:
        """Build StageTiming from event timestamps and accumulated state."""
        timestamps = state.get("event_timestamps", {})

        timing = StageTiming(
            total_ms=total_ms,
            source="streaming",
        )

        # Calculate stage timings from event timestamps
        if timestamps:
            # Try to compute timing from event order
            event_order = ["metadata", "values", "custom"]
            prev_time = None

            for event_type in event_order:
                if event_type in timestamps:
                    curr_time = timestamps[event_type]
                    if prev_time is not None:
                        duration = (curr_time - prev_time) * 1000
                        # Map event types to stages
                        if event_type == "values":
                            timing.query_rewrite_ms = duration * 0.3
                            timing.faq_match_ms = duration * 0.3
                        elif event_type == "custom":
                            timing.retrieval_ms = duration * 0.4
                            timing.rerank_ms = duration * 0.3
                            timing.generation_ms = duration * 0.3
                    prev_time = curr_time

        # If we couldn't extract timing, use fallback proportions
        if timing.get_measured_total() == 0:
            timing.query_rewrite_ms = total_ms * 0.05
            timing.faq_match_ms = total_ms * 0.05
            timing.retrieval_ms = total_ms * 0.20
            timing.rerank_ms = total_ms * 0.10
            timing.generation_ms = total_ms * 0.60
            timing.source = "fallback"

        return timing

    async def query_from_annotation(
        self, annotation: Annotation
    ) -> RAGResponse:
        """Query RAG using annotation data."""
        return await self.query(
            query=annotation.query,
            conversation_history=annotation.conversation_history,
            agent_id=annotation.agent_id or self.config.default_agent_id,
            enable_thinking=annotation.enable_thinking,
        )

    async def health_check(self) -> bool:
        """Check if RAG service is healthy."""
        try:
            session = await self._get_http_session()
            url = f"{self.config.service_url}{self.config.health_endpoint}"

            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=5.0),
            ) as response:
                return response.status == 200

        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    async def close(self) -> None:
        """Close adapter and release resources."""
        if self._http_session:
            await self._http_session.close()
            self._http_session = None

    async def stream_query(
        self,
        query: str,
        conversation_history: Optional[list[str]] = None,
        agent_id: Optional[str] = None,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamingChunk, None]:
        """
        Stream query execution with real-time content blocks.

        Yields StreamingChunk objects for each content block from custom events,
        allowing real-time UI updates.

        Args:
            query: User query
            conversation_history: Conversation history
            agent_id: Agent identifier
            enable_thinking: Enable thinking mode
            **kwargs: Additional parameters

        Yields:
            StreamingChunk objects with stage information
        """
        start_time = time.time()

        try:
            url = f"{self.config.service_url}{self.config.stream_endpoint}"
            payload = {
                "query": query,
                "conversation_history": conversation_history or [],
                "agent_id": agent_id or self.config.default_agent_id,
                "enable_thinking": enable_thinking,
                **kwargs,
            }

            session = await self._get_http_session()
            accumulator = SSEStreamAccumulator()
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)

            async with session.post(
                url,
                json=payload,
                timeout=timeout,
                headers={
                    "Accept": "text/event-stream",
                    "Content-Type": "application/json",
                },
            ) as response:
                if response.status != 200:
                    raise RAGConnectionError(
                        f"RAG streaming returned status {response.status}"
                    )

                # Yield initial chunk
                yield StreamingChunk(
                    stage="start",
                    content="开始处理查询...",
                    metadata={"query": query},
                )

                buffer = []
                async for line in response.content:
                    line_str = line.decode("utf-8").strip()

                    if not line_str:
                        if buffer:
                            events = self._parser.parse_event_batch(buffer)
                            for event in events:
                                accumulator.accumulate(event)

                                # Yield chunks for content blocks
                                async for chunk in self._event_to_chunks(event):
                                    yield chunk

                            buffer = []
                    else:
                        buffer.append(line_str)

                # Process remaining buffer
                if buffer:
                    events = self._parser.parse_event_batch(buffer)
                    for event in events:
                        accumulator.accumulate(event)
                        async for chunk in self._event_to_chunks(event):
                            yield chunk

            # Yield final chunk with complete response
            state = accumulator.get_state()
            latency_ms = (time.time() - start_time) * 1000
            answer_data = state.get("answer_result", {})

            yield StreamingChunk(
                stage="final",
                content=answer_data.get("answer_content", ""),
                is_final=True,
                metadata={
                    "success": True,
                    "latency_ms": latency_ms,
                    "is_refused": answer_data.get("is_refused", False),
                },
            )

        except Exception as e:
            logger.error(f"Streaming query failed: {e}")
            yield StreamingChunk(
                stage="error",
                content=str(e),
                is_final=True,
                metadata={"success": False, "error": str(e)},
            )

    async def _event_to_chunks(
        self, event: SSEEvent
    ) -> AsyncGenerator[StreamingChunk, None]:
        """
        Convert an SSE event to StreamingChunk(s).

        Args:
            event: Parsed SSE event

        Yields:
            StreamingChunk objects for UI display
        """
        if event.event_type == SSEEventType.HEARTBEAT:
            return

        if event.event_type == SSEEventType.METADATA:
            yield StreamingChunk(
                stage="metadata",
                content=f"Run ID: {event.data.get('run_id', 'unknown')}",
                metadata=event.data,
            )
            return

        if event.event_type == SSEEventType.VALUES:
            # Extract relevant state updates
            query = event.data.get("query")
            if query:
                yield StreamingChunk(
                    stage="query",
                    content=query,
                    metadata={"type": "user_query"},
                )
            return

        if event.event_type == SSEEventType.CUSTOM:
            # Parse content block
            content_block = self._parser.parse_content_block(event.data)

            if content_block and content_block.is_show:
                stage = self._parser.get_stage_from_title(content_block.title)

                yield StreamingChunk(
                    stage=stage,
                    content=content_block.content,
                    metadata={
                        "block_id": content_block.block_id,
                        "block_type": content_block.block_type,
                        "title": content_block.title,
                        "is_done": content_block.is_done,
                    },
                )
            return