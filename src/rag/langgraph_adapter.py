"""
LangGraph RemoteGraph adapter for RAG services.
"""

import asyncio
import time
from typing import Any, Optional, AsyncGenerator

import aiohttp

from ..models.rag_response import RAGResponse, RAGResponseAdapter
from ..models.annotation import Annotation
from ..core.exceptions import RAGConnectionError
from ..core.logging import logger

from .base_adapter import RAGAdapter, RAGAdapterConfig, StreamingChunk
from .timing_config import TimingExtractionConfig, get_default_config
from .timing_extractor import TimingExtractor


class LangGraphAdapter(RAGAdapter):
    """
    Adapter for LangGraph RemoteGraph RAG services.
    Implements async calls with retry logic.
    """

    # Default assistant ID for LangGraph
    DEFAULT_ASSISTANT_ID = "rag_agent"

    def __init__(
        self,
        config: RAGAdapterConfig,
        assistant_id: Optional[str] = None,
        timing_config: Optional[TimingExtractionConfig] = None,
    ):
        self.config = config
        self.assistant_id = assistant_id or self.DEFAULT_ASSISTANT_ID
        self.timing_config = timing_config or get_default_config()
        self.timing_extractor = TimingExtractor(self.timing_config)
        self._client = None
        self._initialized = False
        self._http_session: Optional[Any] = None  # aiohttp.ClientSession

    @property
    def name(self) -> str:
        return "langgraph"

    async def _get_http_session(self):
        """Get or create shared HTTP session."""
        import aiohttp
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()
        return self._http_session

    async def initialize(self) -> None:
        """Initialize the LangGraph client."""
        try:
            # Import here to allow module to load without langgraph
            from langgraph.pregel.remote import RemoteGraph

            # LangGraph 1.x: assistant_id is first positional arg, url is keyword arg
            self._client = RemoteGraph(
                self.assistant_id,
                url=self.config.service_url,
            )
            self._initialized = True
            logger.info(f"Initialized LangGraph adapter: {self.config.service_url} (assistant: {self.assistant_id})")
        except ImportError:
            logger.warning(
                "langgraph not installed, using mock client. "
                "Install with: pip install langgraph"
            )
            self._initialized = True
        except Exception as e:
            raise RAGConnectionError(
                f"Failed to initialize LangGraph client: {e}"
            )

    async def query(
        self,
        query: str,
        conversation_history: Optional[list[str]] = None,
        agent_id: Optional[str] = None,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> RAGResponse:
        """
        Send a query to the RAG service with retry logic.
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                response, raw_data = await self._query_internal(
                    query=query,
                    conversation_history=conversation_history or [],
                    agent_id=agent_id or "default",
                    enable_thinking=enable_thinking,
                    **kwargs,
                )

                latency_ms = (time.time() - start_time) * 1000
                response.latency_ms = latency_ms

                # Extract stage timing
                response.stage_timing = self.timing_extractor.extract(
                    raw_data, latency_ms
                )

                return response

            except Exception as e:
                last_error = e
                logger.warning(
                    f"RAG query attempt {attempt + 1} failed: {e}"
                )
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

    async def _query_internal(
        self,
        query: str,
        conversation_history: list[str],
        agent_id: str,
        enable_thinking: bool,
        **kwargs: Any,
    ) -> tuple[RAGResponse, dict[str, Any]]:
        """Internal query implementation. Returns (response, raw_data)."""
        if self._client is None:
            # Fallback to HTTP request if RemoteGraph not available
            return await self._query_via_http(
                query=query,
                conversation_history=conversation_history,
                agent_id=agent_id,
                enable_thinking=enable_thinking,
                **kwargs,
            )

        # Build input for LangGraph
        input_data = {
            "query": query,
            "conversation_history": conversation_history,
            "agent_id": agent_id,
            "enable_thinking": enable_thinking,
            **kwargs,
        }

        # Invoke the graph
        result = await asyncio.wait_for(
            self._client.ainvoke(input_data),
            timeout=self.config.timeout,
        )

        # Parse response
        raw_data = result if isinstance(result, dict) else result.__dict__ if hasattr(result, '__dict__') else {}
        response = RAGResponseAdapter.from_langgraph(result, query)
        return response, raw_data

    async def _query_via_http(
        self,
        query: str,
        conversation_history: list[str],
        agent_id: str,
        enable_thinking: bool,
        **kwargs: Any,
    ) -> tuple[RAGResponse, dict[str, Any]]:
        """Fallback HTTP query when LangGraph client not available. Returns (response, raw_data)."""
        url = f"{self.config.service_url}/invoke"
        payload = {
            "query": query,
            "conversation_history": conversation_history,
            "agent_id": agent_id,
            "enable_thinking": enable_thinking,
            **kwargs,
        }

        session = await self._get_http_session()
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
        ) as response:
            if response.status != 200:
                raise RAGConnectionError(
                    f"RAG service returned status {response.status}"
                )

            data = await response.json()
            response = RAGResponseAdapter.from_langgraph(data, query)
            return response, data

    async def query_from_annotation(
        self,
        annotation: Annotation
    ) -> RAGResponse:
        """Query RAG using annotation data."""
        return await self.query(
            query=annotation.query,
            conversation_history=annotation.conversation_history,
            agent_id=annotation.agent_id,
            enable_thinking=annotation.enable_thinking,
        )

    async def health_check(self) -> bool:
        """Check if RAG service is healthy."""
        try:
            if self._client:
                # Try a simple invocation
                result = await asyncio.wait_for(
                    self._client.ainvoke({"query": "health check"}),
                    timeout=5.0,
                )
                return True
            else:
                # HTTP health check
                import aiohttp
                session = await self._get_http_session()
                async with session.get(
                    f"{self.config.service_url}/health",
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
        self._client = None
        self._initialized = False

    async def stream_query(
        self,
        query: str,
        conversation_history: Optional[list[str]] = None,
        agent_id: Optional[str] = None,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamingChunk, None]:
        """
        Stream query execution with intermediate results.

        Yields StreamingChunk objects for each stage:
        - query_rewrite: Query rewriting results
        - faq_match: FAQ matching results
        - retrieval: Document retrieval results
        - rerank: Reranking results
        - generation: LLM generation (with thinking if enabled)
        - final: Final response
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        try:
            # Try streaming via LangGraph client
            if self._client:
                async for chunk in self._stream_via_langgraph(
                    query=query,
                    conversation_history=conversation_history or [],
                    agent_id=agent_id or "default",
                    enable_thinking=enable_thinking,
                    **kwargs,
                ):
                    yield chunk
            else:
                # Fallback to HTTP streaming
                async for chunk in self._stream_via_http(
                    query=query,
                    conversation_history=conversation_history or [],
                    agent_id=agent_id or "default",
                    enable_thinking=enable_thinking,
                    **kwargs,
                ):
                    yield chunk

        except Exception as e:
            logger.error(f"Streaming query failed: {e}")
            yield StreamingChunk(
                stage="error",
                content=str(e),
                is_final=True,
                metadata={"success": False, "error": str(e)}
            )

    async def _stream_via_langgraph(
        self,
        query: str,
        conversation_history: list[str],
        agent_id: str,
        enable_thinking: bool,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamingChunk, None]:
        """Stream via LangGraph client with streaming support."""
        input_data = {
            "query": query,
            "conversation_history": conversation_history,
            "agent_id": agent_id,
            "enable_thinking": enable_thinking,
            **kwargs,
        }

        # Check if client supports streaming
        if hasattr(self._client, 'astream'):
            try:
                async for event in self._client.astream(input_data):
                    # Parse streaming events
                    chunk = self._parse_stream_event(event)
                    if chunk:
                        yield chunk
                return
            except Exception as e:
                logger.warning(f"LangGraph streaming failed, falling back: {e}")

        # Fallback: run query and yield stages
        response, raw_data = await self._query_internal(
            query=query,
            conversation_history=conversation_history,
            agent_id=agent_id,
            enable_thinking=enable_thinking,
            **kwargs,
        )

        async for chunk in self._yield_response_chunks(response):
            yield chunk

    async def _stream_via_http(
        self,
        query: str,
        conversation_history: list[str],
        agent_id: str,
        enable_thinking: bool,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamingChunk, None]:
        """Stream via HTTP SSE endpoint."""
        import aiohttp
        import json

        url = f"{self.config.service_url}/stream"
        payload = {
            "query": query,
            "conversation_history": conversation_history,
            "agent_id": agent_id,
            "enable_thinking": enable_thinking,
            **kwargs,
        }

        try:
            session = await self._get_http_session()
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            ) as response:
                if response.status != 200:
                    raise RAGConnectionError(
                        f"RAG streaming returned status {response.status}"
                    )

                # Parse SSE stream
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        data = json.loads(line[6:])
                        chunk = self._parse_sse_data(data)
                        if chunk:
                            yield chunk

        except aiohttp.ClientError as e:
            # Fallback to non-streaming
            logger.warning(f"HTTP streaming failed, falling back: {e}")
            response, raw_data = await self._query_via_http(
                query=query,
                conversation_history=conversation_history,
                agent_id=agent_id,
                enable_thinking=enable_thinking,
                **kwargs,
            )
            async for chunk in self._yield_response_chunks(response):
                yield chunk

    def _parse_stream_event(self, event: dict[str, Any]) -> Optional[StreamingChunk]:
        """Parse a LangGraph streaming event."""
        if not event:
            return None

        # Handle different event types
        event_type = event.get("event", "")
        data = event.get("data", {})

        if event_type == "on_chain_start":
            return StreamingChunk(
                stage="start",
                content=f"开始处理: {data.get('name', 'unknown')}",
                metadata={"event": event_type}
            )

        elif event_type == "on_chain_end":
            name = data.get("name", "")
            output = data.get("output", {})

            # Determine stage based on chain name
            if "query" in name.lower() and "rewrite" in name.lower():
                return StreamingChunk(
                    stage="query_rewrite",
                    content=output.get("rewritten_query", ""),
                    metadata={"original": output.get("original_query", "")}
                )
            elif "faq" in name.lower():
                return StreamingChunk(
                    stage="faq_match",
                    content=output.get("faq_answer", ""),
                    metadata={"matched": output.get("matched", False)}
                )
            elif "retriev" in name.lower():
                docs = output.get("documents", [])
                return StreamingChunk(
                    stage="retrieval",
                    content=f"检索到 {len(docs)} 个文档",
                    metadata={"count": len(docs), "documents": docs[:3]}
                )
            elif "rerank" in name.lower():
                docs = output.get("documents", [])
                return StreamingChunk(
                    stage="rerank",
                    content=f"重排序 {len(docs)} 个文档",
                    metadata={"count": len(docs)}
                )
            elif "llm" in name.lower() or "generat" in name.lower():
                return StreamingChunk(
                    stage="generation",
                    content=output.get("content", ""),
                    metadata={"tokens": output.get("token_usage", {})}
                )

        elif event_type == "on_llm_stream":
            # Token-by-token generation
            token = data.get("chunk", {}).get("text", "")
            if token:
                return StreamingChunk(
                    stage="generation",
                    content=token,
                    metadata={"streaming": True}
                )

        elif event_type == "on_chain_stream":
            # Intermediate chain output
            return StreamingChunk(
                stage="processing",
                content=str(data.get("chunk", "")),
                metadata={"event": event_type}
            )

        return None

    def _parse_sse_data(self, data: dict[str, Any]) -> Optional[StreamingChunk]:
        """Parse SSE data from HTTP streaming."""
        stage = data.get("stage", "unknown")
        content = data.get("content", "")
        is_final = data.get("is_final", False)

        return StreamingChunk(
            stage=stage,
            content=content,
            is_final=is_final,
            metadata=data.get("metadata", {})
        )

    async def _yield_response_chunks(
        self,
        response: RAGResponse
    ) -> AsyncGenerator[StreamingChunk, None]:
        """Yield chunks from a complete RAGResponse."""
        import asyncio

        # Query rewrite stage
        if response.query_rewrite:
            yield StreamingChunk(
                stage="query_rewrite",
                content=f"原查询: {response.query_rewrite.original_query}\n改写后: {response.query_rewrite.rewritten_query}",
                metadata={
                    "rewrite_type": response.query_rewrite.rewrite_type,
                    "confidence": response.query_rewrite.confidence
                }
            )
            await asyncio.sleep(0.1)

        # FAQ match stage
        if response.faq_match:
            yield StreamingChunk(
                stage="faq_match",
                content=response.faq_match.faq_answer if response.faq_match.matched else "未匹配到FAQ",
                metadata={
                    "matched": response.faq_match.matched,
                    "confidence": response.faq_match.confidence,
                    "similarity": response.faq_match.similarity_score
                }
            )
            await asyncio.sleep(0.1)

        # Retrieval stage
        if response.retrieval_results:
            docs_info = "\n".join([
                f"{i+1}. {doc.content[:100]}... (得分: {doc.score:.3f})"
                for i, doc in enumerate(response.retrieval_results[:5])
            ])
            yield StreamingChunk(
                stage="retrieval",
                content=f"检索到 {len(response.retrieval_results)} 个文档:\n{docs_info}",
                metadata={"count": len(response.retrieval_results)}
            )
            await asyncio.sleep(0.1)

        # Rerank stage
        if response.rerank_results:
            rerank_info = "\n".join([
                f"{i+1}. {doc.content[:100]}... (重排得分: {doc.rerank_score:.3f})"
                for i, doc in enumerate(response.rerank_results[:5])
            ])
            yield StreamingChunk(
                stage="rerank",
                content=f"重排序后 {len(response.rerank_results)} 个文档:\n{rerank_info}",
                metadata={"count": len(response.rerank_results)}
            )
            await asyncio.sleep(0.1)

        # Generation stage (with thinking if available)
        if response.llm_output:
            if response.llm_output.thinking_process:
                yield StreamingChunk(
                    stage="thinking",
                    content=response.llm_output.thinking_process,
                    metadata={"tokens": response.llm_output.token_usage}
                )
                await asyncio.sleep(0.1)

        # Final answer
        yield StreamingChunk(
            stage="final",
            content=response.final_answer,
            is_final=True,
            metadata={
                "success": response.success,
                "latency_ms": response.latency_ms,
                "is_refused": response.is_refused
            }
        )