"""
LangGraph RemoteGraph adapter for RAG services.
"""

import asyncio
import time
from typing import Any, Optional

from ..models.rag_response import RAGResponse, RAGResponseAdapter
from ..models.annotation import Annotation
from ..core.exceptions import RAGConnectionError
from ..core.logging import logger

from .base_adapter import RAGAdapter, RAGAdapterConfig


class LangGraphAdapter(RAGAdapter):
    """
    Adapter for LangGraph RemoteGraph RAG services.
    Implements async calls with retry logic.
    """

    def __init__(self, config: RAGAdapterConfig):
        self.config = config
        self._client = None
        self._initialized = False

    @property
    def name(self) -> str:
        return "langgraph"

    async def initialize(self) -> None:
        """Initialize the LangGraph client."""
        try:
            # Import here to allow module to load without langgraph
            from langgraph.pregel.remote import RemoteGraph

            self._client = RemoteGraph(self.config.service_url)
            self._initialized = True
            logger.info(f"Initialized LangGraph adapter: {self.config.service_url}")
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
                response = await self._query_internal(
                    query=query,
                    conversation_history=conversation_history or [],
                    agent_id=agent_id or "default",
                    enable_thinking=enable_thinking,
                    **kwargs,
                )

                latency_ms = (time.time() - start_time) * 1000
                response.latency_ms = latency_ms

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
    ) -> RAGResponse:
        """Internal query implementation."""
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
        return RAGResponseAdapter.from_langgraph(result, query)

    async def _query_via_http(
        self,
        query: str,
        conversation_history: list[str],
        agent_id: str,
        enable_thinking: bool,
        **kwargs: Any,
    ) -> RAGResponse:
        """Fallback HTTP query when LangGraph client not available."""
        import aiohttp

        url = f"{self.config.service_url}/invoke"
        payload = {
            "query": query,
            "conversation_history": conversation_history,
            "agent_id": agent_id,
            "enable_thinking": enable_thinking,
            **kwargs,
        }

        async with aiohttp.ClientSession() as session:
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
                return RAGResponseAdapter.from_langgraph(data, query)

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
                async with aiohttp.ClientSession() as session:
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
        self._client = None
        self._initialized = False