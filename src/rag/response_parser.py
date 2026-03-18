"""
RAG Response parser for different formats.
"""

import json
from typing import Any, Optional

from ..models.rag_response import (
    RAGResponse,
    QueryRewrite,
    FAQMatch,
    RetrievalResult,
    RerankResult,
    LLMOutput,
)
from ..core.exceptions import RAGConnectionError
from ..core.logging import logger


class RAGResponseParser:
    """
    Parser for different RAG response formats.
    Handles conversion to standard RAGResponse model.
    """

    @staticmethod
    def parse(data: dict[str, Any], query: str) -> RAGResponse:
        """
        Parse RAG response from dict.
        Auto-detects format and converts to standard model.

        Args:
            data: Raw response data
            query: Original query

        Returns:
            Standardized RAGResponse
        """
        # Check for known formats
        # LangGraph format has specific keys
        langgraph_keys = {"query_rewrite", "faq", "retrieval", "rerank", "generation"}
        if "graph_response" in data or (langgraph_keys & data.keys()):
            return RAGResponseParser._parse_langgraph_format(data, query)
        elif "output" in data and "intermediate_steps" in data:
            return RAGResponseParser._parse_langchain_format(data, query)
        elif "answer" in data or "response" in data:
            return RAGResponseParser._parse_simple_format(data, query)
        else:
            # Try generic parsing
            return RAGResponseParser._parse_generic(data, query)

    @staticmethod
    def _parse_langgraph_format(
        data: dict[str, Any],
        query: str
    ) -> RAGResponse:
        """Parse LangGraph format response."""
        response = RAGResponse(query=query)

        graph_data = data.get("graph_response", data)

        # Parse from different node outputs
        if "query_rewrite" in graph_data:
            qr = graph_data["query_rewrite"]
            response.query_rewrite = QueryRewrite(
                original_query=query,
                rewritten_query=qr.get("rewritten", qr.get("query", query)),
                rewrite_type=qr.get("type", ""),
                confidence=qr.get("confidence", 1.0),
            )

        if "faq" in graph_data:
            faq = graph_data["faq"]
            response.faq_match = FAQMatch(
                matched=faq.get("matched", False),
                faq_id=faq.get("id", ""),
                faq_question=faq.get("question", ""),
                faq_answer=faq.get("answer", ""),
                confidence=faq.get("confidence", 0.0),
            )

        if "retrieval" in graph_data:
            for i, doc in enumerate(graph_data["retrieval"]):
                response.retrieval_results.append(RetrievalResult(
                    document_id=doc.get("id", str(i)),
                    content=doc.get("content", doc.get("text", "")),
                    score=doc.get("score", 0.0),
                    rank=i + 1,
                    metadata=doc.get("metadata", {}),
                ))

        if "rerank" in graph_data:
            for i, doc in enumerate(graph_data["rerank"]):
                response.rerank_results.append(RerankResult(
                    document_id=doc.get("id", str(i)),
                    content=doc.get("content", ""),
                    original_score=doc.get("original_score", 0.0),
                    rerank_score=doc.get("score", 0.0),
                    rank=i + 1,
                ))

        if "generation" in graph_data:
            gen = graph_data["generation"]
            response.final_answer = gen.get("content", gen.get("text", ""))
            response.llm_output = LLMOutput(
                content=response.final_answer,
                thinking_process=gen.get("thinking", ""),
                token_usage=gen.get("usage", {}),
                model=gen.get("model", ""),
            )

        response.success = True
        return response

    @staticmethod
    def _parse_langchain_format(
        data: dict[str, Any],
        query: str
    ) -> RAGResponse:
        """Parse LangChain format response."""
        response = RAGResponse(query=query)

        output = data.get("output", "")
        intermediate_steps = data.get("intermediate_steps", [])

        # Parse intermediate steps
        for step in intermediate_steps:
            if isinstance(step, tuple) and len(step) == 2:
                action, observation = step
                if hasattr(action, "tool"):
                    if action.tool == "retriever":
                        # Parse retrieval results
                        docs = observation if isinstance(observation, list) else [observation]
                        for i, doc in enumerate(docs):
                            content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                            metadata = doc.metadata if hasattr(doc, "metadata") else {}
                            response.retrieval_results.append(RetrievalResult(
                                document_id=metadata.get("id", str(i)),
                                content=content,
                                score=metadata.get("score", 0.0),
                                rank=i + 1,
                                metadata=metadata,
                            ))

        response.final_answer = str(output)
        response.success = True
        return response

    @staticmethod
    def _parse_simple_format(
        data: dict[str, Any],
        query: str
    ) -> RAGResponse:
        """Parse simple format response."""
        response = RAGResponse(query=query)

        # Get answer
        response.final_answer = data.get("answer", data.get("response", ""))

        # Get contexts if available
        contexts = data.get("contexts", data.get("documents", []))
        for i, ctx in enumerate(contexts):
            if isinstance(ctx, str):
                response.retrieval_results.append(RetrievalResult(
                    document_id=str(i),
                    content=ctx,
                    rank=i + 1,
                ))
            elif isinstance(ctx, dict):
                response.retrieval_results.append(RetrievalResult(
                    document_id=ctx.get("id", str(i)),
                    content=ctx.get("content", ctx.get("text", "")),
                    score=ctx.get("score", 0.0),
                    rank=i + 1,
                    metadata=ctx.get("metadata", {}),
                ))

        response.success = True
        return response

    @staticmethod
    def _parse_generic(
        data: dict[str, Any],
        query: str
    ) -> RAGResponse:
        """Generic fallback parser."""
        response = RAGResponse(query=query)

        # Try to extract common fields
        for key in ["answer", "response", "output", "result"]:
            if key in data and isinstance(data[key], str):
                response.final_answer = data[key]
                break

        for key in ["contexts", "documents", "chunks", "retrieval"]:
            if key in data and isinstance(data[key], list):
                for i, item in enumerate(data[key]):
                    if isinstance(item, str):
                        response.retrieval_results.append(RetrievalResult(
                            document_id=str(i),
                            content=item,
                            rank=i + 1,
                        ))
                    elif isinstance(item, dict):
                        response.retrieval_results.append(RetrievalResult(
                            document_id=item.get("id", str(i)),
                            content=item.get("content", item.get("text", str(item))),
                            score=item.get("score", 0.0),
                            rank=i + 1,
                        ))
                break

        response.success = True
        return response

    @staticmethod
    def parse_json(json_str: str, query: str) -> RAGResponse:
        """Parse RAG response from JSON string."""
        try:
            data = json.loads(json_str)
            return RAGResponseParser.parse(data, query)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return RAGResponse(
                query=query,
                success=False,
                error_message=f"JSON parse error: {e}",
            )