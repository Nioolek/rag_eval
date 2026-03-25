"""
RAG Pipeline Graph Nodes.

All 14 node implementations for the LangGraph StateGraph.
"""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Any, Optional

from rag_rag.graph.state import RAGState, create_initial_state
from rag_rag.core.logging import get_logger, log_timing

logger = get_logger("rag_rag.graph.nodes")


# === Decorator for timing and error handling ===

def node_decorator(stage_name: str):
    """Decorator for timing and error handling."""
    def decorator(func):
        async def wrapper(state: RAGState) -> dict[str, Any]:
            start_time = time.time()
            try:
                result = await func(state)
                duration_ms = (time.time() - start_time) * 1000
                log_timing(stage_name, duration_ms)

                # Add timing to result
                if result is None:
                    result = {}
                if "stage_timing" not in result:
                    result["stage_timing"] = {}
                result["stage_timing"][f"{stage_name}_ms"] = duration_ms

                return result

            except Exception as e:
                logger.error(f"Node [{stage_name}] failed: {e}")
                return {
                    "errors": [{
                        "stage": stage_name,
                        "type": type(e).__name__,
                        "message": str(e),
                        "timestamp": datetime.utcnow().isoformat(),
                    }],
                    "stage_timing": {f"{stage_name}_ms": (time.time() - start_time) * 1000},
                }
        return wrapper
    return decorator


# === Node Implementations ===

@node_decorator("input")
async def input_node(state: RAGState) -> dict[str, Any]:
    """
    Input validation and initialization.

    - Validate query
    - Generate conversation_id if missing
    - Load conversation history if exists
    """
    query = state.get("query", "")

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    # Generate conversation_id if missing
    conversation_id = state.get("conversation_id") or str(uuid.uuid4())

    # Initialize metadata
    metadata = state.get("metadata", {})
    metadata["start_time"] = datetime.utcnow().isoformat()

    return {
        "conversation_id": conversation_id,
        "metadata": metadata,
    }


@node_decorator("faq_match")
async def faq_match_node(state: RAGState) -> dict[str, Any]:
    """
    FAQ matching node.

    - Check exact match
    - Check semantic match if enabled
    - Return FAQ result if matched
    """
    from rag_rag.core.config import get_config
    from rag_rag.storage.faq_store import FAQStore
    from pathlib import Path

    query = state["query"]
    config = get_config()

    # Initialize FAQ store with the data directory
    # Use higher threshold (0.85) for more accurate matches
    data_dir = Path("data")
    faq_store = FAQStore(
        db_path=data_dir / "faq.db",
        match_threshold=0.85,  # Higher threshold for precision
    )
    await faq_store.initialize()

    try:
        results = []

        # Strategy 1: Try exact match with full query first
        results = await faq_store.search(query, top_k=3, search_type="exact")

        # Filter results: require at least 60% query word coverage
        if results:
            query_words = set(query.lower().split())
            filtered_results = []
            for r in results:
                question_words = set(r.get("question", "").lower().split())
                common_words = query_words & question_words
                query_coverage = len(common_words) / len(query_words) if query_words else 0

                # Require meaningful overlap
                if query_coverage >= 0.6:
                    filtered_results.append(r)

            results = filtered_results

        # Strategy 2: Try semantic search if available (requires embedding service)
        # Note: For now, we don't have embedding service, so this falls back to exact
        if not results:
            results = await faq_store.search(query, top_k=3, search_type="hybrid")
            # Apply same filtering
            if results:
                query_words = set(query.lower().split())
                filtered_results = []
                for r in results:
                    question_words = set(r.get("question", "").lower().split())
                    common_words = query_words & question_words
                    query_coverage = len(common_words) / len(query_words) if query_words else 0
                    if query_coverage >= 0.6:
                        filtered_results.append(r)
                results = filtered_results

        faq_matched = False
        faq_result = None

        # Only match if we have high-confidence results
        if results:
            top_result = results[0]
            score = top_result.get("score", 0)

            # Require score >= 0.85 for a match
            if score >= 0.85:
                faq_matched = True
                faq_result = {
                    "matched": True,
                    "faq_id": top_result.get("id", ""),
                    "question": top_result.get("question", ""),
                    "answer": top_result.get("answer", ""),
                    "confidence": score,
                    "similarity": score,
                    "match_type": top_result.get("match_type", "exact"),
                    "timing_ms": 0,
                }

        return {
            "faq_matched": faq_matched,
            "faq_result": faq_result,
        }
    finally:
        await faq_store.close()


@node_decorator("query_rewrite")
async def query_rewrite_node(state: RAGState) -> dict[str, Any]:
    """
    Query rewriting node.

    - Expand query with keywords
    - Clarify ambiguous queries
    - Incorporate multi-turn context
    """
    from rag_rag.services.llm_service import LLMService, LLMConfig
    from rag_rag.core.config import get_config, get_env_config

    query = state["query"]
    conversation_history = state.get("conversation_history", [])

    # Determine rewrite type
    if conversation_history:
        rewrite_type = "multi_turn"
    elif len(query.split()) < 3:
        rewrite_type = "clarification"
    else:
        rewrite_type = "expansion"

    # Get LLM service (would be injected in production)
    # In production: llm_service = get_llm_service()
    # result = await llm_service.rewrite_query(query, conversation_history, rewrite_type)

    # Placeholder - return original query
    query_rewrite = {
        "original_query": query,
        "rewritten_query": query,  # Would be rewritten by LLM
        "rewrite_type": rewrite_type,
        "confidence": 0.8,
        "timing_ms": 0,
    }

    return {
        "query_rewrite": query_rewrite,
        "rewritten_query": query,  # For downstream use
    }


@node_decorator("vector_retrieve")
async def vector_retrieve_node(state: RAGState) -> dict[str, Any]:
    """
    Vector retrieval node.

    - Embed query using DashScope embedding service
    - Search Chroma for similar documents
    """
    from rag_rag.core.config import get_config
    from rag_rag.storage.vector_store import VectorStore
    from rag_rag.services.embedding_service import EmbeddingService, EmbeddingConfig
    from pathlib import Path
    import os

    query = state.get("rewritten_query") or state["query"]
    config = get_config()

    # Initialize vector store
    data_dir = Path("data")
    vector_store = VectorStore(
        persist_dir=data_dir / "chroma",
        collection_name="electronics_docs",
        embedding_dimension=1024,
    )
    await vector_store.initialize()

    try:
        # Check if we have a valid embedding API key
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if api_key and api_key != "sk-placeholder-key":
            # Use real embedding service
            embedding_service = EmbeddingService(EmbeddingConfig(
                api_key=api_key,
                model="text-embedding-v3",
                dimension=1024,
            ))
            await embedding_service.initialize()

            # Get query embedding
            query_embedding = await embedding_service.embed_single(query)

            # Search vector store
            top_k = config.retrieval.vector_top_k if hasattr(config, 'retrieval') else 20
            results = await vector_store.search(query_embedding, top_k=top_k)

            # Format results
            vector_results = []
            for result in results:
                vector_results.append({
                    "document_id": result.get("document_id", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0),
                    "source": "vector",
                    "metadata": result.get("metadata", {}),
                })

            await embedding_service.close()
        else:
            # No valid API key - return empty results
            logger.warning("No valid DASHSCOPE_API_KEY - vector retrieval disabled")
            vector_results = []

        return {
            "vector_results": vector_results,
        }
    finally:
        await vector_store.close()


@node_decorator("fulltext_retrieve")
async def fulltext_retrieve_node(state: RAGState) -> dict[str, Any]:
    """
    Fulltext retrieval node.

    - BM25 search with Whoosh
    """
    from rag_rag.core.config import get_config
    from rag_rag.storage.fulltext_store import FulltextStore
    from pathlib import Path

    query = state.get("rewritten_query") or state["query"]
    config = get_config()

    # Initialize fulltext store
    data_dir = Path("data")
    fulltext_store = FulltextStore(
        index_dir=data_dir / "whoosh",
        index_name="electronics_docs",
    )
    await fulltext_store.initialize()

    try:
        # Get top_k from config or use default
        top_k = config.retrieval.fulltext_top_k if hasattr(config, 'retrieval') and hasattr(config.retrieval, 'fulltext_top_k') else 10

        # Search fulltext store
        results = await fulltext_store.search(query, top_k=top_k)

        # Format results for RAGState
        fulltext_results = []
        for result in results:
            fulltext_results.append({
                "document_id": result.get("document_id", ""),
                "content": result.get("content", ""),
                "score": result.get("score", 0),
                "source": "fulltext",
                "metadata": result.get("metadata", {}),
            })

        return {
            "fulltext_results": fulltext_results,
        }
    finally:
        await fulltext_store.close()


@node_decorator("graph_retrieve")
async def graph_retrieve_node(state: RAGState) -> dict[str, Any]:
    """
    Knowledge graph retrieval node.

    - Search Neo4j for related entities
    """
    from rag_rag.core.config import get_config

    query = state.get("rewritten_query") or state["query"]
    config = get_config()

    # Get graph store (would be injected in production)
    # In production: graph_store = get_graph_store()
    # results = await graph_store.search(query, top_k=config.retrieval.graph_top_k)

    # Placeholder - return empty results
    graph_results = []

    return {
        "graph_results": graph_results,
    }


@node_decorator("merge")
async def merge_node(state: RAGState) -> dict[str, Any]:
    """
    Merge retrieval results from multiple sources.

    - Normalize scores
    - Deduplicate by document_id
    - Apply weighted fusion
    """
    from rag_rag.core.config import get_config

    vector_results = state.get("vector_results", [])
    fulltext_results = state.get("fulltext_results", [])
    graph_results = state.get("graph_results", [])

    config = get_config()
    weights = {
        "vector": config.retrieval.vector_weight,
        "fulltext": config.retrieval.fulltext_weight,
        "graph": config.retrieval.graph_weight,
    }

    # Merge results with weighted fusion
    merged = {}

    def normalize_score(score: float, source: str) -> float:
        """Normalize score to 0-1 range."""
        if source == "fulltext":
            return min(score / 10.0, 1.0)
        return score

    # Process vector results
    for result in vector_results:
        doc_id = result.get("document_id", "")
        score = normalize_score(result.get("score", 0), "vector")
        merged[doc_id] = {
            **result,
            "vector_score": score,
            "combined_score": score * weights["vector"],
        }

    # Process fulltext results
    for result in fulltext_results:
        doc_id = result.get("document_id", "")
        score = normalize_score(result.get("score", 0), "fulltext")
        if doc_id in merged:
            merged[doc_id]["fulltext_score"] = score
            merged[doc_id]["combined_score"] += score * weights["fulltext"]
        else:
            merged[doc_id] = {
                **result,
                "fulltext_score": score,
                "combined_score": score * weights["fulltext"],
            }

    # Process graph results
    for result in graph_results:
        doc_id = result.get("document_id", "")
        score = normalize_score(result.get("score", 0), "graph")
        if doc_id in merged:
            merged[doc_id]["graph_score"] = score
            merged[doc_id]["combined_score"] += score * weights["graph"]
        else:
            merged[doc_id] = {
                **result,
                "graph_score": score,
                "combined_score": score * weights["graph"],
            }

    # Sort by combined score
    merged_results = sorted(
        merged.values(),
        key=lambda x: x.get("combined_score", 0),
        reverse=True,
    )

    return {
        "merged_results": merged_results,
    }


@node_decorator("rerank")
async def rerank_node(state: RAGState) -> dict[str, Any]:
    """
    Rerank merged results.

    - Use Alibaba gte-rerank
    - Fallback to local BM25
    """
    from rag_rag.core.config import get_config

    query = state.get("rewritten_query") or state["query"]
    merged_results = state.get("merged_results", [])

    if not merged_results:
        return {
            "reranked_results": [],
            "rerank_scores": [],
        }

    config = get_config()

    # Get rerank service (would be injected in production)
    # In production: rerank_service = get_rerank_service()
    # documents = [r.get("content", "") for r in merged_results]
    # rerank_results = await rerank_service.rerank(query, documents, config.rerank.top_k)

    # Placeholder - return top k from merged
    top_k = config.rerank.top_k
    reranked = merged_results[:top_k]

    reranked_results = []
    rerank_scores = []

    for i, result in enumerate(reranked):
        reranked_results.append({
            **result,
            "rerank_score": result.get("combined_score", 0),
            "original_score": result.get("score", 0),
            "rank": i + 1,
        })
        rerank_scores.append(result.get("combined_score", 0))

    return {
        "reranked_results": reranked_results,
        "rerank_scores": rerank_scores,
    }


@node_decorator("build_prompt")
async def build_prompt_node(state: RAGState) -> dict[str, Any]:
    """
    Build prompt from retrieved context.

    - Format context from reranked results
    - Apply prompt template
    """
    from rag_rag.core.config import get_config

    query = state["query"]
    reranked_results = state.get("reranked_results", [])
    enable_thinking = state.get("enable_thinking", False)

    config = get_config()

    # Build context
    context_parts = []
    for i, result in enumerate(reranked_results[:5]):
        content = result.get("content", "")
        if content:
            context_parts.append(f"[{i+1}] {content}")

    context_prompt = "\n\n".join(context_parts) if context_parts else "无相关上下文"

    # Build system prompt
    if enable_thinking:
        system_prompt = """你是一个专业的企业知识库助手，擅长深度思考和分析。

## 思考模式
在回答之前，请先进行深入思考：
1. 分析问题的核心意图
2. 评估上下文的相关性和可靠性
3. 构建回答的逻辑框架

## 输出格式
<thinking>
[你的思考过程]
</thinking>
[你的最终回答]"""
        template_name = "thinking"
    else:
        system_prompt = """你是一个专业的企业知识库助手。
你的职责是基于提供的知识库内容，准确、专业地回答用户问题。

## 回答原则
1. 只使用提供的上下文信息回答问题
2. 如果上下文不足以回答，请诚实说明
3. 回答要简洁、准确、有条理"""
        template_name = "default"

    # Build final prompt
    final_prompt = f"""## 相关上下文
{context_prompt}

## 用户问题
{query}

请基于上下文回答用户问题。"""

    return {
        "system_prompt": system_prompt,
        "context_prompt": context_prompt,
        "final_prompt": final_prompt,
        "prompt_template_name": template_name,
    }


@node_decorator("refusal_check")
async def refusal_check_node(state: RAGState) -> dict[str, Any]:
    """
    Check if should refuse to answer.

    - Out of domain
    - Sensitive content
    - Low relevance
    """
    from rag_rag.core.config import get_config

    query = state["query"]
    context = state.get("context_prompt", "")
    reranked_results = state.get("reranked_results", [])

    config = get_config()

    # Check for sensitive content
    # In production: use sensitive_filter.check(query)

    # Check relevance
    if not reranked_results:
        return {
            "should_refuse": True,
            "refusal_reason": "知识库中没有找到相关信息",
            "refusal_type": "out_of_domain",
        }

    # Check top relevance score
    top_score = reranked_results[0].get("rerank_score", 0) if reranked_results else 0
    if top_score < config.refusal.low_relevance_threshold:
        return {
            "should_refuse": True,
            "refusal_reason": "检索到的信息与问题相关性过低",
            "refusal_type": "low_relevance",
        }

    # LLM-based refusal check (optional)
    # In production: llm_service.check_refusal(query, context)

    return {
        "should_refuse": False,
        "refusal_reason": "",
        "refusal_type": "",
    }


@node_decorator("generate")
async def generate_node(state: RAGState) -> dict[str, Any]:
    """
    Generate answer using LLM.

    - Stream generation
    - Extract thinking process if enabled
    """
    from rag_rag.services.llm_service import LLMService, LLMConfig
    from rag_rag.core.config import get_env_config
    import os

    system_prompt = state.get("system_prompt", "")
    final_prompt = state.get("final_prompt", "")
    enable_thinking = state.get("enable_thinking", False)

    # Get API configuration from environment
    # Prefer OPENAI_API_KEY, fallback to DASHSCOPE_API_KEY for backward compatibility
    env_config = get_env_config()
    api_key = env_config.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = env_config.openai_base_url or os.environ.get("OPENAI_BASE_URL", "https://coding.dashscope.aliyuncs.com/v1")
    model = env_config.llm_model or os.environ.get("LLM_MODEL", "qwen-plus")

    # Fallback to legacy DASHSCOPE_API_KEY
    if not api_key:
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")

    if api_key and api_key != "sk-placeholder-key" and not api_key.startswith("sk-xxx"):
        # Use real LLM service with OpenAI-compatible API
        llm_service = LLMService(LLMConfig(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=0.7,
            max_tokens=2048,
        ))

        try:
            output = await llm_service.generate(
                system_prompt=system_prompt,
                user_prompt=final_prompt,
                enable_thinking=enable_thinking,
            )

            llm_output = {
                "content": output.content,
                "thinking_process": output.thinking_process,
                "token_usage": output.token_usage,
                "model": output.model,
                "finish_reason": output.finish_reason,
            }

            final_answer = output.content

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            llm_output = {
                "content": f"抱歉，生成回答时发生错误: {str(e)}",
                "thinking_process": "",
                "token_usage": {},
                "model": "error",
                "finish_reason": "error",
            }
            final_answer = llm_output["content"]

        finally:
            await llm_service.close()
    else:
        # No valid API key - return placeholder
        logger.warning("No valid API key configured - LLM generation disabled")
        llm_output = {
            "content": "这是一个示例回答。在生产环境中，这将由LLM生成。",
            "thinking_process": "",
            "token_usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            "model": "placeholder",
            "finish_reason": "placeholder",
        }
        final_answer = llm_output.get("content", "")

    return {
        "llm_output": llm_output,
        "thinking_process": llm_output.get("thinking_process", ""),
        "final_answer": final_answer,
        "is_refused": False,
    }


@node_decorator("output")
async def output_node(state: RAGState) -> dict[str, Any]:
    """
    Format final output.

    - Build response in RAGResponse-compatible format
    - Save conversation history
    - Calculate total timing
    """
    from rag_rag.graph.state import state_to_rag_response_format

    # Calculate total timing
    stage_timing = state.get("stage_timing", {})
    total_ms = sum(v for k, v in stage_timing.items() if k.endswith("_ms") and k != "total_ms")

    # Build output in RAGResponse format
    answer = state.get("final_answer", "")
    is_refused = state.get("is_refused", False)

    # Build retrieval format for RAGResponseAdapter
    retrieval = []
    for result in state.get("reranked_results", []) or state.get("merged_results", []):
        retrieval.append({
            "id": result.get("document_id", ""),
            "content": result.get("content", ""),
            "score": result.get("rerank_score") or result.get("combined_score", 0),
            "metadata": result.get("metadata", {}),
        })

    # Build rerank format
    rerank = []
    for result in state.get("reranked_results", []):
        rerank.append({
            "id": result.get("document_id", ""),
            "content": result.get("content", ""),
            "original_score": result.get("original_score", 0),
            "rerank_score": result.get("rerank_score", 0),
        })

    # Update metadata
    metadata = state.get("metadata", {})
    metadata["end_time"] = datetime.utcnow().isoformat()

    return {
        "answer": answer,
        "retrieval": retrieval,
        "rerank": rerank,
        "stage_timing": {**stage_timing, "total_ms": total_ms},
        "metadata": metadata,
    }


@node_decorator("answer_faq")
async def answer_faq_node(state: RAGState) -> dict[str, Any]:
    """
    Answer directly from FAQ match.

    Used when FAQ is matched to skip retrieval pipeline.
    """
    faq_result = state.get("faq_result", {})

    answer = faq_result.get("answer", "")
    question = faq_result.get("question", "")

    llm_output = {
        "content": answer,
        "thinking_process": "",
        "token_usage": {},
        "model": "faq",
        "finish_reason": "faq_match",
    }

    return {
        "final_answer": answer,
        "llm_output": llm_output,
        "is_refused": False,
    }


@node_decorator("refuse")
async def refuse_node(state: RAGState) -> dict[str, Any]:
    """
    Generate refusal response.
    """
    refusal_type = state.get("refusal_type", "")
    refusal_reason = state.get("refusal_reason", "")

    # Template responses
    templates = {
        "out_of_domain": "抱歉，这个问题超出了我的知识范围，我无法回答。",
        "sensitive": "抱歉，这个问题涉及敏感内容，我无法回答。",
        "low_relevance": "抱歉，我在知识库中没有找到与您问题相关的信息，无法给出准确回答。",
    }

    answer = templates.get(refusal_type, "抱歉，我无法回答这个问题。")

    llm_output = {
        "content": answer,
        "thinking_process": "",
        "token_usage": {},
        "model": "refusal",
        "finish_reason": "refused",
    }

    return {
        "final_answer": answer,
        "llm_output": llm_output,
        "is_refused": True,
    }