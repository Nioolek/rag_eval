"""
Semantic similarity functions using embeddings.
"""

import asyncio
from typing import Optional

import numpy as np

from .embedding_provider import BaseEmbeddingProvider, get_embedding_provider
from ...core.logging import logger


async def compute_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score (-1 to 1)
    """
    a = np.array(vec1)
    b = np.array(vec2)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


async def compute_semantic_similarity(
    text1: str,
    text2: str,
    provider: Optional[BaseEmbeddingProvider] = None,
) -> float:
    """
    Compute semantic similarity between two texts using embeddings.

    Args:
        text1: First text
        text2: Second text
        provider: Optional embedding provider (uses global if not provided)

    Returns:
        Similarity score (0 to 1)
    """
    if not text1 or not text2:
        return 0.0

    if provider is None:
        provider = await get_embedding_provider()

    # Get embeddings concurrently
    emb1, emb2 = await asyncio.gather(
        provider.embed(text1),
        provider.embed(text2),
    )

    # Compute cosine similarity
    similarity = await compute_cosine_similarity(emb1, emb2)

    # Normalize to 0-1 range
    return (similarity + 1) / 2


async def compute_semantic_similarity_batch(
    text: str,
    candidates: list[str],
    provider: Optional[BaseEmbeddingProvider] = None,
) -> list[float]:
    """
    Compute semantic similarity between a text and multiple candidates.

    Args:
        text: Reference text
        candidates: List of candidate texts
        provider: Optional embedding provider

    Returns:
        List of similarity scores
    """
    if not text or not candidates:
        return [0.0] * len(candidates)

    if provider is None:
        provider = await get_embedding_provider()

    # Get embedding for reference text
    ref_embedding = await provider.embed(text)

    # Get embeddings for all candidates
    candidate_embeddings = await provider.embed_batch(candidates)

    # Compute similarities
    similarities = []
    for cand_emb in candidate_embeddings:
        sim = await compute_cosine_similarity(ref_embedding, cand_emb)
        similarities.append((sim + 1) / 2)  # Normalize to 0-1

    return similarities


async def find_most_similar(
    text: str,
    candidates: list[str],
    provider: Optional[BaseEmbeddingProvider] = None,
    top_k: int = 1,
) -> list[tuple[int, float]]:
    """
    Find the most similar candidates to a reference text.

    Args:
        text: Reference text
        candidates: List of candidate texts
        provider: Optional embedding provider
        top_k: Number of top candidates to return

    Returns:
        List of (index, similarity) tuples for top-k candidates
    """
    similarities = await compute_semantic_similarity_batch(text, candidates, provider)

    # Sort by similarity (descending)
    indexed = [(i, sim) for i, sim in enumerate(similarities)]
    indexed.sort(key=lambda x: x[1], reverse=True)

    return indexed[:top_k]


async def compute_semantic_coverage(
    generated: str,
    reference: str,
    provider: Optional[BaseEmbeddingProvider] = None,
    sentence_level: bool = True,
) -> dict:
    """
    Compute semantic coverage of reference by generated text.

    Measures how well the generated text covers the semantic content
    of the reference text.

    Args:
        generated: Generated text
        reference: Reference text
        provider: Optional embedding provider
        sentence_level: Whether to compute at sentence level

    Returns:
        Dictionary with coverage metrics
    """
    if not generated or not reference:
        return {
            "overall_similarity": 0.0,
            "coverage_score": 0.0,
            "details": {},
        }

    # Overall similarity
    overall_sim = await compute_semantic_similarity(generated, reference, provider)

    if sentence_level:
        # Split into sentences
        import re

        def split_sentences(text):
            # Simple sentence splitting
            sentences = re.split(r'[.!?。！？]\s*', text)
            return [s.strip() for s in sentences if s.strip()]

        gen_sentences = split_sentences(generated)
        ref_sentences = split_sentences(reference)

        if not gen_sentences or not ref_sentences:
            return {
                "overall_similarity": overall_sim,
                "coverage_score": overall_sim,
                "details": {"sentence_count": 0},
            }

        # For each reference sentence, find best matching generated sentence
        coverage_scores = []
        for ref_sent in ref_sentences:
            if not ref_sent:
                continue
            sims = await compute_semantic_similarity_batch(ref_sent, gen_sentences, provider)
            best_sim = max(sims) if sims else 0.0
            coverage_scores.append(best_sim)

        # Average coverage
        avg_coverage = sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0.0

        return {
            "overall_similarity": overall_sim,
            "coverage_score": avg_coverage,
            "details": {
                "gen_sentence_count": len(gen_sentences),
                "ref_sentence_count": len(ref_sentences),
                "per_sentence_coverage": coverage_scores,
            },
        }

    return {
        "overall_similarity": overall_sim,
        "coverage_score": overall_sim,
        "details": {},
    }