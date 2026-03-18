"""
Generation quality metrics for RAG evaluation.
Metrics: Factual Consistency, Answer Relevance, Completeness, Fluency, Refusal Accuracy, Hallucination Detection
"""

import re
from typing import Any, Optional

from .base import BaseMetric, MetricContext
from ...models.metric_result import MetricCategory, MetricResult
from ...core.logging import logger


class FactualConsistencyMetric(BaseMetric):
    """
    答案事实一致性（无幻觉）
    检查答案是否基于检索到的文档，不包含虚假信息。

    Uses entailment-style checking between answer and retrieved context.
    """

    name = "factual_consistency"
    category = MetricCategory.GENERATION
    description = "答案事实一致性：检查答案是否基于检索文档，无幻觉"
    requires_llm = True
    threshold = 0.7

    async def calculate(self, context: MetricContext) -> MetricResult:
        answer = context.answer
        retrieved = context.retrieved_documents

        if not answer:
            return self._create_result(
                score=0.0,
                details={"reason": "No answer provided"}
            )

        if not retrieved:
            # Without context, we can't verify factual consistency
            return self._create_result(
                score=0.5,  # Neutral score
                details={"reason": "No retrieved context to verify against"}
            )

        # Combine retrieved documents as context
        context_text = " ".join(retrieved)

        # Use LLM if available
        if context.llm_client:
            return await self._calculate_with_llm(context, answer, context_text)

        # Fallback: Simple overlap-based consistency check
        return self._calculate_simple(answer, context_text)

    async def _calculate_with_llm(
        self,
        context: MetricContext,
        answer: str,
        context_text: str
    ) -> MetricResult:
        """Calculate consistency using LLM."""
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import ChatPromptTemplate

            llm = context.llm_client

            prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个事实一致性评估专家。请判断给定的答案是否完全基于提供的上下文信息。

评分标准：
- 1.0: 答案完全基于上下文，没有任何虚构内容
- 0.7: 答案主要基于上下文，有少量推断但合理
- 0.4: 答案部分基于上下文，有一些无法验证的内容
- 0.0: 答案包含明显的虚假信息或与上下文矛盾

请只返回一个0到1之间的数字，不要包含其他内容。"""),
                ("human", """上下文：
{context}

答案：
{answer}

请评估答案的事实一致性：""")
            ])

            chain = prompt | llm
            result = await chain.ainvoke({
                "context": context_text[:2000],  # Limit context length
                "answer": answer
            })

            score_text = result.content.strip()
            score = float(score_text)

            return self._create_result(
                score=min(max(score, 0.0), 1.0),
                details={
                    "method": "llm",
                    "raw_response": score_text,
                }
            )

        except Exception as e:
            logger.warning(f"LLM calculation failed: {e}, falling back to simple method")
            return self._calculate_simple(answer, context_text)

    def _calculate_simple(self, answer: str, context_text: str) -> MetricResult:
        """Simple overlap-based consistency check."""
        # Extract key phrases from answer
        answer_sentences = re.split(r'[。！？.!?]', answer)
        answer_sentences = [s.strip() for s in answer_sentences if s.strip()]

        if not answer_sentences:
            return self._create_result(score=0.5)

        context_words = set(context_text.lower().split())
        consistent_count = 0

        for sentence in answer_sentences:
            words = set(sentence.lower().split())
            if not words:
                continue

            # Check if sentence words appear in context
            overlap = len(words & context_words)
            if overlap / len(words) >= 0.5:
                consistent_count += 1

        score = consistent_count / len(answer_sentences) if answer_sentences else 0.5

        return self._create_result(
            score=score,
            details={
                "method": "simple_overlap",
                "total_sentences": len(answer_sentences),
                "consistent_sentences": consistent_count,
            }
        )


class AnswerRelevanceMetric(BaseMetric):
    """
    答案与问题的相关性
    检查答案是否回应了用户的问题。
    """

    name = "answer_relevance"
    category = MetricCategory.GENERATION
    description = "答案与问题的相关性"
    requires_llm = True
    threshold = 0.6

    async def calculate(self, context: MetricContext) -> MetricResult:
        query = context.query
        answer = context.answer

        if not answer:
            return self._create_result(
                score=0.0,
                details={"reason": "No answer provided"}
            )

        # Use LLM if available
        if context.llm_client:
            return await self._calculate_with_llm(context, query, answer)

        # Fallback: Keyword overlap
        return self._calculate_simple(query, answer)

    async def _calculate_with_llm(
        self,
        context: MetricContext,
        query: str,
        answer: str
    ) -> MetricResult:
        """Calculate relevance using LLM."""
        try:
            from langchain_core.prompts import ChatPromptTemplate

            llm = context.llm_client

            prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个答案相关性评估专家。请判断给定的答案是否有效回应了用户的问题。

评分标准：
- 1.0: 答案完全回应了问题，非常相关
- 0.7: 答案主要回应了问题，基本相关
- 0.4: 答案与问题有一定关联但不完全匹配
- 0.0: 答案与问题完全无关

请只返回一个0到1之间的数字。"""),
                ("human", """问题：{query}
答案：{answer}

请评估答案的相关性：""")
            ])

            chain = prompt | llm
            result = await chain.ainvoke({"query": query, "answer": answer})

            score = float(result.content.strip())
            return self._create_result(score=min(max(score, 0.0), 1.0))

        except Exception as e:
            logger.warning(f"LLM calculation failed: {e}")
            return self._calculate_simple(query, answer)

    def _calculate_simple(self, query: str, answer: str) -> MetricResult:
        """Simple keyword-based relevance."""
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())

        if not query_words:
            return self._create_result(score=0.5)

        overlap = len(query_words & answer_words)
        score = overlap / len(query_words)

        return self._create_result(
            score=min(score, 1.0),
            details={"method": "keyword_overlap"}
        )


class AnswerCompletenessMetric(BaseMetric):
    """
    答案完整性
    检查答案是否完整回答了问题。
    """

    name = "answer_completeness"
    category = MetricCategory.GENERATION
    description = "答案完整性"
    requires_llm = True
    threshold = 0.6

    async def calculate(self, context: MetricContext) -> MetricResult:
        query = context.query
        answer = context.answer
        std_answers = context.standard_answers

        if not answer:
            return self._create_result(
                score=0.0,
                details={"reason": "No answer provided"}
            )

        # If standard answers available, compare against them
        if std_answers:
            return self._compare_with_standard(answer, std_answers)

        # Use LLM if available
        if context.llm_client:
            return await self._calculate_with_llm(context, query, answer)

        # Fallback: Length-based heuristic
        return self._calculate_heuristic(query, answer)

    def _compare_with_standard(
        self,
        answer: str,
        std_answers: list[str]
    ) -> MetricResult:
        """Compare answer with standard answers."""
        answer_words = set(answer.lower().split())

        best_match = 0.0
        for std in std_answers:
            std_words = set(std.lower().split())
            if not std_words:
                continue

            overlap = len(answer_words & std_words)
            match_score = overlap / len(std_words)
            best_match = max(best_match, match_score)

        return self._create_result(
            score=best_match,
            details={
                "method": "standard_answer_comparison",
                "best_match": best_match,
            }
        )

    async def _calculate_with_llm(
        self,
        context: MetricContext,
        query: str,
        answer: str
    ) -> MetricResult:
        """Calculate completeness using LLM."""
        try:
            from langchain_core.prompts import ChatPromptTemplate

            llm = context.llm_client

            prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个答案完整性评估专家。请判断给定的答案是否完整地回答了用户的问题。

评分标准：
- 1.0: 答案完整，覆盖了问题的所有方面
- 0.7: 答案基本完整，覆盖了主要方面
- 0.4: 答案不完整，只回答了部分问题
- 0.0: 答案完全未回答问题

请只返回一个0到1之间的数字。"""),
                ("human", """问题：{query}
答案：{answer}

请评估答案的完整性：""")
            ])

            chain = prompt | llm
            result = await chain.ainvoke({"query": query, "answer": answer})

            score = float(result.content.strip())
            return self._create_result(score=min(max(score, 0.0), 1.0))

        except Exception as e:
            logger.warning(f"LLM calculation failed: {e}")
            return self._calculate_heuristic(query, answer)

    def _calculate_heuristic(self, query: str, answer: str) -> MetricResult:
        """Length-based completeness heuristic."""
        # Simple heuristic based on answer length relative to question
        query_len = len(query.split())
        answer_len = len(answer.split())

        # Expect answer to be at least as long as query
        if answer_len >= query_len * 2:
            score = 1.0
        elif answer_len >= query_len:
            score = 0.7
        elif answer_len >= query_len * 0.5:
            score = 0.4
        else:
            score = 0.2

        return self._create_result(
            score=score,
            details={
                "method": "length_heuristic",
                "query_length": query_len,
                "answer_length": answer_len,
            }
        )


class AnswerFluencyMetric(BaseMetric):
    """
    答案流畅度、可读性
    评估答案的语言质量和可读性。
    """

    name = "answer_fluency"
    category = MetricCategory.GENERATION
    description = "答案流畅度和可读性"
    requires_llm = False
    threshold = 0.6

    async def calculate(self, context: MetricContext) -> MetricResult:
        answer = context.answer

        if not answer:
            return self._create_result(
                score=0.0,
                details={"reason": "No answer provided"}
            )

        # Calculate fluency metrics
        sentences = re.split(r'[。！？.!?]', answer)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return self._create_result(score=0.0)

        # Average sentence length
        avg_len = sum(len(s.split()) for s in sentences) / len(sentences)

        # Penalize very short or very long sentences
        if avg_len < 3:
            length_score = 0.3
        elif avg_len > 50:
            length_score = 0.5
        else:
            length_score = 1.0

        # Check for repetition
        words = answer.lower().split()
        if words:
            unique_ratio = len(set(words)) / len(words)
        else:
            unique_ratio = 0

        repetition_score = unique_ratio

        # Combined fluency score
        fluency = (length_score * 0.5 + repetition_score * 0.5)

        return self._create_result(
            score=fluency,
            details={
                "sentence_count": len(sentences),
                "avg_sentence_length": avg_len,
                "unique_word_ratio": unique_ratio,
                "length_score": length_score,
                "repetition_score": repetition_score,
            }
        )


class RefusalAccuracyMetric(BaseMetric):
    """
    拒答准确率
    评估系统是否正确识别应该拒答/不拒答的场景。
    """

    name = "refusal_accuracy"
    category = MetricCategory.GENERATION
    description = "拒答准确率：正确识别应该拒答/不拒答的场景"
    requires_llm = False
    threshold = 0.8

    async def calculate(self, context: MetricContext) -> MetricResult:
        should_refuse = context.should_refuse
        is_refused = context.is_refused

        # Calculate accuracy
        # True Positive: should refuse AND did refuse
        # True Negative: should NOT refuse AND did NOT refuse
        # False Positive: should NOT refuse BUT refused
        # False Negative: should refuse BUT did NOT refuse

        if should_refuse and is_refused:
            # True Positive
            score = 1.0
            outcome = "true_positive"
        elif not should_refuse and not is_refused:
            # True Negative
            score = 1.0
            outcome = "true_negative"
        elif not should_refuse and is_refused:
            # False Positive - incorrectly refused
            score = 0.0
            outcome = "false_positive"
        else:
            # False Negative - should have refused but didn't
            score = 0.0
            outcome = "false_negative"

        return self._create_result(
            score=score,
            details={
                "should_refuse": should_refuse,
                "is_refused": is_refused,
                "outcome": outcome,
            }
        )


class HallucinationDetectionMetric(BaseMetric):
    """
    幻觉检测评分
    检测答案中可能的幻觉内容。
    """

    name = "hallucination_detection"
    category = MetricCategory.GENERATION
    description = "幻觉检测评分"
    requires_llm = True
    threshold = 0.7

    async def calculate(self, context: MetricContext) -> MetricResult:
        answer = context.answer
        retrieved = context.retrieved_documents

        if not answer:
            return self._create_result(
                score=0.0,
                details={"reason": "No answer provided"}
            )

        if not retrieved:
            # Can't detect hallucinations without context
            return self._create_result(
                score=0.5,
                details={"reason": "No context to verify against"}
            )

        # Use LLM if available
        if context.llm_client:
            return await self._calculate_with_llm(context, answer, retrieved)

        # Fallback: Simple check for unsupported claims
        return self._calculate_simple(answer, retrieved)

    async def _calculate_with_llm(
        self,
        context: MetricContext,
        answer: str,
        retrieved: list[str]
    ) -> MetricResult:
        """Detect hallucinations using LLM."""
        try:
            from langchain_core.prompts import ChatPromptTemplate

            llm = context.llm_client
            context_text = " ".join(retrieved)[:3000]

            prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个幻觉检测专家。请检查答案中是否包含不在上下文中的虚假信息。

评分标准：
- 1.0: 完全没有幻觉，所有信息都有上下文支持
- 0.7: 轻微幻觉，大部分信息有支持
- 0.4: 明显幻觉，部分信息无法验证
- 0.0: 严重幻觉，大量虚假信息

请只返回一个0到1之间的数字。"""),
                ("human", """上下文：
{context}

答案：
{answer}

请评估幻觉程度：""")
            ])

            chain = prompt | llm
            result = await chain.ainvoke({
                "context": context_text,
                "answer": answer
            })

            score = float(result.content.strip())
            # Higher score = less hallucination
            return self._create_result(
                score=min(max(score, 0.0), 1.0),
                details={"method": "llm"}
            )

        except Exception as e:
            logger.warning(f"LLM hallucination detection failed: {e}")
            return self._calculate_simple(answer, retrieved)

    def _calculate_simple(
        self,
        answer: str,
        retrieved: list[str]
    ) -> MetricResult:
        """Simple hallucination detection based on content overlap."""
        context_words = set(" ".join(retrieved).lower().split())
        answer_words = set(answer.lower().split())

        if not answer_words:
            return self._create_result(score=0.5)

        # Check what portion of answer words are in context
        supported = len(answer_words & context_words)
        total = len(answer_words)

        # Filter out common words for better accuracy
        common_words = {"的", "是", "在", "有", "和", "了", "不", "这", "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "shall", "can", "need", "dare", "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "through", "during", "before", "after", "above", "below", "between", "under", "again", "further", "then", "once"}
        answer_words_filtered = answer_words - common_words

        if answer_words_filtered:
            supported_filtered = len(answer_words_filtered & context_words)
            total_filtered = len(answer_words_filtered)
            score = supported_filtered / total_filtered
        else:
            score = supported / total

        return self._create_result(
            score=score,
            details={
                "method": "content_overlap",
                "supported_words": supported,
                "total_words": total,
            }
        )