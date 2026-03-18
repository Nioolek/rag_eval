"""
Comprehensive and additional metrics for RAG evaluation.
Metrics: Multi-Answer Match, Style Match, Conversation Consistency, Context Utilization, Answer Repetition
"""

import re
from typing import Any, Optional

from .base import BaseMetric, MetricContext
from ...models.metric_result import MetricCategory, MetricResult
from ...core.logging import logger


class MultiAnswerMatchMetric(BaseMetric):
    """
    多标准答案匹配度
    评估答案与多个标准答案的最佳匹配程度。
    """

    name = "multi_answer_match"
    category = MetricCategory.COMPREHENSIVE
    description = "多标准答案匹配度：答案与多个标准答案的最佳匹配"
    requires_llm = False
    threshold = 0.6

    async def calculate(self, context: MetricContext) -> MetricResult:
        answer = context.answer
        standard_answers = context.standard_answers

        if not answer:
            return self._create_result(
                score=0.0,
                details={"reason": "No answer provided"}
            )

        if not standard_answers:
            return self._create_result(
                score=1.0,  # No standard to compare, assume correct
                details={"reason": "No standard answers defined"}
            )

        # Calculate match with each standard answer
        answer_words = set(answer.lower().split())
        match_scores = []

        for idx, std_answer in enumerate(standard_answers):
            std_words = set(std_answer.lower().split())

            if not std_words:
                continue

            # Jaccard similarity
            intersection = len(answer_words & std_words)
            union = len(answer_words | std_words)
            jaccard = intersection / union if union > 0 else 0

            # Coverage (how much of standard answer is covered)
            coverage = intersection / len(std_words) if std_words else 0

            # Combined score
            score = (jaccard + coverage) / 2
            match_scores.append((idx, score, jaccard, coverage))

        if not match_scores:
            return self._create_result(score=0.5)

        # Get best match
        best_match = max(match_scores, key=lambda x: x[1])

        return self._create_result(
            score=best_match[1],
            details={
                "best_match_index": best_match[0],
                "best_match_score": best_match[1],
                "jaccard": best_match[2],
                "coverage": best_match[3],
                "all_scores": [s[1] for s in match_scores],
            }
        )


class StyleMatchMetric(BaseMetric):
    """
    回答风格匹配度
    评估回答风格是否符合要求。
    """

    name = "style_match"
    category = MetricCategory.COMPREHENSIVE
    description = "回答风格匹配度"
    requires_llm = True
    threshold = 0.6

    async def calculate(self, context: MetricContext) -> MetricResult:
        answer = context.answer
        required_style = context.annotation.answer_style

        if not answer:
            return self._create_result(
                score=0.0,
                details={"reason": "No answer provided"}
            )

        if not required_style:
            return self._create_result(
                score=1.0,  # No style requirement
                details={"reason": "No style requirement defined"}
            )

        # Use LLM if available
        if context.llm_client:
            return await self._calculate_with_llm(context, answer, required_style)

        # Fallback: Simple keyword check
        return self._calculate_simple(answer, required_style)

    async def _calculate_with_llm(
        self,
        context: MetricContext,
        answer: str,
        required_style: str
    ) -> MetricResult:
        """Evaluate style using LLM."""
        try:
            from langchain_core.prompts import ChatPromptTemplate

            llm = context.llm_client

            prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个回答风格评估专家。请评估答案是否符合指定的风格要求。

评分标准：
- 1.0: 完全符合风格要求
- 0.7: 基本符合，有小偏差
- 0.4: 部分符合
- 0.0: 完全不符合

请只返回一个0到1之间的数字。"""),
                ("human", """风格要求：{style}
答案：{answer}

请评估风格匹配度：""")
            ])

            chain = prompt | llm
            result = await chain.ainvoke({
                "style": required_style,
                "answer": answer
            })

            score = float(result.content.strip())
            return self._create_result(
                score=min(max(score, 0.0), 1.0),
                details={"method": "llm", "required_style": required_style}
            )

        except Exception as e:
            logger.warning(f"LLM style evaluation failed: {e}")
            return self._calculate_simple(answer, required_style)

    def _calculate_simple(self, answer: str, required_style: str) -> MetricResult:
        """Simple style check based on keywords."""
        style_keywords = {
            "专业": ["根据", "分析", "研究", "数据", "证据", "因此"],
            "简洁": [],  # Short sentences indicator
            "详细": ["首先", "其次", "另外", "此外", "总结", "具体"],
            "友好": ["您好", "请", "谢谢", "希望能帮助", "欢迎"],
            "正式": ["尊敬的", "谨此", "特此", "敬启"],
        }

        style_lower = required_style.lower()
        score = 0.5  # Default neutral score

        for style_name, keywords in style_keywords.items():
            if style_name in style_lower:
                if keywords:
                    matches = sum(1 for kw in keywords if kw in answer)
                    score = matches / len(keywords)
                break

        return self._create_result(
            score=score,
            details={
                "method": "keyword",
                "required_style": required_style
            }
        )


class ConversationConsistencyMetric(BaseMetric):
    """
    多轮对话一致性
    评估多轮对话中回答的一致性。
    """

    name = "conversation_consistency"
    category = MetricCategory.COMPREHENSIVE
    description = "多轮对话一致性"
    requires_llm = False
    threshold = 0.7

    async def calculate(self, context: MetricContext) -> MetricResult:
        answer = context.answer
        history = context.annotation.conversation_history

        if not history:
            return self._create_result(
                score=1.0,  # Single turn, no inconsistency possible
                details={
                    "reason": "Single turn conversation",
                    "applicable": False,
                }
            )

        if not answer:
            return self._create_result(
                score=0.0,
                details={"reason": "No answer provided"}
            )

        # Check for contradictions between current answer and history
        # Simple approach: check for negation patterns that might indicate contradiction

        answer_words = set(answer.lower().split())

        # Look for contradiction indicators
        contradiction_patterns = [
            r"不是",
            r"不对",
            r"错误",
            r"之前说的不对",
            r"实际上",
            r"其实",
            r"纠正",
        ]

        contradiction_count = 0
        for pattern in contradiction_patterns:
            if re.search(pattern, answer):
                contradiction_count += 1

        # High contradiction count suggests inconsistency
        if contradiction_count >= 2:
            score = 0.3
        elif contradiction_count == 1:
            score = 0.6
        else:
            score = 1.0

        return self._create_result(
            score=score,
            details={
                "applicable": True,
                "history_length": len(history),
                "contradiction_indicators": contradiction_count,
            }
        )


class ContextUtilizationMetric(BaseMetric):
    """
    上下文利用率
    评估答案对检索上下文的利用程度。
    """

    name = "context_utilization"
    category = MetricCategory.COMPREHENSIVE
    description = "上下文利用率：答案对检索上下文的利用程度"
    requires_llm = False
    threshold = 0.5

    async def calculate(self, context: MetricContext) -> MetricResult:
        answer = context.answer
        retrieved = context.retrieved_documents

        if not answer:
            return self._create_result(
                score=0.0,
                details={"reason": "No answer provided"}
            )

        if not retrieved:
            return self._create_result(
                score=0.0,
                details={"reason": "No context retrieved"}
            )

        # Calculate how much of the context is utilized
        context_text = " ".join(retrieved)
        context_words = set(context_text.lower().split())
        answer_words = set(answer.lower().split())

        if not context_words:
            return self._create_result(score=0.0)

        # How many context words appear in answer
        utilized = len(answer_words & context_words)

        # Utilization rate relative to answer length
        if answer_words:
            utilization_in_answer = utilized / len(answer_words)
        else:
            utilization_in_answer = 0

        # Utilization rate relative to context length
        utilization_of_context = utilized / len(context_words)

        # Combined score - balance between using context and not just copying
        score = (utilization_in_answer * 0.7 + utilization_of_context * 0.3)

        return self._create_result(
            score=score,
            details={
                "context_word_count": len(context_words),
                "answer_word_count": len(answer_words),
                "utilized_words": utilized,
                "utilization_in_answer": utilization_in_answer,
                "utilization_of_context": utilization_of_context,
            }
        )


class AnswerRepetitionMetric(BaseMetric):
    """
    答案重复率检测
    评估答案中的重复内容。
    """

    name = "answer_repetition"
    category = MetricCategory.COMPREHENSIVE
    description = "答案重复率检测"
    requires_llm = False
    threshold = 0.7  # Higher is better (less repetition)

    async def calculate(self, context: MetricContext) -> MetricResult:
        answer = context.answer

        if not answer:
            return self._create_result(
                score=0.0,
                details={"reason": "No answer provided"}
            )

        # Split into sentences
        sentences = re.split(r'[。！？.!?]', answer)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 1:
            return self._create_result(
                score=1.0,  # No repetition possible
                details={"reason": "Single sentence answer"}
            )

        # Check for repeated sentences
        sentence_counts = {}
        for s in sentences:
            normalized = s.lower().strip()
            sentence_counts[normalized] = sentence_counts.get(normalized, 0) + 1

        repeated_sentences = sum(c - 1 for c in sentence_counts.values() if c > 1)
        repetition_rate = repeated_sentences / len(sentences)

        # Also check for repeated phrases (n-grams)
        words = answer.lower().split()
        if len(words) >= 3:
            trigrams = [' '.join(words[i:i+3]) for i in range(len(words) - 2)]
            trigram_counts = {}
            for t in trigrams:
                trigram_counts[t] = trigram_counts.get(t, 0) + 1
            repeated_trigrams = sum(c - 1 for c in trigram_counts.values() if c > 1)
            trigram_repetition = repeated_trigrams / len(trigrams) if trigrams else 0
        else:
            trigram_repetition = 0

        # Combined repetition score
        repetition = (repetition_rate * 0.6 + trigram_repetition * 0.4)

        # Convert to quality score (higher = less repetition)
        score = 1.0 - min(repetition, 1.0)

        return self._create_result(
            score=score,
            details={
                "sentence_count": len(sentences),
                "repeated_sentences": repeated_sentences,
                "sentence_repetition_rate": repetition_rate,
                "trigram_repetition_rate": trigram_repetition,
                "overall_repetition": repetition,
            }
        )