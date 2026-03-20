"""
Performance metrics for RAG evaluation.
Tracks latency and timing metrics for each RAG pipeline stage.
"""

import time
from typing import Any, Optional

from ..models.metric_result import MetricResult, MetricCategory
from ..models.rag_response import RAGResponse
from ..models.annotation import Annotation
from .base import BaseMetric, MetricContext


class StageLatencyMetric(BaseMetric):
    """
    各阶段延迟指标。
    直接从 stage_timing 读取各阶段耗时。
    """

    name = "stage_latency"
    category = MetricCategory.PERFORMANCE
    description = "RAG 各处理阶段的延迟分析"
    requires_llm = False
    threshold = 0.0  # 性能指标没有通过/失败的阈值概念

    async def calculate(self, context: MetricContext) -> MetricResult:
        """计算各阶段延迟"""
        response = context.rag_response

        if not response or not response.stage_timing:
            return self._create_result(
                score=0.0,
                details={"error": "无 stage_timing 数据"},
            )

        timing = response.stage_timing
        stage_timings = timing.get_stage_timings()
        percentages = timing.get_percentages()

        details = {
            "total_ms": timing.total_ms,
            "stages": {},
            "extraction_source": timing.source,
        }

        for stage, ms in stage_timings.items():
            details["stages"][stage] = {
                "latency_ms": round(ms, 2),
                "percentage": round(percentages.get(stage, 0), 2),
                "source": timing.extraction_details.get(stage, "unknown"),
            }

        # 性能分数：基于总延迟（越低越好）
        # 假设 1秒以内为满分，超过5秒为0分
        if timing.total_ms <= 1000:
            score = 1.0
        elif timing.total_ms >= 5000:
            score = 0.0
        else:
            score = 1.0 - (timing.total_ms - 1000) / 4000

        return self._create_result(
            score=score,
            raw_score=timing.total_ms,
            details=details,
        )


class TotalLatencyMetric(BaseMetric):
    """
    总延迟指标。
    记录整体 RAG 响应时间。
    """

    name = "total_latency"
    category = MetricCategory.PERFORMANCE
    description = "RAG 系统整体响应延迟"
    requires_llm = False
    threshold = 0.0

    # 延迟评级阈值（毫秒）
    EXCELLENT_THRESHOLD = 500  # 优秀
    GOOD_THRESHOLD = 1000      # 良好
    ACCEPTABLE_THRESHOLD = 2000  # 可接受

    async def calculate(self, context: MetricContext) -> MetricResult:
        """计算总延迟"""
        response = context.rag_response

        if not response:
            return self._create_result(
                score=0.0,
                details={"error": "无 RAG 响应"},
            )

        latency_ms = response.latency_ms

        # 评级
        if latency_ms <= self.EXCELLENT_THRESHOLD:
            rating = "excellent"
            score = 1.0
        elif latency_ms <= self.GOOD_THRESHOLD:
            rating = "good"
            score = 0.8
        elif latency_ms <= self.ACCEPTABLE_THRESHOLD:
            rating = "acceptable"
            score = 0.6
        else:
            rating = "slow"
            score = max(0.0, 0.4 - (latency_ms - self.ACCEPTABLE_THRESHOLD) / 10000)

        details = {
            "latency_ms": round(latency_ms, 2),
            "latency_s": round(latency_ms / 1000, 3),
            "rating": rating,
        }

        return self._create_result(
            score=score,
            raw_score=latency_ms,
            details=details,
        )


class LatencyDistributionMetric(BaseMetric):
    """
    延迟分布指标。
    计算批量评测中的 P50/P95/P99 延迟。
    需要聚合多个结果，单个结果只返回原始值。
    """

    name = "latency_distribution"
    category = MetricCategory.PERFORMANCE
    description = "延迟分布统计（P50/P95/P99）"
    requires_llm = False
    threshold = 0.0

    async def calculate(self, context: MetricContext) -> MetricResult:
        """记录延迟值（聚合计算在外部进行）"""
        response = context.rag_response

        if not response:
            return self._create_result(
                score=0.0,
                details={"error": "无 RAG 响应"},
            )

        latency_ms = response.latency_ms

        # 单条记录
        details = {
            "latency_ms": round(latency_ms, 2),
            "note": "聚合统计需要多条结果",
        }

        # 基于 latency 的分数
        score = max(0.0, min(1.0, 1.0 - latency_ms / 5000))

        return self._create_result(
            score=score,
            raw_score=latency_ms,
            details=details,
        )


class PerformanceComparisonMetric(BaseMetric):
    """
    双 RAG 性能对比指标。
    用于对比两个 RAG 接口的性能差异。
    """

    name = "performance_comparison"
    category = MetricCategory.PERFORMANCE
    description = "双 RAG 接口性能对比"
    requires_llm = False
    threshold = 0.0

    async def calculate(self, context: MetricContext) -> MetricResult:
        """
        计算性能对比。
        需要在 context.extra 中提供对比数据。
        """
        response = context.rag_response
        extra = context.extra

        # 获取对比数据
        compare_response = extra.get("compare_response")

        if not response:
            return self._create_result(
                score=0.0,
                details={"error": "无主 RAG 响应"},
            )

        main_latency = response.latency_ms

        details = {
            "main_latency_ms": round(main_latency, 2),
        }

        if compare_response and isinstance(compare_response, RAGResponse):
            compare_latency = compare_response.latency_ms
            diff_ms = main_latency - compare_latency
            diff_percent = (diff_ms / compare_latency * 100) if compare_latency > 0 else 0

            details.update({
                "compare_latency_ms": round(compare_latency, 2),
                "difference_ms": round(diff_ms, 2),
                "difference_percent": round(diff_percent, 2),
                "faster": "compare" if diff_ms > 0 else "main" if diff_ms < 0 else "equal",
            })

            # 分数：更快的一方得分更高
            if diff_ms <= 0:
                score = 1.0  # 主接口更快或相等
            else:
                # 慢的比例越大，分数越低
                score = max(0.0, 1.0 - diff_percent / 100)
        else:
            details["note"] = "无对比数据"
            score = 0.5  # 无对比数据时返回中性分数

        return self._create_result(
            score=score,
            raw_score=main_latency,
            details=details,
        )


class StageEfficiencyMetric(BaseMetric):
    """
    阶段效率指标。
    分析各阶段耗时占比，识别瓶颈。
    """

    name = "stage_efficiency"
    category = MetricCategory.PERFORMANCE
    description = "各阶段效率分析，识别性能瓶颈"
    requires_llm = False
    threshold = 0.0

    # 各阶段的理想占比
    IDEAL_PROPORTIONS = {
        "query_rewrite": 0.05,
        "faq_match": 0.03,
        "retrieval": 0.15,
        "rerank": 0.10,
        "generation": 0.50,  # 生成应该占大头
    }

    async def calculate(self, context: MetricContext) -> MetricResult:
        """计算阶段效率"""
        response = context.rag_response

        if not response or not response.stage_timing:
            return self._create_result(
                score=0.0,
                details={"error": "无 stage_timing 数据"},
            )

        timing = response.stage_timing
        percentages = timing.get_percentages()

        # 计算效率分数
        # 如果某阶段占比过高，说明可能是瓶颈
        efficiency_scores = {}
        bottlenecks = []

        for stage, actual_pct in percentages.items():
            ideal_pct = self.IDEAL_PROPORTIONS.get(stage, 0.1)

            # 效率分数：实际占比与理想占比的差异
            if actual_pct > 0:
                diff = abs(actual_pct / 100 - ideal_pct)
                stage_score = max(0, 1 - diff * 2)
                efficiency_scores[stage] = round(stage_score, 3)

                # 检测瓶颈
                if actual_pct / 100 > ideal_pct * 1.5:
                    bottlenecks.append({
                        "stage": stage,
                        "actual_percent": round(actual_pct, 2),
                        "ideal_percent": round(ideal_pct * 100, 2),
                        "severity": "high" if actual_pct / 100 > ideal_pct * 2 else "medium",
                    })
            else:
                efficiency_scores[stage] = 1.0  # 未使用的阶段

        # 总体效率分数
        overall_score = sum(efficiency_scores.values()) / len(efficiency_scores) if efficiency_scores else 0

        details = {
            "stage_efficiency": efficiency_scores,
            "bottlenecks": bottlenecks,
            "stage_percentages": {k: round(v, 2) for k, v in percentages.items()},
            "recommendations": self._generate_recommendations(bottlenecks),
        }

        return self._create_result(
            score=overall_score,
            details=details,
        )

    def _generate_recommendations(self, bottlenecks: list[dict]) -> list[str]:
        """生成优化建议"""
        recommendations = []

        for b in bottlenecks:
            stage = b["stage"]
            if stage == "query_rewrite":
                recommendations.append("查询改写耗时较长，可考虑缓存改写结果或简化改写逻辑")
            elif stage == "retrieval":
                recommendations.append("检索阶段耗时较长，可考虑优化索引、减少检索文档数或启用缓存")
            elif stage == "rerank":
                recommendations.append("重排序耗时较长，可考虑减少输入文档数或使用更快的重排序模型")
            elif stage == "generation":
                recommendations.append("生成阶段耗时较长，可考虑使用更快的模型或减少输出长度")
            elif stage == "faq_match":
                recommendations.append("FAQ匹配耗时较长，可考虑优化FAQ索引或减少匹配阈值")

        return recommendations


class ThroughputMetric(BaseMetric):
    """
    吞吐量指标。
    计算每秒可处理的请求数估算。
    """

    name = "throughput"
    category = MetricCategory.PERFORMANCE
    description = "RAG 系统吞吐量估算"
    requires_llm = False
    threshold = 0.0

    async def calculate(self, context: MetricContext) -> MetricResult:
        """计算吞吐量估算"""
        response = context.rag_response

        if not response:
            return self._create_result(
                score=0.0,
                details={"error": "无 RAG 响应"},
            )

        latency_ms = response.latency_ms

        if latency_ms <= 0:
            return self._create_result(
                score=0.0,
                details={"error": "延迟值无效"},
            )

        # 计算吞吐量（请求/秒）
        latency_s = latency_ms / 1000
        requests_per_second = 1 / latency_s

        # 吞吐量评级
        if requests_per_second >= 10:
            rating = "excellent"
            score = 1.0
        elif requests_per_second >= 5:
            rating = "good"
            score = 0.8
        elif requests_per_second >= 2:
            rating = "acceptable"
            score = 0.6
        elif requests_per_second >= 1:
            rating = "slow"
            score = 0.4
        else:
            rating = "very_slow"
            score = 0.2

        details = {
            "latency_ms": round(latency_ms, 2),
            "requests_per_second": round(requests_per_second, 3),
            "requests_per_minute": round(requests_per_second * 60, 1),
            "rating": rating,
        }

        return self._create_result(
            score=score,
            raw_score=requests_per_second,
            details=details,
        )