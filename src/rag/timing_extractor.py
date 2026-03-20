"""
Timing extractor for RAG performance tracking.
Extracts timing data from various sources based on configuration.
"""

import re
from typing import Any, Optional

from ..models.rag_response import StageTiming
from .timing_config import (
    TimingExtractionConfig,
    TimingExtractionStrategy,
    StageTimingExtractor,
)


class TimingExtractor:
    """
    高度灵活的 timing 提取器。
    支持多种数据来源和自定义逻辑。

    使用方式:
        config = TimingExtractionConfig(
            stages={
                "query_rewrite": StageTimingExtractor(
                    stage="query_rewrite",
                    strategy=TimingExtractionStrategy.STATE,
                    field_path="query_rewrite.timing_ms",
                ),
                ...
            }
        )
        extractor = TimingExtractor(config)
        timing = extractor.extract(response_data, overall_ms)
    """

    # 安全表达式允许的函数和常量
    SAFE_BUILTINS = {
        "len": len,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "round": round,
    }

    def __init__(self, config: Optional[TimingExtractionConfig] = None):
        """
        初始化提取器。

        Args:
            config: Timing提取配置，如果为None则使用默认配置
        """
        self.config = config or TimingExtractionConfig()

    def extract(
        self,
        response_data: dict[str, Any],
        overall_ms: float,
    ) -> StageTiming:
        """
        从 RAG 响应中提取 timing。

        Args:
            response_data: RAG 原始响应数据（字典格式）
            overall_ms: 整体延迟（用于 fallback 和计算）

        Returns:
            StageTiming 对象
        """
        timing = StageTiming(
            source="extracted",
            total_ms=overall_ms,
        )

        # 遍历所有阶段
        all_stages = ["query_rewrite", "faq_match", "retrieval", "rerank", "generation"]

        for stage in all_stages:
            # 获取该阶段的提取配置
            extractor = self.config.stages.get(stage)

            if extractor:
                ms = self._extract_stage(response_data, extractor, overall_ms)
            else:
                # 使用 fallback 策略
                ms = overall_ms * self.config.fallback_proportions.get(stage, 0.1)
                timing.extraction_details[stage] = "fallback.default"

            # 设置阶段耗时
            setattr(timing, f"{stage}_ms", ms)

            # 记录提取来源
            if extractor:
                timing.extraction_details[stage] = extractor.strategy.value

        return timing

    def _extract_stage(
        self,
        data: dict[str, Any],
        extractor: StageTimingExtractor,
        overall_ms: float,
    ) -> float:
        """
        提取单个阶段的 timing。

        Args:
            data: RAG 响应数据
            extractor: 阶段提取配置
            overall_ms: 整体延迟

        Returns:
            该阶段的耗时（毫秒）
        """
        strategy = extractor.strategy

        try:
            if strategy == TimingExtractionStrategy.STATE:
                return self._from_state(data, extractor)

            elif strategy == TimingExtractionStrategy.METADATA:
                return self._from_metadata(data, extractor)

            elif strategy == TimingExtractionStrategy.FIELD_PATH:
                return self._from_path(data, extractor)

            elif strategy == TimingExtractionStrategy.CALCULATED:
                return self._calculate(data, extractor, overall_ms)

            elif strategy == TimingExtractionStrategy.CUSTOM:
                return self._from_custom(data, extractor)

            elif strategy == TimingExtractionStrategy.STREAMING:
                # 流式测量在外部处理，这里返回默认值
                return extractor.default_ms

            else:  # FALLBACK
                return overall_ms * self.config.fallback_proportions.get(
                    extractor.stage, 0.1
                )

        except Exception as e:
            # 提取失败时返回默认值或 fallback
            if extractor.default_ms > 0:
                return extractor.default_ms
            return overall_ms * self.config.fallback_proportions.get(
                extractor.stage, 0.1
            )

    def _from_state(
        self,
        data: dict[str, Any],
        extractor: StageTimingExtractor,
    ) -> float:
        """
        从 LangGraph state 提取 timing。

        支持 state.xxx.xxx 格式的路径。
        """
        # 尝试主路径
        value = self._get_nested(data, extractor.field_path)

        # 尝试备用路径
        if value is None and extractor.fallback_paths:
            for path in extractor.fallback_paths:
                value = self._get_nested(data, path)
                if value is not None:
                    break

        return self._convert_unit(value, extractor.unit)

    def _from_metadata(
        self,
        data: dict[str, Any],
        extractor: StageTimingExtractor,
    ) -> float:
        """从 metadata 字段提取 timing"""
        metadata = data.get("metadata", {})

        # 支持 metadata.xxx 和直接 xxx 两种格式
        if extractor.field_path:
            # 先尝试从 metadata 中获取
            value = self._get_nested({"metadata": metadata}, extractor.field_path)
            if value is None:
                # 再尝试直接从 metadata 字典获取
                value = self._get_nested(metadata, extractor.field_path.replace("metadata.", ""))

        else:
            value = None

        # 尝试备用路径
        if value is None and extractor.fallback_paths:
            for path in extractor.fallback_paths:
                value = self._get_nested({"metadata": metadata}, path)
                if value is not None:
                    break

        return self._convert_unit(value, extractor.unit)

    def _from_path(
        self,
        data: dict[str, Any],
        extractor: StageTimingExtractor,
    ) -> float:
        """按任意路径提取 timing"""
        value = self._get_nested(data, extractor.field_path)

        # 尝试备用路径
        if value is None and extractor.fallback_paths:
            for path in extractor.fallback_paths:
                value = self._get_nested(data, path)
                if value is not None:
                    break

        return self._convert_unit(value, extractor.unit)

    def _calculate(
        self,
        data: dict[str, Any],
        extractor: StageTimingExtractor,
        overall_ms: float,
    ) -> float:
        """
        基于表达式计算 timing。

        支持的表达式变量:
        - retrieval: 检索结果列表
        - rerank: 重排结果列表
        - query_rewrite: 查询改写结果
        - faq_match: FAQ 匹配结果
        - llm_output: LLM 输出
        - token_count: token 数量
        - overall_ms: 总延迟
        - response: 完整响应数据
        """
        expr = extractor.calculate_expression
        if not expr:
            return extractor.default_ms

        # 准备上下文变量
        retrieval = data.get("retrieval", [])
        rerank = data.get("rerank", [])
        query_rewrite = data.get("query_rewrite")
        faq_match = data.get("faq_match", {})
        llm_output = data.get("llm_output", {})
        token_usage = llm_output.get("token_usage", {})

        context = {
            "retrieval": retrieval,
            "rerank": rerank,
            "query_rewrite": query_rewrite,
            "faq_match": faq_match,
            "llm_output": llm_output,
            "token_count": token_usage.get("total_tokens", 0),
            "overall_ms": overall_ms,
            "response": data,
            "len": len,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "round": round,
        }

        try:
            # 安全表达式求值
            result = eval(expr, {"__builtins__": self.SAFE_BUILTINS}, context)
            return float(result) if result is not None else extractor.default_ms
        except Exception:
            return extractor.default_ms

    def _from_custom(
        self,
        data: dict[str, Any],
        extractor: StageTimingExtractor,
    ) -> float:
        """使用自定义函数提取"""
        if extractor.custom_extractor is None:
            return extractor.default_ms

        try:
            result = extractor.custom_extractor(data)
            return float(result) if result is not None else extractor.default_ms
        except Exception:
            return extractor.default_ms

    def _get_nested(self, data: dict[str, Any], path: Optional[str]) -> Any:
        """
        获取嵌套值，支持 path.to.field 格式。

        Args:
            data: 数据字典
            path: 字段路径，如 "state.query_rewrite.timing_ms"

        Returns:
            字段值，如果路径不存在则返回 None
        """
        if not path or not data:
            return None

        keys = path.split(".")
        value = data

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            elif hasattr(value, key):
                # 支持对象属性访问
                value = getattr(value, key)
            else:
                return None

        return value

    def _convert_unit(self, value: Any, unit: str) -> float:
        """
        单位转换。

        Args:
            value: 原始值
            unit: 单位 ("ms", "s")

        Returns:
            毫秒值
        """
        if value is None:
            return 0.0

        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0

        if unit == "s":
            return value * 1000  # 秒转毫秒

        return value

    def extract_from_response(
        self,
        response: Any,
        overall_ms: float,
    ) -> StageTiming:
        """
        从 RAGResponse 对象提取 timing。

        Args:
            response: RAGResponse 对象
            overall_ms: 整体延迟

        Returns:
            StageTiming 对象
        """
        # 转换为字典
        if hasattr(response, "to_dict"):
            data = response.to_dict()
        elif hasattr(response, "model_dump"):
            data = response.model_dump(mode="json")
        elif hasattr(response, "__dict__"):
            data = response.__dict__
        else:
            data = dict(response) if response else {}

        return self.extract(data, overall_ms)