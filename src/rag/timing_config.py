"""
Timing extraction configuration for RAG performance tracking.
Supports flexible, configurable extraction of timing data from various sources.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class TimingExtractionStrategy(str, Enum):
    """Timing提取策略枚举"""

    STATE = "state"              # 从 LangGraph state 提取
    METADATA = "metadata"        # 从 metadata 字段提取
    FIELD_PATH = "field_path"    # 按字段路径提取
    CALCULATED = "calculated"    # 基于返回内容计算
    STREAMING = "streaming"      # 流式测量
    CUSTOM = "custom"            # 自定义函数
    FALLBACK = "fallback"        # 按比例估算


@dataclass
class StageTimingExtractor:
    """
    单个阶段的 timing 提取配置。

    Attributes:
        stage: 阶段名称 (query_rewrite, faq_match, retrieval, rerank, generation)
        strategy: 提取策略
        field_path: 字段路径，用于 STATE/METADATA/FIELD_PATH 策略
        fallback_paths: 备用字段路径列表
        unit: 单位 ("ms", "s")
        calculate_expression: 计算表达式，用于 CALCULATED 策略
        custom_extractor: 自定义提取函数，用于 CUSTOM 策略
        default_ms: 默认值（无法提取时使用）
    """

    stage: str

    strategy: TimingExtractionStrategy = TimingExtractionStrategy.FALLBACK

    # 字段路径配置
    field_path: Optional[str] = None
    fallback_paths: list[str] = field(default_factory=list)

    # 单位配置
    unit: str = "ms"  # "ms" 或 "s"

    # CALCULATED 策略配置
    calculate_expression: Optional[str] = None

    # CUSTOM 策略配置
    custom_extractor: Optional[Callable[[dict], float]] = None

    # 默认值
    default_ms: float = 0.0

    def __post_init__(self):
        """验证配置"""
        if self.strategy == TimingExtractionStrategy.FIELD_PATH and not self.field_path:
            raise ValueError(f"FIELD_PATH strategy requires field_path for stage: {self.stage}")

        if self.strategy == TimingExtractionStrategy.CUSTOM and not self.custom_extractor:
            raise ValueError(f"CUSTOM strategy requires custom_extractor for stage: {self.stage}")


@dataclass
class TimingExtractionConfig:
    """
    Timing提取全局配置。

    Attributes:
        stages: 各阶段的提取配置
        default_strategy: 默认策略（未配置的阶段使用此策略）
        fallback_proportions: Fallback 比例配置
        enable_streaming_measurement: 是否在流式输出时测量 timing
    """

    stages: dict[str, StageTimingExtractor] = field(default_factory=dict)

    default_strategy: TimingExtractionStrategy = TimingExtractionStrategy.FALLBACK

    # Fallback 比例配置（当无法提取时使用）
    fallback_proportions: dict[str, float] = field(default_factory=lambda: {
        "query_rewrite": 0.05,
        "faq_match": 0.05,
        "retrieval": 0.20,
        "rerank": 0.10,
        "generation": 0.60,
    })

    # 流式测量配置
    enable_streaming_measurement: bool = True

    @classmethod
    def from_dict(cls, config: dict) -> "TimingExtractionConfig":
        """从字典创建配置"""
        stages = {}
        for stage_name, stage_config in config.get("stages", {}).items():
            strategy = TimingExtractionStrategy(
                stage_config.get("strategy", "fallback")
            )
            stages[stage_name] = StageTimingExtractor(
                stage=stage_name,
                strategy=strategy,
                field_path=stage_config.get("field_path"),
                fallback_paths=stage_config.get("fallback_paths", []),
                unit=stage_config.get("unit", "ms"),
                calculate_expression=stage_config.get("calculate_expression"),
                default_ms=stage_config.get("default_ms", 0.0),
            )

        return cls(
            stages=stages,
            default_strategy=TimingExtractionStrategy(
                config.get("default_strategy", "fallback")
            ),
            fallback_proportions=config.get("fallback_proportions", {
                "query_rewrite": 0.05,
                "faq_match": 0.05,
                "retrieval": 0.20,
                "rerank": 0.10,
                "generation": 0.60,
            }),
            enable_streaming_measurement=config.get("enable_streaming_measurement", True),
        )


# ============== 预设配置模板 ==============

def get_default_config() -> TimingExtractionConfig:
    """获取默认配置（纯 fallback）"""
    return TimingExtractionConfig()


def get_langgraph_config() -> TimingExtractionConfig:
    """
    获取适用于标准 LangGraph RAG 的配置。
    假设 timing 数据在 state 中。
    """
    return TimingExtractionConfig(
        stages={
            "query_rewrite": StageTimingExtractor(
                stage="query_rewrite",
                strategy=TimingExtractionStrategy.STATE,
                field_path="query_rewrite.timing_ms",
                fallback_paths=["timing.query_rewrite_ms"],
            ),
            "faq_match": StageTimingExtractor(
                stage="faq_match",
                strategy=TimingExtractionStrategy.STATE,
                field_path="faq_match.timing_ms",
                fallback_paths=["timing.faq_match_ms"],
            ),
            "retrieval": StageTimingExtractor(
                stage="retrieval",
                strategy=TimingExtractionStrategy.STATE,
                field_path="retrieval.timing_ms",
                fallback_paths=["timing.retrieval_ms"],
            ),
            "rerank": StageTimingExtractor(
                stage="rerank",
                strategy=TimingExtractionStrategy.STATE,
                field_path="rerank.timing_ms",
                fallback_paths=["timing.rerank_ms"],
            ),
            "generation": StageTimingExtractor(
                stage="generation",
                strategy=TimingExtractionStrategy.STATE,
                field_path="llm_output.timing_ms",
                fallback_paths=["timing.generation_ms", "generation.timing_ms"],
            ),
        },
    )


def get_metadata_config() -> TimingExtractionConfig:
    """
    获取适用于 timing 在 metadata 中的服务配置。
    """
    return TimingExtractionConfig(
        stages={
            "query_rewrite": StageTimingExtractor(
                stage="query_rewrite",
                strategy=TimingExtractionStrategy.METADATA,
                field_path="timing.query_rewrite_ms",
            ),
            "faq_match": StageTimingExtractor(
                stage="faq_match",
                strategy=TimingExtractionStrategy.METADATA,
                field_path="timing.faq_match_ms",
            ),
            "retrieval": StageTimingExtractor(
                stage="retrieval",
                strategy=TimingExtractionStrategy.METADATA,
                field_path="timing.retrieval_ms",
            ),
            "rerank": StageTimingExtractor(
                stage="rerank",
                strategy=TimingExtractionStrategy.METADATA,
                field_path="timing.rerank_ms",
            ),
            "generation": StageTimingExtractor(
                stage="generation",
                strategy=TimingExtractionStrategy.METADATA,
                field_path="timing.generation_ms",
            ),
        },
    )


def get_calculated_config() -> TimingExtractionConfig:
    """
    获取基于内容计算的配置。
    适用于没有 timing 数据返回的场景。
    """
    return TimingExtractionConfig(
        stages={
            "query_rewrite": StageTimingExtractor(
                stage="query_rewrite",
                strategy=TimingExtractionStrategy.CALCULATED,
                calculate_expression="50 if query_rewrite else 0",
            ),
            "faq_match": StageTimingExtractor(
                stage="faq_match",
                strategy=TimingExtractionStrategy.CALCULATED,
                calculate_expression="30 if faq_match and faq_match.get('matched') else 0",
            ),
            "retrieval": StageTimingExtractor(
                stage="retrieval",
                strategy=TimingExtractionStrategy.CALCULATED,
                calculate_expression="len(retrieval) * 15 + 50",
            ),
            "rerank": StageTimingExtractor(
                stage="rerank",
                strategy=TimingExtractionStrategy.CALCULATED,
                calculate_expression="len(rerank) * 10 + 30",
            ),
            "generation": StageTimingExtractor(
                stage="generation",
                strategy=TimingExtractionStrategy.CALCULATED,
                calculate_expression="token_count * 2 + 200",
            ),
        },
    )


def get_mock_config() -> TimingExtractionConfig:
    """
    获取适用于 Mock 适配器的配置。
    使用自定义提取函数模拟各阶段耗时。
    """
    import random

    return TimingExtractionConfig(
        stages={
            "query_rewrite": StageTimingExtractor(
                stage="query_rewrite",
                strategy=TimingExtractionStrategy.CUSTOM,
                custom_extractor=lambda data: random.uniform(50, 150) if data.get("query_rewrite") else 0,
            ),
            "faq_match": StageTimingExtractor(
                stage="faq_match",
                strategy=TimingExtractionStrategy.CUSTOM,
                custom_extractor=lambda data: random.uniform(20, 80) if data.get("faq_match", {}).get("matched") else 0,
            ),
            "retrieval": StageTimingExtractor(
                stage="retrieval",
                strategy=TimingExtractionStrategy.CUSTOM,
                custom_extractor=lambda data: len(data.get("retrieval", [])) * random.uniform(10, 25) + random.uniform(30, 70),
            ),
            "rerank": StageTimingExtractor(
                stage="rerank",
                strategy=TimingExtractionStrategy.CUSTOM,
                custom_extractor=lambda data: len(data.get("rerank", [])) * random.uniform(8, 18) + random.uniform(20, 50),
            ),
            "generation": StageTimingExtractor(
                stage="generation",
                strategy=TimingExtractionStrategy.CUSTOM,
                custom_extractor=lambda data: data.get("llm_output", {}).get("token_usage", {}).get("total_tokens", 100) * random.uniform(1.5, 3) + random.uniform(100, 300),
            ),
        },
    )