"""
SSE (Server-Sent Events) parser for LangGraph streaming responses.
Parses the new SSE format with metadata, values, and custom events.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class SSEEventType(str, Enum):
    """Types of SSE events from LangGraph streaming."""

    METADATA = "metadata"  # Run metadata (run_id, attempt)
    VALUES = "values"  # State updates
    CUSTOM = "custom"  # UI content + intermediate results
    HEARTBEAT = "heartbeat"  # Keep-alive signal


@dataclass
class SSEEvent:
    """Parsed SSE event with type and data."""

    event_type: SSEEventType
    data: dict[str, Any] = field(default_factory=dict)
    raw_line: str = ""

    def is_heartbeat(self) -> bool:
        """Check if this is a heartbeat event."""
        return self.event_type == SSEEventType.HEARTBEAT


@dataclass
class ContentBlock:
    """UI display content block from custom events."""

    block_id: str
    block_type: str  # "reasonContent" or "text"
    title: str
    content: str
    is_show: bool = True
    is_done: bool = False
    raw_payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class IntermediateResult:
    """Intermediate result from a pipeline stage."""

    stage: str  # e.g., "query_rewrite", "faq_match", "retrieve", "rerank", "answer"
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None


class SSEEventParser:
    """
    Parser for SSE events from LangGraph streaming responses.

    Handles the new SSE format:
    - event: metadata -> run metadata
    - event: values -> state updates
    - event: custom -> UI content blocks and intermediate results
    - : (heartbeat) -> keep-alive signal
    """

    # Known paths for intermediate results in custom events
    INTERMEDIATE_RESULT_PATHS = {
        "query_rewrite": "faq_on_fail_query_rewrite._query_rewrite_result",
        "faq_match": "faq_on_fail_query_rewrite._faq_result",
        "retrieve": "retrieve._retrieve_result",
        "api_retrieve": "api_retrieve._external_retrieve_result",
        "rerank": "rerank._rerank_result",
        "prompt": "prompt._prompt_result",
        "answer": "answer._answer_result",
    }

    # Mapping from content block titles to stage names
    TITLE_TO_STAGE = {
        "意图识别": "intent",
        "意图解析": "intent_parse",
        "知识检索": "retrieval",
        "生成": "generation",
        "查询改写": "query_rewrite",
        "FAQ匹配": "faq_match",
        "重排序": "rerank",
        "回答生成": "answer",
    }

    def __init__(self):
        """Initialize the parser."""
        self._buffer = ""

    def parse_line(self, line: str) -> Optional[SSEEvent]:
        """
        Parse a single SSE line into an event.

        Args:
            line: Raw SSE line

        Returns:
            SSEEvent if valid, None for heartbeat or empty lines
        """
        line = line.strip()

        # Empty line - end of event
        if not line:
            return None

        # Heartbeat signal
        if line.startswith(":"):
            return SSEEvent(
                event_type=SSEEventType.HEARTBEAT,
                raw_line=line,
            )

        # Event type line
        if line.startswith("event: "):
            event_name = line[7:].strip()
            try:
                event_type = SSEEventType(event_name)
            except ValueError:
                # Unknown event type, treat as custom
                event_type = SSEEventType.CUSTOM
            return SSEEvent(
                event_type=event_type,
                raw_line=line,
            )

        # Data line
        if line.startswith("data: "):
            data_str = line[6:]
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                data = {"raw": data_str}

            return SSEEvent(
                event_type=SSEEventType.CUSTOM,  # Default, will be updated by event line
                data=data,
                raw_line=line,
            )

        return None

    def parse_event_batch(self, lines: list[str]) -> list[SSEEvent]:
        """
        Parse a batch of SSE lines into events.

        SSE events are separated by blank lines, with event type
        and data on separate lines.

        Args:
            lines: List of raw SSE lines

        Returns:
            List of parsed SSEEvent objects
        """
        events = []
        current_event: Optional[SSEEvent] = None

        for line in lines:
            parsed = self.parse_line(line)

            if parsed is None:
                # Blank line - finalize current event
                if current_event and current_event.data:
                    events.append(current_event)
                    current_event = None
                continue

            if parsed.event_type in (SSEEventType.METADATA, SSEEventType.VALUES, SSEEventType.CUSTOM):
                if current_event and current_event.data:
                    events.append(current_event)
                current_event = parsed
            elif parsed.event_type == SSEEventType.HEARTBEAT:
                # Heartbeats don't need to be accumulated
                events.append(parsed)
            elif current_event is not None and line.startswith("data: "):
                # Data line for current event
                current_event.data.update(parsed.data)

        # Don't forget the last event
        if current_event and current_event.data:
            events.append(current_event)

        return events

    def parse_content_block(self, data: dict[str, Any]) -> Optional[ContentBlock]:
        """
        Parse a content block from a custom event.

        Args:
            data: Data dict from SSE event

        Returns:
            ContentBlock if valid, None otherwise
        """
        content = data.get("content", {})
        if not content:
            return None

        block_id = content.get("blockId", "")
        block_type = content.get("type", "")
        payload = content.get("payload", {})

        if not block_id or not block_type:
            return None

        return ContentBlock(
            block_id=block_id,
            block_type=block_type,
            title=payload.get("title", ""),
            content=payload.get("content", ""),
            is_show=payload.get("isShow", True),
            is_done=payload.get("isDone", False),
            raw_payload=payload,
        )

    def extract_intermediate_results(
        self, data: dict[str, Any]
    ) -> list[IntermediateResult]:
        """
        Extract intermediate results from a custom event data.

        Looks for known nested paths and extracts results.

        Args:
            data: Data dict from SSE event

        Returns:
            List of IntermediateResult objects
        """
        results = []

        for stage, path in self.INTERMEDIATE_RESULT_PATHS.items():
            result_data = self._get_nested(data, path)
            if result_data is not None:
                results.append(IntermediateResult(
                    stage=stage,
                    data=result_data,
                ))

        return results

    def get_stage_from_title(self, title: str) -> str:
        """
        Map content block title to stage name.

        Args:
            title: Content block title (e.g., "意图识别")

        Returns:
            Stage name (e.g., "intent")
        """
        return self.TITLE_TO_STAGE.get(title, title.lower().replace(" ", "_"))

    def _get_nested(self, data: dict[str, Any], path: str) -> Any:
        """
        Get a nested value from a dict using dot notation.

        Args:
            data: Source dictionary
            path: Dot-separated path (e.g., "faq_on_fail_query_rewrite._query_rewrite_result")

        Returns:
            Value at path, or None if not found
        """
        if not data or not path:
            return None

        keys = path.split(".")
        value = data

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None

        return value


class SSEStreamAccumulator:
    """
    Accumulates state from SSE events across a streaming session.

    Collects:
    - Run metadata
    - State updates (query, messages, etc.)
    - Intermediate results from all stages
    - Content blocks for UI display
    """

    def __init__(self):
        """Initialize empty accumulator."""
        self._state: dict[str, Any] = {
            "run_id": None,
            "attempt": None,
            "query": None,
            "messages": [],
            "routing_result": None,
            "intent_result": None,
            # Intermediate results
            "query_rewrite_result": None,
            "faq_result": None,
            "retrieve_result": None,
            "external_retrieve_result": None,
            "rerank_result": None,
            "prompt_result": None,
            "answer_result": None,
            # UI content blocks
            "content_blocks": [],
            # Timing
            "event_timestamps": {},
            "start_time": None,
            "end_time": None,
        }

    def accumulate(self, event: SSEEvent) -> None:
        """
        Accumulate data from an SSE event.

        Args:
            event: Parsed SSE event
        """
        import time

        if event.event_type == SSEEventType.METADATA:
            self._accumulate_metadata(event.data)

        elif event.event_type == SSEEventType.VALUES:
            self._accumulate_values(event.data)

        elif event.event_type == SSEEventType.CUSTOM:
            self._accumulate_custom(event.data)

        # Record timestamp
        if event.event_type != SSEEventType.HEARTBEAT:
            self._state["event_timestamps"][event.event_type.value] = time.time()

    def _accumulate_metadata(self, data: dict[str, Any]) -> None:
        """Accumulate metadata event data."""
        if "run_id" in data:
            self._state["run_id"] = data["run_id"]
        if "attempt" in data:
            self._state["attempt"] = data["attempt"]
        if self._state["start_time"] is None:
            import time
            self._state["start_time"] = time.time()

    def _accumulate_values(self, data: dict[str, Any]) -> None:
        """Accumulate values (state update) event data."""
        # Core state fields
        if "query" in data:
            self._state["query"] = data["query"]

        if "messages" in data:
            self._state["messages"] = data["messages"]

        if "routing_result" in data:
            self._state["routing_result"] = data["routing_result"]

        if "intent_result" in data:
            self._state["intent_result"] = data["intent_result"]

    def _accumulate_custom(self, data: dict[str, Any]) -> None:
        """Accumulate custom event data (intermediate results + content blocks)."""
        parser = SSEEventParser()

        # Extract intermediate results
        intermediate_results = parser.extract_intermediate_results(data)

        for result in intermediate_results:
            key = f"{result.stage}_result"
            if key in self._state:
                self._state[key] = result.data

        # Extract content blocks
        content_block = parser.parse_content_block(data)
        if content_block:
            self._state["content_blocks"].append(content_block)

    def get_state(self) -> dict[str, Any]:
        """
        Get the accumulated state.

        Returns:
            Copy of accumulated state dictionary
        """
        import time
        import copy

        state = copy.deepcopy(self._state)
        state["end_time"] = time.time()
        return state

    def get_query(self) -> Optional[str]:
        """Get the accumulated query."""
        return self._state.get("query")

    def get_answer_result(self) -> Optional[dict[str, Any]]:
        """Get the answer result."""
        return self._state.get("answer_result")

    def get_retrieve_result(self) -> Optional[dict[str, Any]]:
        """Get the retrieve result."""
        return self._state.get("retrieve_result")

    def get_rerank_result(self) -> Optional[dict[str, Any]]:
        """Get the rerank result."""
        return self._state.get("rerank_result")

    def get_faq_result(self) -> Optional[dict[str, Any]]:
        """Get the FAQ result."""
        return self._state.get("faq_result")

    def get_query_rewrite_result(self) -> Optional[dict[str, Any]]:
        """Get the query rewrite result."""
        return self._state.get("query_rewrite_result")

    def reset(self) -> None:
        """Reset the accumulator for a new session."""
        self.__init__()