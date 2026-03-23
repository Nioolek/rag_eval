"""
Sensitive Word Filter Implementation.

Content safety filtering for sensitive words and topics.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from rag_rag.core.exceptions import ValidationError
from rag_rag.core.logging import get_logger

logger = get_logger("rag_rag.services.filter")


@dataclass
class FilterConfig:
    """Sensitive filter configuration."""

    enabled: bool = True
    custom_words: list[str] = field(default_factory=list)
    custom_patterns: list[str] = field(default_factory=list)
    word_file: Optional[str] = None


class SensitiveFilter:
    """
    Sensitive Word Filter.

    Features:
    - Word-based filtering
    - Regex pattern matching
    - Custom word lists
    - Configurable actions (mask, reject, warn)
    """

    # Default sensitive patterns (examples - should be customized for production)
    DEFAULT_PATTERNS = [
        # PII patterns
        r"\b\d{17}[\dXx]\b",  # Chinese ID card
        r"\b\d{15}\b",  # Old Chinese ID
        r"\b1[3-9]\d{9}\b",  # Phone number
        r"\b[\w.-]+@[\w.-]+\.\w+\b",  # Email
    ]

    # Common sensitive words (examples - should be customized)
    DEFAULT_WORDS = [
        # Add sensitive words as needed
    ]

    def __init__(self, config: FilterConfig):
        self.config = config
        self._words: set[str] = set()
        self._patterns: list[re.Pattern] = []

    async def initialize(self) -> None:
        """Initialize the filter with word lists and patterns."""
        # Load default words
        self._words.update(self.DEFAULT_WORDS)

        # Load custom words
        if self.config.custom_words:
            self._words.update(self.config.custom_words)

        # Load words from file
        if self.config.word_file:
            await self._load_words_from_file(self.config.word_file)

        # Compile default patterns
        for pattern in self.DEFAULT_PATTERNS:
            try:
                self._patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                logger.warning(f"Invalid pattern: {pattern} - {e}")

        # Compile custom patterns
        for pattern in self.config.custom_patterns:
            try:
                self._patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                logger.warning(f"Invalid custom pattern: {pattern} - {e}")

        logger.info(
            f"Sensitive Filter initialized: {len(self._words)} words, "
            f"{len(self._patterns)} patterns"
        )

    async def close(self) -> None:
        """Close the filter."""
        self._words.clear()
        self._patterns.clear()

    async def _load_words_from_file(self, file_path: str) -> None:
        """Load sensitive words from a file."""
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"Word file not found: {file_path}")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip()
                    if word and not word.startswith("#"):
                        self._words.add(word.lower())
        except Exception as e:
            logger.error(f"Failed to load word file: {e}")

    def check(self, text: str) -> dict[str, Any]:
        """
        Check text for sensitive content.

        Args:
            text: Text to check

        Returns:
            Dict with is_sensitive, matches, and action
        """
        if not self.config.enabled:
            return {
                "is_sensitive": False,
                "matches": [],
                "action": "pass",
            }

        matches = []

        # Check words
        text_lower = text.lower()
        for word in self._words:
            if word in text_lower:
                # Find all occurrences
                start = 0
                while True:
                    pos = text_lower.find(word, start)
                    if pos == -1:
                        break
                    matches.append({
                        "type": "word",
                        "value": word,
                        "position": pos,
                    })
                    start = pos + 1

        # Check patterns
        for pattern in self._patterns:
            for match in pattern.finditer(text):
                matches.append({
                    "type": "pattern",
                    "value": match.group(),
                    "position": match.start(),
                })

        is_sensitive = len(matches) > 0

        return {
            "is_sensitive": is_sensitive,
            "matches": matches,
            "action": "reject" if is_sensitive else "pass",
        }

    def mask(self, text: str, mask_char: str = "*") -> str:
        """
        Mask sensitive content in text.

        Args:
            text: Text to mask
            mask_char: Character to use for masking

        Returns:
            Masked text
        """
        if not self.config.enabled:
            return text

        result = text

        # Mask words
        text_lower = text.lower()
        for word in self._words:
            if word in text_lower:
                # Find and replace
                start = 0
                while True:
                    pos = text_lower.find(word, start)
                    if pos == -1:
                        break
                    result = (
                        result[:pos]
                        + mask_char * len(word)
                        + result[pos + len(word) :]
                    )
                    start = pos + len(word)

        # Mask patterns
        for pattern in self._patterns:
            result = pattern.sub(mask_char * 4, result)

        return result

    def is_sensitive(self, text: str) -> bool:
        """
        Quick check if text contains sensitive content.

        Args:
            text: Text to check

        Returns:
            True if sensitive
        """
        result = self.check(text)
        return result["is_sensitive"]

    def add_word(self, word: str) -> None:
        """Add a sensitive word."""
        self._words.add(word.lower())

    def remove_word(self, word: str) -> None:
        """Remove a sensitive word."""
        self._words.discard(word.lower())

    def add_pattern(self, pattern: str) -> bool:
        """
        Add a regex pattern.

        Args:
            pattern: Regex pattern string

        Returns:
            True if added successfully
        """
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
            self._patterns.append(compiled)
            return True
        except re.error as e:
            logger.warning(f"Invalid pattern: {pattern} - {e}")
            return False

    def get_stats(self) -> dict[str, int]:
        """Get filter statistics."""
        return {
            "word_count": len(self._words),
            "pattern_count": len(self._patterns),
        }