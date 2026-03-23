"""
Prompt Template Manager.

Manages prompt templates for the RAG pipeline.
"""

import json
from pathlib import Path
from typing import Any, Optional

import yaml

from rag_rag.core.logging import get_logger

logger = get_logger("rag_rag.prompts")


class PromptTemplateManager:
    """
    Prompt Template Manager.

    Features:
    - Load templates from YAML files
    - Render templates with variables
    - Fallback to default template
    """

    def __init__(self, template_dir: Optional[str | Path] = None):
        self.template_dir = Path(template_dir) if template_dir else Path(__file__).parent / "templates"
        self._templates: dict[str, dict[str, Any]] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        """Load all templates from template directory."""
        if not self.template_dir.exists():
            logger.warning(f"Template directory not found: {self.template_dir}")
            return

        for template_file in self.template_dir.glob("*.yaml"):
            try:
                with open(template_file, encoding="utf-8") as f:
                    template = yaml.safe_load(f)
                    name = template.get("name", template_file.stem)
                    self._templates[name] = template
                    logger.debug(f"Loaded template: {name}")
            except Exception as e:
                logger.warning(f"Failed to load template {template_file}: {e}")

        logger.info(f"Loaded {len(self._templates)} prompt templates")

    def get_template(self, name: str) -> Optional[dict[str, Any]]:
        """Get a template by name."""
        return self._templates.get(name)

    def render(
        self,
        template_name: str,
        context: str = "",
        query: str = "",
        conversation_history: str = "",
        domain: str = "企业知识库",
        **kwargs: Any,
    ) -> tuple[str, str]:
        """
        Render a prompt template.

        Args:
            template_name: Name of the template
            context: Retrieved context
            query: User query
            conversation_history: Previous turns
            domain: Knowledge domain
            **kwargs: Additional variables

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        template = self._templates.get(template_name)

        if not template:
            # Fallback to default
            template = self._templates.get("default", {})

        system_template = template.get("system", "")
        user_template = template.get("user", "")

        # Render system prompt
        system_prompt = system_template.format(domain=domain, **kwargs)

        # Render user prompt
        user_prompt = user_template.format(
            context=context,
            query=query,
            conversation_history=conversation_history,
            domain=domain,
            **kwargs,
        )

        return system_prompt, user_prompt

    def render_refusal(self, refusal_type: str) -> str:
        """Render a refusal response."""
        refusal_template = self._templates.get("refusal", {})
        templates = refusal_template.get("templates", {})
        return templates.get(refusal_type, "抱歉，我无法回答这个问题。")

    def list_templates(self) -> list[str]:
        """List available template names."""
        return list(self._templates.keys())

    def add_template(self, name: str, template: dict[str, Any]) -> None:
        """Add a custom template."""
        self._templates[name] = template

    def reload(self) -> None:
        """Reload templates from disk."""
        self._templates.clear()
        self._load_templates()


# Singleton instance
_template_manager: Optional[PromptTemplateManager] = None


def get_template_manager() -> PromptTemplateManager:
    """Get the template manager singleton."""
    global _template_manager
    if _template_manager is None:
        _template_manager = PromptTemplateManager()
    return _template_manager