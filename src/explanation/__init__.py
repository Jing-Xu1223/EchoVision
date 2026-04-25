"""Label → natural-language explanations (default: Google Gemini API)."""

from src.explanation.llm_explain import explain_from_labels

__all__ = ["explain_from_labels"]
