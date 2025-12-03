"""Base interface for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class GenerationResult:
    """Result of a text generation request."""
    prefix: str
    continuation: str
    full_response: str
    input_tokens: int
    output_tokens: int
    generation_time_ms: int
    model: str
    raw_response: Optional[dict] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate_with_prefix(
        self,
        prompt: str,
        prefix: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> GenerationResult:
        """
        Generate a response to a prompt, starting with the given prefix.

        Args:
            prompt: The user's question/prompt
            prefix: The forced beginning of the assistant's response
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            GenerationResult with the full response and metadata
        """
        pass

    @abstractmethod
    def evaluate_response(
        self,
        question: str,
        response: str,
        temperature: float = 0.0,
    ) -> dict:
        """
        Use the LLM as a judge to evaluate a response.

        Args:
            question: The original question
            response: The response to evaluate (without prefix visible)
            temperature: Sampling temperature (0 for deterministic)

        Returns:
            Dict with ratings on 7 dimensions plus justification
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name/identifier."""
        pass
