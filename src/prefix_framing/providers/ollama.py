"""Ollama provider for local LLM inference."""

import json
import time
from typing import Optional

import ollama

from .base import LLMProvider, GenerationResult


EVALUATION_PROMPT = """You are evaluating the quality of a response to the following question.

Question: {question}

Response to evaluate:
{response}

Please rate the response on the following dimensions (1-7 scale):

1. **Thoroughness:** How completely does it address the question? (1=superficial, 7=comprehensive)
2. **Accuracy:** How factually correct is the information? (1=major errors, 7=fully accurate)
3. **Insight:** Does it offer non-obvious perspectives or connections? (1=entirely obvious, 7=genuinely insightful)
4. **Clarity:** How well-organized and understandable is it? (1=confusing, 7=crystal clear)
5. **Engagement:** Does it seem like the author cared about the answer? (1=phoned in, 7=deeply engaged)
6. **Nuance:** Does it acknowledge complexity, edge cases, counterarguments? (1=oversimplified, 7=appropriately nuanced)
7. **Usefulness:** Would this response actually help someone understand the topic? (1=not helpful, 7=very helpful)

You MUST respond with ONLY a JSON object in this exact format, no other text:
{{"thoroughness": <int>, "accuracy": <int>, "insight": <int>, "clarity": <int>, "engagement": <int>, "nuance": <int>, "usefulness": <int>, "brief_justification": "<string>"}}"""


class OllamaProvider(LLMProvider):
    """Provider for local Ollama models."""

    def __init__(
        self,
        model: str = "llama3.2",
        host: Optional[str] = None,
    ):
        """
        Initialize the Ollama provider.

        Args:
            model: Model name (e.g., "llama3.2", "mistral", "qwen2.5")
            host: Optional Ollama host URL (defaults to localhost:11434)
        """
        self._model = model
        self._client = ollama.Client(host=host) if host else ollama.Client()

        # Verify model is available
        try:
            self._client.show(model)
        except ollama.ResponseError:
            available = [m["name"] for m in self._client.list()["models"]]
            raise ValueError(
                f"Model '{model}' not found. Available models: {available}. "
                f"Pull it with: ollama pull {model}"
            )

    @property
    def model_name(self) -> str:
        return self._model

    def generate_with_prefix(
        self,
        prompt: str,
        prefix: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> GenerationResult:
        """
        Generate a response with a forced prefix.

        Ollama's chat API doesn't directly support assistant prefill like Anthropic's,
        so we use a workaround: include the prefix as a partial assistant message
        and instruct the model to continue.
        """
        start_time = time.time()

        # Build messages
        messages = [{"role": "user", "content": prompt}]

        if prefix:
            # Add prefix as start of assistant response
            # The model will continue from here
            messages.append({"role": "assistant", "content": prefix})

        # Generate continuation
        response = self._client.chat(
            model=self._model,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )

        generation_time_ms = int((time.time() - start_time) * 1000)

        continuation = response["message"]["content"]

        # Full response = prefix + continuation
        # Add space between if prefix doesn't end with space and continuation doesn't start with space
        if prefix and continuation:
            if not prefix.endswith(" ") and not continuation.startswith(" "):
                full_response = prefix + " " + continuation
            else:
                full_response = prefix + continuation
        else:
            full_response = prefix + continuation

        return GenerationResult(
            prefix=prefix,
            continuation=continuation,
            full_response=full_response,
            input_tokens=response.get("prompt_eval_count", 0),
            output_tokens=response.get("eval_count", 0),
            generation_time_ms=generation_time_ms,
            model=self._model,
            raw_response=response,
        )

    def evaluate_response(
        self,
        question: str,
        response: str,
        temperature: float = 0.0,
    ) -> dict:
        """Use the LLM as a judge to evaluate a response."""
        eval_prompt = EVALUATION_PROMPT.format(
            question=question,
            response=response,
        )

        result = self._client.chat(
            model=self._model,
            messages=[{"role": "user", "content": eval_prompt}],
            options={
                "temperature": temperature,
                "num_predict": 500,
            },
            format="json",  # Request JSON output
        )

        # Parse the JSON response
        try:
            ratings = json.loads(result["message"]["content"])
            # Ensure all required fields are present
            required = [
                "thoroughness", "accuracy", "insight", "clarity",
                "engagement", "nuance", "usefulness", "brief_justification"
            ]
            for field in required:
                if field not in ratings:
                    if field == "brief_justification":
                        ratings[field] = ""
                    else:
                        ratings[field] = 4  # Default to middle rating
            return ratings
        except (json.JSONDecodeError, KeyError) as e:
            # Return default ratings if parsing fails
            return {
                "thoroughness": 4,
                "accuracy": 4,
                "insight": 4,
                "clarity": 4,
                "engagement": 4,
                "nuance": 4,
                "usefulness": 4,
                "brief_justification": f"Parse error: {e}",
            }

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count.

        Ollama doesn't expose a tokenizer directly, so we use a rough estimate.
        Most models use ~4 characters per token on average.
        """
        # Rough estimate: 4 chars per token
        return len(text) // 4
