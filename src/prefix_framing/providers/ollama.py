"""Ollama provider for local LLM inference."""

import json
import time
import os
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
        self._debug = os.getenv("PF_OLLAMA_DEBUG", "").lower() in {"1", "true", "yes", "debug"}

        # Verify model is available
        try:
            self._client.show(model)
        except ollama.ResponseError:
            # Handle different ollama API response structures
            try:
                list_response = self._client.list()
                # Could be dict with "models" key or object with models attribute
                if isinstance(list_response, dict):
                    models_list = list_response.get("models", [])
                else:
                    models_list = getattr(list_response, "models", [])

                # Extract model names - handle both dict and object formats
                available = []
                for m in models_list:
                    if isinstance(m, dict):
                        name = m.get("name") or m.get("model") or str(m)
                    else:
                        name = getattr(m, "name", None) or getattr(m, "model", None) or str(m)
                    available.append(name)

                # Try to find a matching model with a tag (e.g., "gpt-oss" -> "gpt-oss:20b")
                matching = [m for m in available if m.startswith(model + ":") or m == model]
                if matching:
                    # Use the first matching model
                    self._model = matching[0]
                    print(f"Note: Using '{self._model}' (matched from '{model}')")
                    return  # Successfully found a match

            except Exception:
                available = ["(unable to list models)"]

            raise ValueError(
                f"Model '{model}' not found. Available models: {available}. "
                f"Pull it with: ollama pull {model}"
            )

    def _debug_log(self, msg: str):
        if self._debug:
            print(f"[ollama-debug] {msg}", flush=True)

    @staticmethod
    def _truncate(text: str, limit: int = 400) -> str:
        if text is None:
            return ""
        if len(text) <= limit:
            return text
        return text[:limit] + "... [truncated]"

    def _build_prefilled_prompt(self, prompt: str, prefix: str) -> str:
        """
        Build a single-turn prompt that forces the assistant to start with the prefix.

        Using the /api/generate endpoint keeps us compatible with models like gpt-oss
        that don't reliably handle the chat API.
        """
        if prefix:
            return (
                "You are answering the question below. Begin your answer exactly with the provided "
                "prefix, then continue naturally.\n\n"
                f"Question:\n{prompt}\n\n"
                f"Prefix:\n{prefix}\n\n"
                f"Answer starting with the prefix:\n{prefix}"
            )
        # No prefix: simple instruction to answer the question
        return f"{prompt}\n\nAnswer:"

    @staticmethod
    def _extract_text(response) -> str:
        """Normalize text extraction from Ollama generate/chat responses."""
        if response is None:
            return ""

        # Dict-like responses
        if isinstance(response, dict):
            if "response" in response:
                return response.get("response") or ""
            if "content" in response:
                return response.get("content") or ""
            if "message" in response and isinstance(response["message"], dict):
                return response["message"].get("content") or ""
            if "choices" in response and response["choices"]:
                choice = response["choices"][0]
                if isinstance(choice, dict):
                    return choice.get("content") or choice.get("text") or ""
            return ""

        # Pydantic response objects (GenerateResponse/ChatResponse)
        text = getattr(response, "response", None)
        if not text and hasattr(response, "message"):
            message = getattr(response, "message")
            text = getattr(message, "content", None) or ""
        # Some integrations may expose .content directly
        if not text and hasattr(response, "content"):
            text = getattr(response, "content")
        return text or ""

    @staticmethod
    def _get_token_counts(response) -> tuple[int, int]:
        """Extract token usage fields from either dict or response object."""
        def _grab(obj, key):
            if isinstance(obj, dict):
                return obj.get(key, 0) or 0
            return getattr(obj, key, 0) or 0

        return _grab(response, "prompt_eval_count"), _grab(response, "eval_count")

    def _describe_response(self, resp):
        """Summarize response shape for debugging."""
        if resp is None:
            return "None"
        if isinstance(resp, dict):
            keys = list(resp.keys())
            return (
                f"dict keys={keys} "
                f"done_reason={resp.get('done_reason')} "
                f"eval_count={resp.get('eval_count')} "
                f"prompt_eval_count={resp.get('prompt_eval_count')}"
            )
        attrs = {k: getattr(resp, k, None) for k in [
            "done_reason", "eval_count", "prompt_eval_count", "response"
        ]}
        attrs["message_has_content"] = bool(
            getattr(getattr(resp, "message", None), "content", None)
        )
        return f"{resp.__class__.__name__} attrs={attrs}"

    def _stream_generate(self, prompt: str, temperature: float, max_tokens: int):
        """
        Stream generation and accumulate text.

        Some models (e.g., gpt-oss) occasionally return empty responses when
        called with stream=False. Streaming avoids that edge case.
        """
        chunks = self._client.generate(
            model=self._model,
            prompt=prompt,
            stream=True,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )

        full_text = []
        last_chunk = None
        for chunk in chunks:
            last_chunk = chunk
            if self._debug:
                self._debug_log(
                    f"stream chunk: {self._describe_response(chunk)} "
                    f"snippet={self._truncate(self._extract_text(chunk))}"
                )
            piece = self._extract_text(chunk)
            if piece:
                full_text.append(piece)

        return "".join(full_text), (last_chunk or {})

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> GenerationResult:
        """
        Generate a natural response to a prompt (no prefix forcing).

        Uses chat API for natural conversational responses.
        """
        start_time = time.time()

        self._debug_log(
            f"generate model={self._model} temp={temperature} "
            f"max_tokens={max_tokens} prompt_len={len(prompt)}"
        )

        # Use chat API for natural response
        response = self._client.chat(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )

        text = self._extract_text(response)
        self._debug_log(
            f"generate response_len={len(text)} "
            f"meta={self._describe_response(response)} "
            f"snippet={self._truncate(text)}"
        )

        # Fallback to streaming if empty
        if not text.strip():
            text, response = self._stream_generate(prompt, temperature, max_tokens)
            self._debug_log(
                f"generate stream fallback response_len={len(text)} "
                f"snippet={self._truncate(text)}"
            )

        if not text.strip():
            meta = self._describe_response(response)
            raise RuntimeError(
                f"Model '{self._model}' returned an empty response for prompt "
                f"(last_response={meta}). Set PF_OLLAMA_DEBUG=1 and retry for diagnostics."
            )

        generation_time_ms = int((time.time() - start_time) * 1000)
        input_tokens, output_tokens = self._get_token_counts(response)

        return GenerationResult(
            prefix="",
            continuation=text,
            full_response=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            generation_time_ms=generation_time_ms,
            model=self._model,
            raw_response=response,
        )

    def generate_with_prefix(
        self,
        prompt: str,
        prefix: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> GenerationResult:
        """
        Generate a response with a forced prefix.

        Uses the /api/generate endpoint so models that don't fully support chat
        (e.g., gpt-oss) still return a continuation.
        """
        start_time = time.time()

        # First attempt: /api/generate with a prefilled prompt
        prompt_text = self._build_prefilled_prompt(prompt, prefix)

        self._debug_log(
            f"generate_with_prefix model={self._model} temp={temperature} "
            f"max_tokens={max_tokens} prefix_len={len(prefix)} prompt_len={len(prompt)}"
        )

        response = self._client.generate(
            model=self._model,
            prompt=prompt_text,
            stream=False,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )

        continuation = self._extract_text(response)
        self._debug_log(
            f"primary generate response_len={len(continuation)} "
            f"meta={self._describe_response(response)} "
            f"snippet={self._truncate(continuation)}"
        )

        # Some models (notably gpt-oss) can return an empty string when called
        # via /api/generate with a prefill-style prompt. Fallback to chat with
        # an explicit instruction if that happens. If chat is also empty, retry
        # using a raw generate, then streaming generate which is more reliable on some builds.
        if not continuation.strip():
            # Try again with raw=True (bypass model template)
            raw_response = self._client.generate(
                model=self._model,
                prompt=prompt_text,
                stream=False,
                raw=True,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            )
            continuation = self._extract_text(raw_response)
            self._debug_log(
                f"raw generate response_len={len(continuation)} "
                f"meta={self._describe_response(raw_response)} "
                f"snippet={self._truncate(continuation)}"
            )
            if continuation.strip():
                response = raw_response

        if not continuation.strip():
            chat_prompt = (
                "Answer the user's question. Start your reply with the exact prefix provided, "
                "then continue normally.\n\n"
                f"Question: {prompt}\n"
                f"Prefix: {prefix}\n\n"
                "Begin your answer now, starting with the prefix."
            )
            response = self._client.chat(
                model=self._model,
                messages=[{"role": "user", "content": chat_prompt}],
                stream=False,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            )
            continuation = self._extract_text(response)
            self._debug_log(
                f"chat fallback response_len={len(continuation)} "
                f"meta={self._describe_response(response)} "
                f"snippet={self._truncate(continuation)}"
            )

            if not continuation.strip():
                # Streaming generate fallback
                continuation, response = self._stream_generate(
                    prompt_text, temperature, max_tokens
                )
                self._debug_log(
                    f"stream fallback response_len={len(continuation)} "
                    f"snippet={self._truncate(continuation)}"
                )

            if not continuation.strip():
                meta = self._describe_response(response)
                raise RuntimeError(
                    f"Model '{self._model}' returned an empty response for prompt "
                    f"(last_response={meta}). Set PF_OLLAMA_DEBUG=1 and retry for diagnostics."
                )

        generation_time_ms = int((time.time() - start_time) * 1000)

        # If the model echoed the prefix, strip it so continuation_only is clean
        continuation_only = continuation
        if prefix and continuation.startswith(prefix):
            continuation_only = continuation[len(prefix):].lstrip()

        # Full response = prefix + continuation
        # Add space between if prefix doesn't end with space and continuation doesn't start with space
        if prefix and continuation_only:
            if not prefix.endswith(" ") and not continuation_only.startswith(" "):
                full_response = prefix + " " + continuation_only
            else:
                full_response = prefix + continuation_only
        else:
            full_response = prefix + continuation_only

        input_tokens, output_tokens = self._get_token_counts(response)

        return GenerationResult(
            prefix=prefix,
            continuation=continuation_only,
            full_response=full_response,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            generation_time_ms=generation_time_ms,
            model=self._model,
            raw_response=response,
        )

    def evaluate_response(
        self,
        question: str,
        response: str,
        temperature: float = 0.0,
        max_retries: int = 3,
    ) -> dict:
        """Use the LLM as a judge to evaluate a response."""
        eval_prompt = EVALUATION_PROMPT.format(
            question=question,
            response=response,
        )

        self._debug_log(
            f"evaluate_response model={self._model} temp={temperature} prompt_len={len(eval_prompt)}"
        )

        last_error = None
        for attempt in range(max_retries):
            try:
                # Try with format="json" first
                result = self._client.generate(
                    model=self._model,
                    prompt=eval_prompt,
                    stream=False,
                    options={
                        "temperature": temperature,
                        "num_predict": 500,
                    },
                    format="json",
                )

                raw_text = self._extract_text(result)
                self._debug_log(
                    f"evaluate_response attempt={attempt+1} raw_len={len(raw_text or '')} "
                    f"snippet={self._truncate(raw_text)}"
                )

                # If empty, try without format="json" (some models don't support it)
                if not raw_text or not raw_text.strip():
                    self._debug_log("Empty response with format=json, retrying without format constraint")
                    result = self._client.generate(
                        model=self._model,
                        prompt=eval_prompt,
                        stream=False,
                        options={
                            "temperature": temperature,
                            "num_predict": 500,
                        },
                    )
                    raw_text = self._extract_text(result)
                    self._debug_log(
                        f"evaluate_response retry without format: raw_len={len(raw_text or '')} "
                        f"snippet={self._truncate(raw_text)}"
                    )

                if not raw_text or not raw_text.strip():
                    last_error = "Empty response from model"
                    continue

                # Try to extract JSON from the response (model might include extra text)
                json_text = raw_text.strip()
                # Look for JSON object in the response
                if not json_text.startswith("{"):
                    # Try to find JSON in the text
                    start_idx = json_text.find("{")
                    end_idx = json_text.rfind("}") + 1
                    if start_idx != -1 and end_idx > start_idx:
                        json_text = json_text[start_idx:end_idx]

                ratings = json.loads(json_text)
                self._debug_log(
                    f"evaluate_response parsed ratings keys={list(ratings.keys())}"
                )

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

                # Validate ratings are in range 1-7
                for field in required:
                    if field != "brief_justification":
                        val = ratings[field]
                        if not isinstance(val, (int, float)) or val < 1 or val > 7:
                            ratings[field] = 4

                return ratings

            except json.JSONDecodeError as e:
                last_error = f"JSON parse error: {e}"
                self._debug_log(f"evaluate_response attempt={attempt+1} failed: {last_error}")
            except Exception as e:
                last_error = f"Error: {e}"
                self._debug_log(f"evaluate_response attempt={attempt+1} failed: {last_error}")

        # Return default ratings if all retries failed
        self._debug_log(f"evaluate_response all {max_retries} attempts failed, returning defaults")
        return {
            "thoroughness": 4,
            "accuracy": 4,
            "insight": 4,
            "clarity": 4,
            "engagement": 4,
            "nuance": 4,
            "usefulness": 4,
            "brief_justification": f"Evaluation failed after {max_retries} attempts: {last_error}",
        }

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count.

        Ollama doesn't expose a tokenizer directly, so we use a rough estimate.
        Most models use ~4 characters per token on average.
        """
        # Rough estimate: 4 chars per token
        return len(text) // 4
