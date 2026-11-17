"""Language-model clients. Anything with `generate_json(prompt) -> dict` can be plugged in."""
from __future__ import annotations

import json
import re
from typing import Callable, Optional


class LLMError(RuntimeError):
    pass


def parse_json_object(text: str) -> dict:
    """Parse a JSON object from model output, tolerating code fences and stray prose."""
    text = (text or "").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise LLMError("The model did not return JSON.")
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise LLMError("The model returned malformed JSON.") from exc
    if not isinstance(value, dict):
        raise LLMError("The model returned JSON that is not an object.")
    return value


class OllamaClient:
    """Calls a local Ollama server over HTTP. The model stays loaded between
    requests, unlike launching `ollama run` once per question."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.name = f"ollama:{model}"

    def available(self) -> bool:
        import requests

        try:
            return requests.get(f"{self.base_url}/api/tags", timeout=3).ok
        except requests.RequestException:
            return False

    def generate_json(self, prompt: str, temperature: float = 0.7) -> dict:
        import requests

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature},
        }
        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
        except requests.RequestException as exc:
            raise LLMError(
                f"Could not reach Ollama at {self.base_url}. Start it with 'ollama serve' "
                f"and make sure the model is pulled: 'ollama pull {self.model}'."
            ) from exc
        if response.status_code == 404:
            raise LLMError(f"Ollama does not have the model '{self.model}'. Run: ollama pull {self.model}")
        if not response.ok:
            raise LLMError(f"Ollama returned HTTP {response.status_code}: {response.text[:200]}")
        return parse_json_object(response.json().get("response", ""))


class FakeLLM:
    """Deterministic stand-in for tests: replies come from a function of the prompt."""

    name = "fake"

    def __init__(self, responder: Optional[Callable[[str], dict]] = None):
        self.responder = responder
        self.calls = 0

    def available(self) -> bool:
        return True

    def generate_json(self, prompt: str, temperature: float = 0.7) -> dict:
        self.calls += 1
        if self.responder:
            return self.responder(prompt)
        topic = re.search(r"^Topic: (.+)$", prompt, flags=re.M)
        verbs = re.search(r"fits the material best: (.+)$", prompt, flags=re.M)
        verb = verbs.group(1).split(",")[0].strip() if verbs else "explain"
        return {
            "question": f"{verb.capitalize()} {topic.group(1) if topic else 'the concept'} "
            f"in detail (variant {self.calls}).",
            "verb": verb,
        }
