"""Answer generation.

``ExtractiveGenerator`` needs no LLM: it selects the passages' most relevant
sentences. The other generators call an LLM over plain HTTP (stdlib only) and
answer in the language of the question, citing sources as [1], [2], ...
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from rag_engine.text import language_name, split_sentences

log = logging.getLogger(__name__)


@dataclass
class Passage:
    text: str
    title: str
    score: float


class GenerationError(RuntimeError):
    pass


class Generator(Protocol):
    name: str

    def generate(self, question: str, language: str, passages: list[Passage]) -> str: ...


SYSTEM_PROMPT = (
    "You are a precise assistant answering questions from a multilingual knowledge base. "
    "Use ONLY the numbered context passages. The passages may be in a different language "
    "from the question; translate as needed. Cite the passages you use with their numbers "
    "in square brackets, e.g. [1] or [2][3]. If the context does not contain the answer, "
    "say so plainly instead of guessing. Answer in {language}."
)


def build_user_prompt(question: str, passages: list[Passage]) -> str:
    ctx = "\n\n".join(f"[{i}] ({p.title})\n{p.text}" for i, p in enumerate(passages, 1))
    return f"Context passages:\n\n{ctx}\n\nQuestion: {question}"


NO_CONTEXT = {
    "en": "I couldn't find anything in the knowledge base that answers this question.",
    "sw": "Sikupata taarifa yoyote kwenye hifadhi ya maarifa inayojibu swali hili.",
    "ru": "В базе знаний не найдено информации, отвечающей на этот вопрос.",
    "uk": "У базі знань не знайдено інформації, що відповідає на це питання.",
    "fr": "Je n'ai rien trouvé dans la base de connaissances qui réponde à cette question.",
    "es": "No encontré nada en la base de conocimiento que responda a esta pregunta.",
    "de": "In der Wissensdatenbank wurde nichts gefunden, das diese Frage beantwortet.",
}


def no_context_answer(language: str) -> str:
    return NO_CONTEXT.get(language, NO_CONTEXT["en"])


def _post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise GenerationError(f"LLM request failed ({exc.code}): {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise GenerationError(f"LLM endpoint unreachable at {url}: {exc}") from exc


class ExtractiveGenerator:
    """Pick the sentences most similar to the question, keep their source numbers."""

    name = "extractive"

    def __init__(self, embedder, max_sentences: int = 3, relative_margin: float = 0.35):
        self.embedder = embedder
        self.max_sentences = max_sentences
        self.relative_margin = relative_margin

    def generate(self, question: str, language: str, passages: list[Passage]) -> str:
        candidates: list[tuple[int, int, str]] = []  # (passage no., order, sentence)
        for pno, p in enumerate(passages, 1):
            for order, s in enumerate(split_sentences(p.text)):
                if len(s) > 15:
                    candidates.append((pno, order, s))
        if not candidates:
            return passages[0].text if passages else no_context_answer(language)
        vecs = self.embedder.embed([question] + [c[2] for c in candidates])
        sims = vecs[1:] @ vecs[0]
        # Favour sentences from better-ranked passages a little.
        sims = sims - 0.02 * np.array([c[0] - 1 for c in candidates])
        order = np.argsort(-sims)
        top = float(sims[order[0]])
        # Keep only sentences reasonably close to the best one, so weak matches don't pad the answer.
        chosen = [i for i in order[: self.max_sentences] if sims[i] >= top - self.relative_margin * abs(top)]
        best = sorted(chosen, key=lambda i: (candidates[i][0], candidates[i][1]))
        return " ".join(f"{candidates[i][2]} [{candidates[i][0]}]" for i in best)


class OpenAIGenerator:
    """Any OpenAI-compatible Chat Completions endpoint (OpenAI, Azure proxies, vLLM, LM Studio, ...)."""

    def __init__(self, model: str, api_key: str, base_url: str = "", temperature: float = 0.2,
                 max_tokens: int = 600, timeout: float = 60.0):
        self.model = model or "gpt-4o-mini"
        self.api_key = api_key
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.temperature, self.max_tokens, self.timeout = temperature, max_tokens, timeout
        self.name = f"openai:{self.model}"

    def generate(self, question: str, language: str, passages: list[Passage]) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data = _post_json(f"{self.base_url}/chat/completions", {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT.format(language=language_name(language))},
                {"role": "user", "content": build_user_prompt(question, passages)},
            ],
        }, headers, self.timeout)
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise GenerationError(f"Unexpected OpenAI response: {str(data)[:300]}") from exc


class AnthropicGenerator:
    def __init__(self, model: str, api_key: str, base_url: str = "", temperature: float = 0.2,
                 max_tokens: int = 600, timeout: float = 60.0):
        if not api_key:
            raise ValueError("Anthropic provider needs RAG_LLM_API_KEY or ANTHROPIC_API_KEY")
        self.model = model or "claude-haiku-5-5"
        self.api_key = api_key
        self.base_url = (base_url or "https://api.anthropic.com").rstrip("/")
        self.temperature, self.max_tokens, self.timeout = temperature, max_tokens, timeout
        self.name = f"anthropic:{self.model}"

    def generate(self, question: str, language: str, passages: list[Passage]) -> str:
        data = _post_json(f"{self.base_url}/v1/messages", {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": SYSTEM_PROMPT.format(language=language_name(language)),
            "messages": [{"role": "user", "content": build_user_prompt(question, passages)}],
        }, {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}, self.timeout)
        try:
            return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text").strip()
        except (KeyError, TypeError) as exc:
            raise GenerationError(f"Unexpected Anthropic response: {str(data)[:300]}") from exc


class OllamaGenerator:
    def __init__(self, model: str, base_url: str = "", temperature: float = 0.2,
                 max_tokens: int = 600, timeout: float = 120.0):
        self.model = model or "llama3.1"
        self.base_url = (base_url or "http://localhost:11434").rstrip("/")
        self.temperature, self.max_tokens, self.timeout = temperature, max_tokens, timeout
        self.name = f"ollama:{self.model}"

    def generate(self, question: str, language: str, passages: list[Passage]) -> str:
        data = _post_json(f"{self.base_url}/api/chat", {
            "model": self.model,
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT.format(language=language_name(language))},
                {"role": "user", "content": build_user_prompt(question, passages)},
            ],
        }, {}, self.timeout)
        try:
            return data["message"]["content"].strip()
        except (KeyError, TypeError) as exc:
            raise GenerationError(f"Unexpected Ollama response: {str(data)[:300]}") from exc


def build_generator(settings, embedder) -> Generator:
    p = settings.llm_provider
    common = dict(temperature=settings.llm_temperature, max_tokens=settings.llm_max_tokens,
                  timeout=settings.llm_timeout)
    if p == "openai":
        return OpenAIGenerator(settings.llm_model, settings.llm_api_key, settings.llm_base_url, **common)
    if p == "anthropic":
        return AnthropicGenerator(settings.llm_model, settings.llm_api_key, settings.llm_base_url, **common)
    if p == "ollama":
        return OllamaGenerator(settings.llm_model, settings.llm_base_url, **common)
    return ExtractiveGenerator(embedder)
