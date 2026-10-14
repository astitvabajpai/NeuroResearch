"""
LLM layer — Groq primary, OpenRouter fallback.

Rotation strategy:
  1. Try the requested model on Groq
  2. If Groq is rate-limited or unavailable, try other Groq models
  3. If ALL Groq models fail, try the same model family on OpenRouter
  4. OpenRouter has 100+ models and separate rate limits — acts as a deep fallback

Model naming:
  - Groq models:       "openai/gpt-oss-20b", "openai/gpt-oss-120b"
  - OpenRouter models: "meta-llama/llama-3.3-70b-instruct:free",
                       "mistralai/mistral-7b-instruct:free", etc.
  Models prefixed with "or:" are sent to OpenRouter directly.
  All others go to Groq first.
"""
from __future__ import annotations
import re
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Model registry ─────────────────────────────────────────────────────────────

# Models shown in the UI — Groq models first, then OpenRouter free models
AVAILABLE_MODELS = [
    # Groq
    {"id": "openai/gpt-oss-20b",                           "name": "GPT OSS 20B (Groq)",          "tag": "Recommended",   "provider": "groq"},
    {"id": "openai/gpt-oss-120b",                          "name": "GPT OSS 120B (Groq)",         "tag": "Most Powerful", "provider": "groq"},
    # OpenRouter free models (verified working)
    {"id": "nvidia/nemotron-3-ultra-550b-a55b:free",        "name": "Nemotron 3 Ultra 550B (OpenRouter)", "tag": "Most Powerful", "provider": "openrouter"},
    {"id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", "name": "Nemotron 3 Nano 30B (OpenRouter)", "tag": "Reasoning", "provider": "openrouter"},
    {"id": "nvidia/nemotron-3.5-lightning:free",            "name": "Nemotron 3.5 Lightning (OpenRouter)", "tag": "Fast",         "provider": "openrouter"},
    {"id": "poolside/laguna-s-2.1:free",                    "name": "Laguna S 2.1 (OpenRouter)",           "tag": "Free",         "provider": "openrouter"},
    {"id": "openrouter/free",                               "name": "Auto Free Router (OpenRouter)",        "tag": "Auto",         "provider": "openrouter"},
]

DEFAULT_MODEL        = "openai/gpt-oss-20b"
DEFAULT_WRITER_MODEL = "openai/gpt-oss-20b"
DEFAULT_CRITIC_MODEL = "openai/gpt-oss-120b"  # different from writer — reduces self-preference bias

# Groq-only rotation (tried in order on Groq rate-limit)
_GROQ_MODELS = [m["id"] for m in AVAILABLE_MODELS if m["provider"] == "groq"]

# OpenRouter fallback rotation (tried after all Groq models fail)
_OR_FALLBACK_MODELS = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3.5-lightning:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "poolside/laguna-s-2.1:free",
    "openrouter/free",
]

GROQ_API_URL       = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _strip_thinking(text: str) -> str:
    """Strip <think>...</think> blocks from reasoning models."""
    if "</think>" in text:
        after = text.split("</think>")[-1].strip()
        if after:
            return after
        inner = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        return inner.group(1).strip() if inner else text.strip()
    elif "<think>" in text:
        return text[:text.index("<think>")].strip()
    return text.strip()


def _is_openrouter_model(model_id: str) -> bool:
    """Return True if this model should be routed to OpenRouter."""
    model_info = next((m for m in AVAILABLE_MODELS if m["id"] == model_id), None)
    if model_info:
        return model_info["provider"] == "openrouter"
    # If not in registry but contains known OR patterns, route to OR
    return ":free" in model_id or "/" in model_id and model_id.split("/")[0] in (
        "meta-llama", "mistralai", "google", "microsoft", "anthropic",
        "openai", "cohere", "nousresearch"
    ) and model_id not in _GROQ_MODELS


def _get_groq_token() -> Optional[str]:
    try:
        from src.config.settings import get_settings
        s = get_settings()
        return (s.GROQ_API_KEY or s.GROQAPIKEY or "").strip() or None
    except Exception:
        return None


def _get_openrouter_token() -> Optional[str]:
    try:
        from src.config.settings import get_settings
        return get_settings().OPENROUTER_API_KEY.strip() or None
    except Exception:
        return None


# ── API callers ────────────────────────────────────────────────────────────────

def _call_endpoint(
    url: str,
    model_id: str,
    messages: list,
    max_tokens: int,
    temperature: float,
    token: str,
    extra_headers: Optional[dict] = None,
) -> Optional[str]:
    """Generic OpenAI-compatible chat completion call."""
    import requests as _requests
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "NeuroResearch/1.0 (Python; +https://github.com/astitvabajpai/NeuroResearch)",
    }
    if extra_headers:
        headers.update(extra_headers)

    logger.info("[LLM] %s → %s (max_tokens=%d)", url.split("/")[2], model_id, max_tokens)
    try:
        resp = _requests.post(
            url,
            headers=headers,
            json={
                "model":       model_id,
                "messages":    messages,
                "max_tokens":  max_tokens,
                "temperature": temperature,
                "stream":      False,
            },
            timeout=60,
        )
        if resp.ok:
            text    = resp.json()["choices"][0]["message"]["content"]
            cleaned = _strip_thinking(text)
            if not cleaned.strip():
                logger.warning("[LLM] %s returned empty after think-strip", model_id)
                return None
            logger.info("[LLM] %s → %d chars", model_id, len(cleaned))
            return cleaned
        if resp.status_code == 429:
            logger.warning("[LLM] %s rate-limited", model_id)
            return None
        if resp.status_code in (400, 404):
            logger.warning("[LLM] %s not available (%s)", model_id, resp.status_code)
            return None
        logger.warning("[LLM] %s returned HTTP %s", model_id, resp.status_code)
        return None
    except Exception as exc:
        logger.warning("[LLM] %s error: %s", model_id, exc)
        return None


def _call_groq(model_id: str, messages: list, max_tokens: int,
               temperature: float) -> Optional[str]:
    token = _get_groq_token()
    if not token:
        return None
    return _call_endpoint(GROQ_API_URL, model_id, messages, max_tokens, temperature, token)


def _call_openrouter(model_id: str, messages: list, max_tokens: int,
                     temperature: float) -> Optional[str]:
    token = _get_openrouter_token()
    if not token:
        return None
    return _call_endpoint(
        OPENROUTER_API_URL, model_id, messages, max_tokens, temperature, token,
        extra_headers={
            "HTTP-Referer": "https://github.com/astitvabajpai/NeuroResearch",
            "X-Title":      "NeuroResearch",
        },
    )


# ── Main LLM factory ──────────────────────────────────────────────────────────

def get_llm(model_id: str | None = None, deep: bool = False):
    """
    Return a LangChain-compatible chat LLM.

    Rotation order:
      1. Requested model on its native provider (Groq or OpenRouter)
      2. Other Groq models (if Groq was primary)
      3. OpenRouter fallback models (if all Groq models exhausted)
    """
    from src.config.settings import get_settings
    settings   = get_settings()
    model_id   = model_id or settings.HF_MODEL_ID or DEFAULT_MODEL
    max_tokens = 3000 if deep else 2000

    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from typing import Any, List

    _mid  = model_id
    _maxt = max_tokens

    class _MultiProviderLLM(BaseChatModel):
        model:          str   = _mid
        max_new_tokens: int   = _maxt
        temperature:    float = 0.7

        class Config:
            arbitrary_types_allowed = True

        @property
        def _llm_type(self) -> str:
            return "multi_provider"

        def _generate(
            self,
            messages: List[BaseMessage],
            stop=None,
            **kwargs: Any,
        ) -> ChatResult:
            # Convert LangChain messages → OpenAI format
            chat_msgs = []
            for m in messages:
                if isinstance(m, SystemMessage):
                    chat_msgs.append({"role": "system",    "content": m.content})
                elif isinstance(m, HumanMessage):
                    chat_msgs.append({"role": "user",      "content": m.content})
                elif isinstance(m, AIMessage):
                    chat_msgs.append({"role": "assistant", "content": m.content})
                else:
                    chat_msgs.append({"role": "user",      "content": str(m.content)})

            text = self._try_all(chat_msgs)
            if text is None:
                raise RuntimeError(
                    "All LLM providers are rate-limited or unavailable. "
                    "Wait 1 minute and try again."
                )
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

        def _try_all(self, chat_msgs: list) -> Optional[str]:
            """Try Groq first, then OpenRouter fallback."""

            # ── Phase 1: try on native provider ──────────────────────────
            if _is_openrouter_model(self.model):
                text = _call_openrouter(self.model, chat_msgs, self.max_new_tokens, self.temperature)
                if text:
                    return text
                logger.info("[LLM] OpenRouter model %s failed, trying Groq", self.model)
            else:
                # Groq primary
                text = _call_groq(self.model, chat_msgs, self.max_new_tokens, self.temperature)
                if text:
                    return text
                logger.info("[LLM] Groq model %s failed, trying other Groq models", self.model)

            # ── Phase 2: try all other Groq models ───────────────────────
            for candidate in _GROQ_MODELS:
                if candidate == self.model:
                    continue
                time.sleep(0.5)
                text = _call_groq(candidate, chat_msgs, self.max_new_tokens, self.temperature)
                if text:
                    logger.info("[LLM] Groq fallback used: %s", candidate)
                    return text

            # ── Phase 3: OpenRouter fallback ─────────────────────────────
            or_token = _get_openrouter_token()
            if or_token:
                logger.info("[LLM] All Groq models failed — trying OpenRouter fallback")
                for candidate in _OR_FALLBACK_MODELS:
                    time.sleep(0.5)
                    text = _call_openrouter(candidate, chat_msgs, self.max_new_tokens, self.temperature)
                    if text:
                        logger.info("[LLM] OpenRouter fallback used: %s", candidate)
                        return text
            else:
                logger.info("[LLM] OpenRouter key not set — skipping OR fallback")

            return None

    return _MultiProviderLLM()


# Backward-compatible alias
def get_hf_llm(model_id: str | None = None, deep: bool = False):
    return get_llm(model_id=model_id, deep=deep)


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    from src.config.settings import get_settings
    return HuggingFaceEmbeddings(model_name=get_settings().EMBEDDING_MODEL)
