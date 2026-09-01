"""
LLM layer — Groq only.
All 5 models are free, confirmed working at 2048 tokens (deep research).
Falls back through the rotation automatically on rate limits.
"""
from __future__ import annotations
import re
import time
import logging

logger = logging.getLogger(__name__)

AVAILABLE_MODELS = [
    {"id": "openai/gpt-oss-20b",  "name": "GPT OSS 20B",   "tag": "Recommended"},
    {"id": "openai/gpt-oss-120b", "name": "GPT OSS 120B",  "tag": "Most Powerful"},
    {"id": "qwen/qwen3.6-27b",    "name": "Qwen 3.6 27B",  "tag": "Deep Reasoning"},
]

DEFAULT_MODEL        = "openai/gpt-oss-20b"
DEFAULT_WRITER_MODEL = "openai/gpt-oss-20b"
DEFAULT_CRITIC_MODEL = "openai/gpt-oss-20b"

# Rotation order — tried in sequence when primary model is unavailable
_ROTATION = [m["id"] for m in AVAILABLE_MODELS]

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def _strip_thinking(text: str) -> str:
    """
    Handle <think>...</think> blocks from reasoning models (Qwen 3.6, etc).
    
    Strategy:
    1. If </think> exists, take everything AFTER it (the actual answer)
    2. If only <think> with no closing tag, strip everything from <think> onward
    3. Otherwise return text as-is
    """
    if "</think>" in text:
        # Take content after the last </think> tag
        after = text.split("</think>")[-1].strip()
        if after:
            return after
        # Edge case: content is INSIDE think block only — extract it
        inner = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        return inner.group(1).strip() if inner else text.strip()
    elif "<think>" in text:
        # Unclosed think block — strip everything from <think> onward
        return text[:text.index("<think>")].strip()
    return text.strip()


def _get_groq_token() -> str:
    from src.config.settings import get_settings
    s = get_settings()
    # Support GROQ_API_KEY (local .env) and GROQAPIKEY (HF Spaces — no underscores)
    token = s.GROQ_API_KEY or s.GROQAPIKEY or ""
    if not token:
        raise RuntimeError(
            "Groq API key not set. Add GROQ_API_KEY to .env (local) "
            "or GROQAPIKEY to HF Spaces secrets."
        )
    return token


def _call_groq(model_id: str, messages: list, max_tokens: int,
               temperature: float, token: str) -> str | None:
    """Single Groq call. Returns cleaned text on success, None on rate-limit/unavailable."""
    import requests
    logger.info("[LLM] Calling %s (max_tokens=%d)", model_id, max_tokens)
    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type": "application/json"},
            json={"model": model_id, "messages": messages,
                  "max_tokens": max_tokens, "temperature": temperature,
                  "stream": False},
            timeout=60,
        )
        if resp.ok:
            text    = resp.json()["choices"][0]["message"]["content"]
            cleaned = _strip_thinking(text)
            logger.info("[LLM] %s -> %d chars", model_id, len(cleaned))
            if not cleaned.strip():
                logger.warning("[LLM] %s empty after think-strip, raw length=%d", model_id, len(text))
                # Don't return None — let rotation handle it only if truly empty
                # This can happen if model outputs ONLY thinking; try next
                return None
            return cleaned
        if resp.status_code == 429:
            logger.warning("[LLM] %s rate-limited, trying next model", model_id)
            return None
        if resp.status_code in (400, 404):
            logger.warning("[LLM] %s not available (%s)", model_id, resp.status_code)
            return None
        logger.warning("[LLM] %s returned %s: %s", model_id, resp.status_code, resp.text[:80])
        return None
    except Exception as exc:
        logger.warning("[LLM] %s error: %s", model_id, exc)
        return None


def get_llm(model_id: str | None = None, deep: bool = False):
    """
    Return a LangChain-compatible chat LLM backed by Groq.
    Automatically rotates through all 5 models if the selected one is rate-limited.
    <think> blocks from reasoning models are stripped automatically.
    """
    from src.config.settings import get_settings
    settings   = get_settings()
    model_id   = model_id or settings.HF_MODEL_ID or DEFAULT_MODEL
    max_tokens = 2048 if deep else 1024
    groq_token = _get_groq_token()

    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from typing import Any, List, Optional

    _mid  = model_id
    _tok  = groq_token
    _maxt = max_tokens

    class _GroqLLM(BaseChatModel):
        model:          str   = _mid
        groq_token:     str   = _tok
        max_new_tokens: int   = _maxt
        temperature:    float = 0.7

        class Config:
            arbitrary_types_allowed = True

        @property
        def _llm_type(self) -> str:
            return "groq"

        def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
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

            # Build rotation: selected model first, then all others
            rotation = [self.model]
            for m in _ROTATION:
                if m not in rotation:
                    rotation.append(m)

            text = None
            for candidate in rotation:
                text = _call_groq(candidate, chat_msgs,
                                  self.max_new_tokens, self.temperature,
                                  self.groq_token)
                if text is not None:
                    if candidate != self.model:
                        logger.info("[LLM] Used fallback model: %s", candidate)
                    break
                time.sleep(1)  # small gap before next model

            if text is None:
                raise RuntimeError(
                    "All Groq models are currently rate-limited. "
                    "Wait 1 minute and try again — Groq free tier resets per minute."
                )

            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=text))]
            )

    return _GroqLLM()


# Backward-compatible alias
def get_hf_llm(model_id: str | None = None, deep: bool = False):
    return get_llm(model_id=model_id, deep=deep)


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    from src.config.settings import get_settings
    return HuggingFaceEmbeddings(model_name=get_settings().EMBEDDING_MODEL)
