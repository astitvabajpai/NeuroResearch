"""
LLM layer — Groq only.
All 5 models are free, confirmed working at 2048 tokens (deep research).
Falls back through the rotation automatically on rate limits.
"""
from __future__ import annotations
import time
import logging

logger = logging.getLogger(__name__)

AVAILABLE_MODELS = [
    {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B",  "tag": "Recommended"},
    {"id": "qwen/qwen3.6-27b",        "name": "Qwen 3.6 27B",   "tag": "Best Depth"},
    {"id": "openai/gpt-oss-120b",     "name": "GPT OSS 120B",   "tag": "Most Powerful"},
    {"id": "openai/gpt-oss-20b",      "name": "GPT OSS 20B",    "tag": "Balanced"},
    {"id": "llama-3.1-8b-instant",    "name": "Llama 3.1 8B",   "tag": "Fastest"},
]

DEFAULT_MODEL        = "llama-3.3-70b-versatile"
DEFAULT_WRITER_MODEL = "llama-3.3-70b-versatile"
DEFAULT_CRITIC_MODEL = "llama-3.3-70b-versatile"

# Rotation order when primary model is rate-limited
_ROTATION = [m["id"] for m in AVAILABLE_MODELS]

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def _get_groq_token() -> str:
    from src.config.settings import get_settings
    token = get_settings().GROQ_API_KEY or ""
    if not token:
        raise RuntimeError("GROQ_API_KEY is not set in .env")
    return token


def _call_groq(model_id: str, messages: list, max_tokens: int,
               temperature: float, token: str) -> str | None:
    """Single Groq call. Returns text on success, None on rate-limit/unavailable."""
    import requests
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
            return resp.json()["choices"][0]["message"]["content"]
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
    """
    from src.config.settings import get_settings
    settings  = get_settings()
    model_id  = model_id or settings.HF_MODEL_ID or DEFAULT_MODEL
    max_tokens = 2048 if deep else 1024
    groq_token = _get_groq_token()

    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from typing import Any, List, Optional

    _mid   = model_id
    _tok   = groq_token
    _maxt  = max_tokens

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
    """Embeddings still use HuggingFace (local, no API key needed)."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from src.config.settings import get_settings
    return HuggingFaceEmbeddings(model_name=get_settings().EMBEDDING_MODEL)
