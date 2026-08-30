"""
HuggingFace Inference API — LangChain-compatible chat LLM wrapper.
Uses requests directly to avoid DNS routing issues on HF Spaces.
Settings are loaded lazily so the module is safe to import without a .env.
"""

from __future__ import annotations

AVAILABLE_MODELS = [
    {"id": "Qwen/Qwen2.5-7B-Instruct",  "name": "Qwen 2.5 7B",  "tag": "Fast"},
    {"id": "Qwen/Qwen2.5-14B-Instruct", "name": "Qwen 2.5 14B", "tag": "Balanced"},
    {"id": "Qwen/Qwen2.5-32B-Instruct", "name": "Qwen 2.5 32B", "tag": "Strong"},
    {"id": "Qwen/Qwen2.5-72B-Instruct", "name": "Qwen 2.5 72B", "tag": "Best Quality"},
]

DEFAULT_MODEL        = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_WRITER_MODEL = "Qwen/Qwen2.5-14B-Instruct"
DEFAULT_CRITIC_MODEL = "Qwen/Qwen2.5-72B-Instruct"

_llm_cache: dict = {}

# Providers in priority order — direct HF inference first (separate quota),
# then router providers as fallback
_PROVIDERS = ["hf-inference", "featherless-ai", "nebius", "novita", "together", "sambanova"]


def get_hf_llm(model_id: str | None = None, deep: bool = False):
    """
    Return a cached LangChain-compatible chat LLM backed by the HF Inference API.
    Settings are read lazily so this is safe to call after startup.
    """
    from src.config.settings import get_settings
    settings = get_settings()

    model_id = model_id or settings.HF_MODEL_ID or DEFAULT_MODEL
    cache_key = f"{model_id}:{'deep' if deep else 'std'}"

    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from typing import Any, Iterator, List, Optional

    _token   = settings.HF_API_TOKEN
    _mid     = model_id
    _tokens  = 2048 if deep else 1024

    class _HFChatLLM(BaseChatModel):
        model:          str   = _mid
        hf_token:       str   = _token
        max_new_tokens: int   = _tokens
        temperature:    float = 0.7

        class Config:
            arbitrary_types_allowed = True

        @property
        def _llm_type(self) -> str:
            return "hf-inference-api"

        def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            **kwargs: Any,
        ) -> ChatResult:
            import requests

            hf_msgs = []
            for m in messages:
                if isinstance(m, SystemMessage):
                    hf_msgs.append({"role": "system",    "content": m.content})
                elif isinstance(m, HumanMessage):
                    hf_msgs.append({"role": "user",      "content": m.content})
                elif isinstance(m, AIMessage):
                    hf_msgs.append({"role": "assistant", "content": m.content})
                else:
                    hf_msgs.append({"role": "user",      "content": str(m.content)})

            payload = {
                "model":       self.model,
                "messages":    hf_msgs,
                "max_tokens":  self.max_new_tokens,
                "temperature": self.temperature,
                "stream":      False,
            }
            headers = {
                "Authorization": f"Bearer {self.hf_token}",
                "Content-Type":  "application/json",
            }

            last_exc: Exception | None = None
            for provider in _PROVIDERS:
                url = f"https://router.huggingface.co/{provider}/v1/chat/completions"
                try:
                    resp = requests.post(url, headers=headers, json=payload, timeout=30)
                    if resp.ok:
                        data = resp.json()
                        text = data["choices"][0]["message"]["content"]
                        return ChatResult(
                            generations=[ChatGeneration(message=AIMessage(content=text))]
                        )
                    # "not supported" or 400 → try next provider silently
                    if resp.status_code in (400, 404) or "not supported" in resp.text.lower():
                        continue
                    # Any other error — raise immediately with full detail
                    resp.raise_for_status()
                except requests.exceptions.Timeout:
                    last_exc = Exception(f"{provider} timed out")
                    continue
                except requests.exceptions.ConnectionError as exc:
                    last_exc = exc
                    continue
                except requests.exceptions.RequestException as exc:
                    last_exc = exc
                    continue

            raise RuntimeError(
                f"All HF providers failed for model '{self.model}'. "
                f"Last error: {last_exc}. "
                f"Try a different model — Qwen/Qwen2.5-7B-Instruct has the widest provider support."
            )

    llm = _HFChatLLM()
    _llm_cache[cache_key] = llm
    return llm


def get_embeddings():
    from src.config.settings import get_settings
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=get_settings().EMBEDDING_MODEL)
