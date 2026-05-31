from src.config.settings import get_settings

settings = get_settings()

AVAILABLE_MODELS = [
    {"id": "Qwen/Qwen2.5-7B-Instruct",             "name": "Qwen 2.5 7B",     "tag": "Recommended"},
    {"id": "meta-llama/Meta-Llama-3-8B-Instruct",  "name": "Llama 3 8B",      "tag": "Gated"},
    {"id": "mistralai/Mistral-7B-Instruct-v0.3",   "name": "Mistral 7B v0.3", "tag": ""},
    {"id": "microsoft/Phi-3-mini-4k-instruct",     "name": "Phi-3 Mini 4K",   "tag": "Lightweight"},
    {"id": "HuggingFaceH4/zephyr-7b-beta",         "name": "Zephyr 7B Beta",  "tag": ""},
    {"id": "google/gemma-2-2b-it",                 "name": "Gemma 2 2B",      "tag": "Small"},
    {"id": "tiiuae/falcon-7b-instruct",            "name": "Falcon 7B",       "tag": ""},
]

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

_llm_cache: dict = {}


def get_hf_llm(model_id: str | None = None):
    """
    LangChain-compatible chat LLM using the HuggingFace Inference API.
    Uses requests directly to avoid DNS routing issues on HF Spaces.
    """
    model_id = model_id or settings.HF_MODEL_ID or DEFAULT_MODEL

    if model_id in _llm_cache:
        return _llm_cache[model_id]

    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from typing import Any, List, Optional

    token    = settings.HF_API_TOKEN
    _mid     = model_id

    class _HFChatLLM(BaseChatModel):
        model:          str = _mid
        hf_token:       str = token
        max_new_tokens: int = 1024
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

            # Use router.huggingface.co which works inside HF Spaces
            # (api-inference.huggingface.co has DNS issues on Spaces)
            url = f"https://router.huggingface.co/featherless-ai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.hf_token}",
                "Content-Type":  "application/json",
            }
            payload = {
                "model":       self.model,
                "messages":    hf_msgs,
                "max_tokens":  self.max_new_tokens,
                "temperature": self.temperature,
                "stream":      False,
            }

            resp = requests.post(url, headers=headers, json=payload, timeout=120)

            # If featherless-ai doesn't support this model, try nebius
            if resp.status_code == 400 and "not supported" in resp.text:
                for provider in ["nebius", "novita", "together"]:
                    url2 = f"https://router.huggingface.co/{provider}/v1/chat/completions"
                    resp = requests.post(url2, headers=headers, json=payload, timeout=120)
                    if resp.ok:
                        break

            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=text))]
            )

    llm = _HFChatLLM()
    _llm_cache[model_id] = llm
    return llm


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
