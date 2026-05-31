from src.config.settings import get_settings

settings = get_settings()

AVAILABLE_MODELS = [
    {"id": "meta-llama/Meta-Llama-3-8B-Instruct",  "name": "Llama 3 8B",      "tag": "Recommended"},
    {"id": "mistralai/Mistral-7B-Instruct-v0.3",    "name": "Mistral 7B v0.3", "tag": "Fast"},
    {"id": "Qwen/Qwen2.5-7B-Instruct",              "name": "Qwen 2.5 7B",     "tag": "Strong"},
    {"id": "microsoft/Phi-3-mini-4k-instruct",      "name": "Phi-3 Mini 4K",   "tag": "Lightweight"},
    {"id": "HuggingFaceH4/zephyr-7b-beta",          "name": "Zephyr 7B Beta",  "tag": ""},
    {"id": "google/gemma-2-2b-it",                  "name": "Gemma 2 2B",      "tag": "Small"},
    {"id": "tiiuae/falcon-7b-instruct",             "name": "Falcon 7B",       "tag": ""},
]

DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

_llm_cache: dict = {}


def get_hf_llm(model_id: str | None = None):
    """
    LangChain chat LLM backed by huggingface_hub.InferenceClient.
    All heavy imports are deferred to first call to avoid blocking at startup.
    """
    model_id = model_id or settings.HF_MODEL_ID or DEFAULT_MODEL

    if model_id in _llm_cache:
        return _llm_cache[model_id]

    # Deferred imports — avoids network calls at module load time
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from typing import Any, List, Optional

    token = settings.HF_API_TOKEN
    _model_id = model_id  # capture for closure

    class _HFChatLLM(BaseChatModel):
        model: str = _model_id
        hf_token: str = token
        max_new_tokens: int = 1024
        temperature: float = 0.7

        class Config:
            arbitrary_types_allowed = True

        @property
        def _llm_type(self) -> str:
            return "hf-inference-client"

        def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            **kwargs: Any,
        ) -> ChatResult:
            # Import here so no network call happens until actual inference
            from huggingface_hub import InferenceClient

            client = InferenceClient(model=self.model, token=self.hf_token)

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

            response = client.chat_completion(
                messages=hf_msgs,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            text = response.choices[0].message.content
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=text))]
            )

    llm = _HFChatLLM()
    _llm_cache[model_id] = llm
    return llm


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
