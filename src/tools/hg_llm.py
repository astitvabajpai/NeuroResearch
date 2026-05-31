from src.config.settings import get_settings

settings = get_settings()

# Curated list of free HuggingFace models that work with the Inference API
AVAILABLE_MODELS = [
    {"id": "meta-llama/Meta-Llama-3-8B-Instruct",    "name": "Llama 3 8B",         "tag": "Recommended"},
    {"id": "mistralai/Mistral-7B-Instruct-v0.3",      "name": "Mistral 7B v0.3",    "tag": "Fast"},
    {"id": "Qwen/Qwen2.5-7B-Instruct",                "name": "Qwen 2.5 7B",        "tag": "Strong"},
    {"id": "microsoft/Phi-3-mini-4k-instruct",        "name": "Phi-3 Mini 4K",      "tag": "Lightweight"},
    {"id": "HuggingFaceH4/zephyr-7b-beta",            "name": "Zephyr 7B Beta",     "tag": ""},
    {"id": "google/gemma-2-2b-it",                    "name": "Gemma 2 2B",         "tag": "Small"},
    {"id": "tiiuae/falcon-7b-instruct",               "name": "Falcon 7B",          "tag": ""},
]

DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

# Cache LLM instances by model_id to avoid re-initialising on every request
_llm_cache: dict = {}


def get_hf_llm(model_id: str | None = None):
    """Return a ChatHuggingFace instance for the given model_id.
    Results are cached so the same model is not re-initialised per request."""
    from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

    model_id = model_id or settings.HF_MODEL_ID or DEFAULT_MODEL

    if model_id in _llm_cache:
        return _llm_cache[model_id]

    endpoint = HuggingFaceEndpoint(
        repo_id=model_id,
        huggingfacehub_api_token=settings.HF_API_TOKEN,
        temperature=0.7,
        max_new_tokens=1024,
        task="conversational",
    )
    llm = ChatHuggingFace(llm=endpoint)
    _llm_cache[model_id] = llm
    return llm


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)