"""
HuggingFace Spaces entrypoint.
Spaces looks for app.py at the root and expects it to bind on port 7860.
"""
import uvicorn
from src.api import app  # noqa: F401 — imported so uvicorn can find it

if __name__ == "__main__":
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )
