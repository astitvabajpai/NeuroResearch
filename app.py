"""
HuggingFace Spaces entrypoint.
Spaces expects app.py at the root binding on port 7860.
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )
