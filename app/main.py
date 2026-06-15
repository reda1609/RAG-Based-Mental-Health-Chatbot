import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# ── Resolve project root so all relative paths work correctly ────────────────
# __file__ is  .../app/main.py  →  .parent = app/  →  .parent = project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.language_detector import load_langauge_detector
from src.emotion_classifer import load_emotion_classifier
from src.rag import RAG
from app.dependencies import app_state
from app.routes import router

load_dotenv(ROOT / ".env")

# ── Model / service configuration ───────────────────────────────────────────
LANG_MODEL_PATH    = ROOT / "models" / "language_detector.pkl"
EMOTION_MODEL_PATH = ROOT / "models" / "emotion_model"
COLLECTION_NAME    = "mental_health_kb"
INTENT_MODEL       = "openai/gpt-oss-20b"
RAG_MODEL          = "openai/gpt-oss-120b"


# ── Lifespan: runs on startup, then yields, then runs on shutdown ────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Everything BEFORE yield  → executed once when the server starts.
    Everything AFTER  yield  → executed once when the server shuts down.

    We load all heavy models here so they live in memory for every request,
    instead of being reloaded on every call (which would take ~30 seconds each).
    """
    print("\n⏳  Loading models — this takes about 20–30 seconds on first run...")

    LLM_client = OpenAI(
        api_key  = os.getenv("OPENAI_API_KEY"),
        base_url = "https://openrouter.ai/api/v1",
    )

    qdrant_client = QdrantClient(
        url     = os.getenv("QDRANT_URL"),
        api_key = os.getenv("QDRANT_API_KEY"),
    )

    print("    • Language detector...")
    language_detector = load_langauge_detector(model_path=str(LANG_MODEL_PATH))

    print("    • Emotion classifier (distilbert-base-uncased)...")
    emotion_tokenizer, emotion_model = load_emotion_classifier(model_path=str(EMOTION_MODEL_PATH))

    print("    • Sentence embedder (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    rag = RAG(db_client=qdrant_client, LLM_client=LLM_client, embedder=embedder)

    # Store everything in the shared app_state dict (defined in dependencies.py)
    app_state.update({
        "language_detector":  language_detector,
        "emotion_tokenizer":  emotion_tokenizer,
        "emotion_model":      emotion_model,
        "LLM_client":         LLM_client,
        "rag":                rag,
        "intent_model_name":  INTENT_MODEL,
        "rag_model_name":     RAG_MODEL,
        "collection_name":    COLLECTION_NAME,
    })

    print("✅  All models loaded. Server is ready at http://127.0.0.1:8000\n")
    yield                       # ← server is running and accepting requests here

    # Shutdown cleanup
    print("\n🛑  Shutting down — clearing model state...")
    app_state.clear()


# ── Create the FastAPI application ───────────────────────────────────────────
app = FastAPI(
    title       = "Mental Health Chatbot API",
    description = "A RAG-based mental health support chatbot",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# Register the API routes (/health  and  /chat)
app.include_router(router)

# Mount the frontend/ folder as static files.
# MUST be last — static mount is a catch-all and would swallow API routes
# if registered before them.
app.mount(
    "/",
    StaticFiles(directory=str(ROOT / "frontend"), html=True),
    name="frontend",
)
