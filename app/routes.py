import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

# Ensure src/ is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import run_pipeline
from app.schemas import ChatRequest, ChatResponse
from app.dependencies import app_state


router = APIRouter()


@router.get("/health")
def health_check():
    """
    Simple liveness probe.
    Returns 'ready' once all models are loaded, 'loading' otherwise.
    Render (and other platforms) ping this to confirm the server is alive.
    """
    return {
        "status": "ready" if app_state else "loading",
        "models_loaded": bool(app_state),
    }


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main endpoint.  Receives a user message, runs the full pipeline,
    and returns the chatbot response with metadata.
    """
    if not app_state:
        raise HTTPException(
            status_code=503,
            detail="Server is still loading models. Please try again in a moment."
        )

    try:
        result = run_pipeline(
            user_message        = request.message,
            language_detector   = app_state["language_detector"],
            LLM_client          = app_state["LLM_client"],
            emotion_tokenizer   = app_state["emotion_tokenizer"],
            emotion_model       = app_state["emotion_model"],
            rag                 = app_state["rag"],
            intent_model_name   = app_state["intent_model_name"],
            rag_model_name      = app_state["rag_model_name"],
            collection_name     = app_state["collection_name"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(
        response  = result["response"],
        intent    = result["intent"],
        emotion   = result["emotion"],
        lang_name = result["lang_name"],
        lang_code = result["lang_code"],
    )
