import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from deep_translator import GoogleTranslator

from src.language_detector import predict_language_code, LANGUAGE_NAMES
from src.intent_classifer import classify_intent
from src.emotion_classifer import predict_emotion


# ── Hardcoded direct responses for non-mental-health intents ─────────────────
DIRECT_RESPONSES = {
    "greeting":     "Hello! I'm here to support you. How are you feeling today?",
    "goodbye":      "Take care of yourself. Remember, help is always here when you need it.",
    "gratitude":    "You're very welcome. I'm glad I could help. Please don't hesitate to reach out anytime.",
    "out_of_scope": "I'm specialised in mental health support and can't help with that topic, "
                    "but I'm here if you'd like to talk about how you're feeling.",
}


# ── Translation helpers ──────────────────────────────────────────────────────
def translate_to_english(text, lang_code):
    """Translate text to English; pass through if already English."""
    if lang_code == "en":
        return text
    return GoogleTranslator(source=lang_code, target="en").translate(text)


def translate_from_english(text, lang_code):
    """Translate an English response back to the user's language."""
    if lang_code == "en":
        return text
    return GoogleTranslator(source="en", target=lang_code).translate(text)


# ── Main pipeline function ───────────────────────────────────────────────────
def run_pipeline(
    user_message,
    language_detector,
    LLM_client,
    emotion_tokenizer,
    emotion_model,
    rag,
    intent_model_name="openai/gpt-oss-20b",
    rag_model_name="openai/gpt-oss-120b",
    collection_name="mental_health_kb",
):
    """
    Run the full mental-health chatbot pipeline for a single user message.

    Parameters
    ----------
    user_message       : raw text from the user (any language)
    language_detector  : loaded sklearn language-detector pipeline (joblib)
    LLM_client         : OpenAI-compatible client (OpenRouter) for intent + RAG
    emotion_tokenizer  : HuggingFace tokenizer for the emotion model
    emotion_model      : HuggingFace model for emotion classification
    rag                : RAG instance (qdrant + embedder + LLM client)
    intent_model_name  : LLM model name used by the intent classifier
    rag_model_name     : LLM model name used by the RAG generator
    collection_name    : Qdrant collection name for the knowledge base

    Returns
    -------
    dict with keys: lang_code, lang_name, english_text, intent, emotion, response
    """

    # Step 1 ── Detect language
    lang_code = predict_language_code(language_detector, user_message)
    lang_name = LANGUAGE_NAMES.get(lang_code, lang_code)

    # Step 2 ── Translate to English (if needed)
    english_text = translate_to_english(user_message, lang_code)

    # Step 3 ── Classify intent (always on English text)
    intent = classify_intent(LLM_client, intent_model_name, user_message=english_text)

    # Step 4 ── Route by intent ───────────────────────────────────────────────

    if intent != "asking_mental_health_question":
        # Non-MH branch: hardcoded English response → translate back if needed
        response_en = DIRECT_RESPONSES.get(intent, DIRECT_RESPONSES["out_of_scope"])
        response    = translate_from_english(response_en, lang_code)

        return {
            "lang_code":    lang_code,
            "lang_name":    lang_name,
            "english_text": english_text,
            "intent":       intent,
            "emotion":      None,
            "response":     response,
        }

    # Mental-health branch ────────────────────────────────────────────────────

    # Step 5 ── Classify emotion
    emotion = predict_emotion(emotion_model, emotion_tokenizer, text=english_text)

    # Step 6 ── RAG: retrieve + generate
    # The system prompt already instructs the LLM to reply in `language`,
    # so no separate back-translation is needed on this path.
    response = rag.generate(
        user_query=english_text,
        model_name=rag_model_name,
        collection_name=collection_name,
        emotion=emotion,
        language=lang_name,
    )

    return {
        "lang_code":    lang_code,
        "lang_name":    lang_name,
        "english_text": english_text,
        "intent":       intent,
        "emotion":      emotion,
        "response":     response,
    }
