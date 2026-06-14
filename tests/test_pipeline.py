import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openai import OpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.language_detector import load_langauge_detector
from src.emotion_classifer import load_emotion_classifier
from src.rag import RAG
from src.pipeline import run_pipeline


# ── Load env & clients ───────────────────────────────────────────────────────
load_dotenv()

LLM_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)


# ── Config ───────────────────────────────────────────────────────────────────
LANG_MODEL_PATH    = "./models/language_detector.pkl"
EMOTION_MODEL_PATH = "./models/emotion_model"
COLLECTION_NAME    = "mental_health_kb"
INTENT_MODEL       = "openai/gpt-oss-20b"
RAG_MODEL          = "openai/gpt-oss-120b"


# ── Load all models (once) ───────────────────────────────────────────────────
print("Loading language detector...")
language_detector = load_langauge_detector(model_path=LANG_MODEL_PATH)

print("Loading emotion classifier...")
emotion_tokenizer, emotion_model = load_emotion_classifier(model_path=EMOTION_MODEL_PATH)

print("Loading sentence embedder...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("All models loaded!\n")

rag = RAG(db_client=qdrant_client, LLM_client=LLM_client, embedder=embedder)


# ── Test cases covering every branch and several languages ───────────────────
test_cases = [
    # (user_message,                                                   description)
    ("Hello! How are you?",
     "greeting      | English"),

    ("See you later, bye!",
     "goodbye       | English"),

    ("Thank you so much, that really helped!",
     "gratitude     | English"),

    ("Can you help me write a Python script?",
     "out_of_scope  | English"),

    ("I've been feeling really anxious and can't sleep at night.",
     "mental_health | English"),

    ("أشعر بقلق شديد ولا أستطيع النوم.",
     "mental_health | Arabic"),

    ("Je me sens très déprimé ces derniers temps.",
     "mental_health | French"),

    ("Ich fühle mich sehr einsam und traurig.",
     "mental_health | German"),
]


# ── Run pipeline on every test case ─────────────────────────────────────────
print("=" * 65)
print("  Full Pipeline Test Results")
print("=" * 65)

for message, description in test_cases:
    print(f"\n[{description}]")
    print(f"Input    : {message}")

    result = run_pipeline(
        user_message=message,
        language_detector=language_detector,
        LLM_client=LLM_client,
        emotion_tokenizer=emotion_tokenizer,
        emotion_model=emotion_model,
        rag=rag,
        intent_model_name=INTENT_MODEL,
        rag_model_name=RAG_MODEL,
        collection_name=COLLECTION_NAME,
    )

    print(f"Language : {result['lang_name']} ({result['lang_code']})")
    if result["english_text"] != message:
        print(f"Translated: {result['english_text']}")
    print(f"Intent   : {result['intent']}")
    if result["emotion"]:
        print(f"Emotion  : {result['emotion']}")
    print(f"\n  --> Response:\n{result['response']}")
    print("\n" + "-" * 65)

print("\n" + "=" * 65)
