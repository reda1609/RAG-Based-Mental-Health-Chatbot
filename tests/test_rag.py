import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openai import OpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from src.rag import RAG

# ── Load env & clients ───────────────────────────────────────────────────────
load_dotenv()

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

LLM_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

MODEL_NAME = "openai/gpt-oss-120b"
COLLECTION_NAME = "mental_health_kb"
EMBED_DIM = 384   # all-MiniLM-L6-v2

# ── Load embedder & RAG ──────────────────────────────────────────────────────
print("Loading sentence embedder...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedder loaded successfully!\n")

rag = RAG(db_client=qdrant_client, LLM_client=LLM_client, embedder=embedder)

# ── Sample queries to test the full RAG pipeline ─────────────────────────────
test_queries = [
    {
        "query"  : "I feel really overwhelmed with school and can't focus.",
        "emotion": "anxiety",
    },
    {
        "query"  : "I've been feeling so lonely and disconnected from everyone around me.",
        "emotion": "sadness",
    },
    {
        "query"  : "I keep having panic attacks and I don't know how to stop them.",
        "emotion": "fear",
    },
]

# ── Run RAG pipeline ─────────────────────────────────────────────────────────
print("=" * 60)
print("  RAG Pipeline Results")
print("=" * 60)

for item in test_queries:
    print(f"\nQuery   : {item['query']}")
    print(f"Emotion : {item['emotion']}")
    answer = rag.generate(
        user_query=item["query"],
        model_name=MODEL_NAME,
        emotion=item["emotion"],
        collection_name=COLLECTION_NAME
    )
    print(f"\n  --> Response:\n{answer}")
    print("\n" + "-" * 60)

print("\n" + "=" * 60)
