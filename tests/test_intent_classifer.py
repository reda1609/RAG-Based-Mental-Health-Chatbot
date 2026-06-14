import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openai import OpenAI
from dotenv import load_dotenv
from src.intent_classifer import classify_intent

# ── Load env & client ────────────────────────────────────────────────────────
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

MODEL_NAME = "openai/gpt-oss-20b"

# ── Sample messages covering all five intents ────────────────────────────────
test_messages = [
    ("Hello! How are you?", "greeting"),
    ("See you again, Bye!", "goodbye"),
    ("Thanks so much, that really helped me.", "gratitude"),
    ("I keep overthinking everything and I can't sleep.", "asking_mental_health_question"),
    ("Can you help me write a Python script?", "out_of_scope"),
]

# ── Run predictions ──────────────────────────────────────────────────────────
print("=" * 60)
print("  Intent Classification Results")
print("=" * 60)

for message, expected in test_messages:
    intent = classify_intent(client, MODEL_NAME, user_message=message)
    status = "✓" if intent == expected else "✗"
    print(f"\nMessage  : {message}")
    print(f"Expected : {expected}")
    print(f"  --> Predicted intent: {intent}  [{status}]")

print("\n" + "=" * 60)
