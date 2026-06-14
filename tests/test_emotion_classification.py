import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Emotion label mapping ───────────────────────────────────────────────────
# Labels from dair-ai/emotion dataset (0-indexed)
EMOTION_LABELS = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}

# ── Load model & tokenizer ──────────────────────────────────────────────────
MODEL_PATH = "./models/emotion_model"

print("Loading emotion classification model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
print("Model loaded successfully!\n")

# ── Sample sentences covering different emotions ────────────────────────────
test_sentences = [
    "I feel so down and hopeless, nothing seems to matter anymore.",   # sadness
    "Today was amazing! I got the job I always dreamed of!",           # joy
    "I am deeply in love with you and I cherish every moment with you.",  # love
    "I am furious! How could they do this to me?",                     # anger
    "I am terrified of what might happen next, I can barely breathe.", # fear
    "I had no idea this was going to happen, I am completely shocked!", # surprise
]

# ── Predict emotion ─────────────────────────────────────────────────────────
def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs.pop("token_type_ids", None)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_id = outputs.logits.argmax(dim=-1).item()
    return EMOTION_LABELS[predicted_id]

# ── Run predictions ─────────────────────────────────────────────────────────
print("=" * 60)
print("  Emotion Classification Results")
print("=" * 60)

for sentence in test_sentences:
    emotion = predict_emotion(sentence)
    print(f"\nSentence : {sentence}")
    print(f"  --> The user is feeling {emotion}")

print("\n" + "=" * 60)
