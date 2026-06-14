import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.emotion_classifer import load_emotion_classifier, predict_emotion


MODEL_PATH = "./models/emotion_model"

print("Loading emotion classification model...")
tokenizer, model = load_emotion_classifier(model_path=MODEL_PATH)
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

# ── Run predictions ─────────────────────────────────────────────────────────
print("=" * 60)
print("  Emotion Classification Results")
print("=" * 60)

for sentence in test_sentences:
    emotion = predict_emotion(model, tokenizer, text=sentence)
    print(f"\nSentence : {sentence}")
    print(f"  --> The user is feeling {emotion}")

print("\n" + "=" * 60)
