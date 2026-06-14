import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.language_detector import load_langauge_detector, predict_language_name

# ── Load model ──────────────────────────────────────────────────────────────
MODEL_PATH = "./models/language_detector.pkl"

print("Loading language detector model...")
language_detector = load_langauge_detector(model_path=MODEL_PATH)
print("Model loaded successfully!\n")

# ── Sample sentences in different languages ─────────────────────────────────
test_sentences = [
    "I feel very anxious and I don't know what to do.",          # English
    "Je me sens très anxieux et je ne sais pas quoi faire.",     # French
    "أشعر بقلق شديد ولا أعرف ماذا أفعل.",                       # Arabic
    "Me siento muy ansioso y no sé qué hacer.",                  # Spanish
    "Ich fühle mich sehr ängstlich und weiß nicht was ich tun soll.",  # German
    "私はとても不安で、どうすればいいかわかりません。",              # Japanese
    "Мне очень тревожно, и я не знаю, что делать.",              # Russian
    "我感到非常焦虑，不知道该怎么办。",                            # Chinese
]

# ── Run predictions ─────────────────────────────────────────────────────────
print("=" * 55)
print("  Language Detection Results")
print("=" * 55)

for sentence in test_sentences:
    predicted_language = predict_language_name(model=language_detector, text=sentence)
    print(f"\nSentence : {sentence}")
    print(f"  -->  The user is speaking {predicted_language}")

print("\n" + "=" * 55)