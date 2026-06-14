import joblib


# ── Language code → human-readable name ────────────────────────────────────
LANGUAGE_NAMES = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sw": "Swahili",
    "th": "Thai",
    "tr": "Turkish",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
}

# ── Load model ──────────────────────────────────────────────────────────────
MODEL_PATH = "./models/language_detector.pkl"

print("Loading language detector model...")
language_detector = joblib.load(MODEL_PATH)
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
    predicted_code = language_detector.predict([sentence])[0]
    predicted_language = LANGUAGE_NAMES.get(predicted_code, predicted_code)
    print(f"\nSentence : {sentence}")
    print(f"  ➜  The user is speaking {predicted_language}")

print("\n" + "=" * 55)