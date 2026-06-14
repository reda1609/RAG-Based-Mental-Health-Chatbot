import joblib
from deep_translator import GoogleTranslator


lang_detector = joblib.load("models/language_detector.pkl")

def detect_language(text):
    return lang_detector.predict([text])[0]

def translate_to_english(text, detected_lang):
    """Translate text to English if it's not already in English."""
    if detected_lang == "en":
        return text
    return GoogleTranslator(source=detected_lang, target="en").translate(text)

def preprocess_query(user_input):
    detected_lang = detect_language(user_input)
    english_text = translate_to_english(user_input, detected_lang)
    return english_text, detected_lang


# --- Quick test ---
if __name__ == "__main__":
    test_inputs = [
        "I feel really overwhelmed with school and can't focus",  # English
        "أشعر بالقلق الشديد ولا أستطيع النوم",                   # Arabic
        "Je me sens très déprimé ces derniers temps",             # French
        "Ich fühle mich sehr einsam und traurig",                 # German
    ]

    for text in test_inputs:
        translated, lang = preprocess_query(text)
        print(f"[{lang}] Original : {text}")
        print(f"      Translated: {translated}")
        print()
