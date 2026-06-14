import joblib

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

def load_langauge_detector(model_path:str):
    return joblib.load(model_path)

def predict_language_code(model, text):
    return model.predict([text])[0]

def predict_language_name(model, text):
    predicted_code = model.predict([text])[0]
    return LANGUAGE_NAMES.get(predicted_code, predicted_code)