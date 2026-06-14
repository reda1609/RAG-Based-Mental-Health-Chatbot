import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

EMOTION_LABELS = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}

def load_emotion_classifier(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).eval()
    return tokenizer, model

def predict_emotion(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs.pop("token_type_ids", None)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_id = outputs.logits.argmax(dim=-1).item()
    return EMOTION_LABELS[predicted_id]