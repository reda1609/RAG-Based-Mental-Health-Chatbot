import re
import json

VALID_INTENTS = [
    "asking_mental_health_question",   # longest first to avoid partial matches
    "out_of_scope",
    "gratitude",
    "greeting",
    "goodbye",
]

INTENT_PROMPT = """You are an intent classifier for a mental health support chatbot.
Classify the user's message into EXACTLY ONE of these intents:
- greeting: user is saying hello/hi
- goodbye: user is ending the conversation
- gratitude: user is thanking the assistant
- asking_mental_health_question: user is asking about anxiety, depression, stress, coping strategies, or seeking emotional/crisis support
- out_of_scope: anything unrelated to mental health (e.g. coding help, sports, recipes)

Respond with ONLY a JSON object: {{"intent": "<one_of_the_five_labels>"}}

Examples:
User: "Hi there"
{{"intent": "greeting"}}

User: "I've been feeling really anxious lately and can't sleep"
{{"intent": "asking_mental_health_question"}}

User: "Thanks so much, that helped"
{{"intent": "gratitude"}}

User message: "{user_message}"
"""

def _extract_from_raw(raw: str) -> str:
    """Last-resort: scan raw text for any known intent label."""
    for intent in VALID_INTENTS:
        if re.search(re.escape(intent), raw):
            return intent
    raise ValueError(f"Could not extract a valid intent from model response: {raw!r}")


def classify_intent(client, model_name, user_message, max_retries: int = 3):
    for attempt in range(1, max_retries + 1):
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": INTENT_PROMPT.format(user_message=user_message)}],
            temperature=0,
            response_format={"type": "json_object"}
        )

        raw = response.choices[0].message.content
        parsed = json.loads(raw)

        # Unwrap list wrapper if model returns [...] instead of {...}
        if isinstance(parsed, list):
            parsed = parsed[0]

        # Plain string — the intent label itself
        if isinstance(parsed, str) and parsed in VALID_INTENTS:
            return parsed

        # Normal case: {"intent": "..."}
        if isinstance(parsed, dict):
            if "intent" in parsed and parsed["intent"] in VALID_INTENTS:
                return parsed["intent"]
            # Dict without "intent" key — return first string value that is a valid intent
            for v in parsed.values():
                if isinstance(v, str) and v in VALID_INTENTS:
                    return v

        # Try scanning the raw text for a known label before retrying
        try:
            return _extract_from_raw(raw)
        except ValueError:
            if attempt == max_retries:
                raise ValueError(
                    f"Model returned an unrecognisable response after {max_retries} attempts. "
                    f"Last raw response: {raw!r}"
                )
            # else: loop and retry