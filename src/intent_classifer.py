import json

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

def classify_intent(client, model_name, user_message):
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

    # If the element is itself a plain string (the intent label directly)
    if isinstance(parsed, str):
        return parsed

    # Normal case: {"intent": "..."}
    return parsed["intent"]