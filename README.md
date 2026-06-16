# MindEase — RAG-Based Mental Health Chatbot

A conversational AI system built for mental health support. It detects the user's language, classifies their intent and emotion, and retrieves relevant counseling knowledge to generate empathetic, grounded responses — in whatever language the user writes in.

---

## How It Works

The system runs every message through a sequential pipeline:

```
User message
    │
    ▼
1. Language Detection       ← sklearn model (20 languages)
    │
    ▼
2. Translate to English     ← Google Translate (if not already English)
    │
    ▼
3. Intent Classification    ← LLM via OpenRouter (gpt-oss-20b)
    │
    ├─ greeting / goodbye / gratitude / out_of_scope
    │       └─→ Hardcoded response, translated back → done
    │
    └─ asking_mental_health_question
            │
            ▼
        4. Emotion Classification   ← Fine-tuned DistilBERT
            │
            ▼
        5. RAG: Retrieve + Generate ← Qdrant + MiniLM + LLM (gpt-oss-120b)
            │
            ▼
        Response in the user's language
```

Non-mental-health intents (greetings, thanks, off-topic requests) are handled with hardcoded responses and never touch the RAG pipeline. Only genuine mental health queries trigger retrieval and generation.

---

## Project Structure

```
RAG-Based-Mental-Health-Chatbot/
├── app/
│   ├── main.py            # FastAPI app — lifespan, model loading, mounts frontend
│   ├── routes.py          # /health and /chat endpoints
│   ├── schemas.py         # Pydantic request/response models
│   └── dependencies.py    # Shared app_state dict
│
├── src/
│   ├── pipeline.py        # Orchestrates the full pipeline end-to-end
│   ├── rag.py             # RAG class: ingest, retrieve, generate
│   ├── intent_classifer.py
│   ├── emotion_classifer.py
│   └── language_detector.py
│
├── frontend/
│   ├── index.html         # Chat UI (MindEase)
│   ├── style.css
│   └── script.js
│
├── models/
│   ├── language_detector.pkl   # Trained sklearn pipeline
│   └── emotion_model/          # Fine-tuned DistilBERT weights
│
├── data/
│   ├── emotion_classification/ # train / validation / test CSVs
│   └── language_identification/
│
├── notebooks/
│   ├── 01_language_detection.ipynb
│   ├── 02_emotion_classifier.ipynb
│   ├── 03_intent_classifier.ipynb
│   └── 04_qa_rag_pipeline.ipynb
│
└── tests/
    ├── test_pipeline.py
    ├── test_rag.py
    ├── test_intent_classifer.py
    ├── test_emotion_classification.py
    ├── test_language_detection.py
    └── test_translation.py
```

---

## Components

### Language Detection
A scikit-learn text classification pipeline trained on 20 languages (Arabic, Bulgarian, Chinese, Dutch, English, French, German, Greek, Hindi, Italian, Japanese, Polish, Portuguese, Russian, Swahili, Thai, Turkish, Urdu, Vietnamese). The trained model is saved as a joblib pickle (`models/language_detector.pkl`).

### Intent Classification
Uses an LLM (via OpenRouter) to classify each user message into one of five intents:

| Intent | Description |
|---|---|
| `greeting` | Hello, hi, etc. |
| `goodbye` | Ending the conversation |
| `gratitude` | Thanking the assistant |
| `asking_mental_health_question` | Anxiety, depression, stress, crisis support |
| `out_of_scope` | Anything unrelated to mental health |

The classifier uses `temperature=0` and a JSON response format. It retries up to 3 times with fallback regex parsing if the model returns an unexpected format.

### Emotion Classification
A DistilBERT model fine-tuned for 6-class emotion detection:

| Label | Label | Label |
|---|---|---|
| sadness | joy | love |
| anger | fear | surprise |

Only runs when the intent is `asking_mental_health_question`. The detected emotion is passed to the RAG generator so the tone of the response can be adjusted accordingly.

### RAG (Retrieval-Augmented Generation)
- **Vector store**: Qdrant (cloud-hosted)
- **Embedder**: `all-MiniLM-L6-v2` via sentence-transformers
- **Knowledge base**: Mental health counseling Q&A dataset, stored as context-response pairs
- **Generator**: LLM (gpt-oss-120b via OpenRouter)

The system prompt instructs the model to synthesize its own response rather than copy retrieved examples verbatim, and to nudge users toward crisis helplines if self-harm is mentioned.

### Translation
Uses `deep-translator` (Google Translate) to handle non-English input. The message is translated to English before classification and retrieval. The RAG system prompt instructs the LLM to respond directly in the user's language, so no back-translation step is needed on that path.

---

## Setup

### Prerequisites
- Python 3.9+
- A [Qdrant Cloud](https://cloud.qdrant.io) account (free tier works)
- An [OpenRouter](https://openrouter.ai) API key

### Install dependencies

```bash
pip install fastapi uvicorn openai qdrant-client sentence-transformers \
            transformers torch scikit-learn joblib deep-translator \
            python-dotenv pydantic
```

### Environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openrouter_api_key
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
```

> The `OPENAI_API_KEY` here is your **OpenRouter** key, not an OpenAI key. The app points the OpenAI client at `https://openrouter.ai/api/v1`.

### Models

The `models/` directory is not tracked in git (see `.gitignore`). You need to either train them yourself using the notebooks or download them separately.

- **Language detector**: Run `notebooks/01_language_detection.ipynb` → saves `models/language_detector.pkl`
- **Emotion model**: Run `notebooks/02_emotion_classifier.ipynb` → saves `models/emotion_model/`

### Populate the vector store

Before the first run, ingest the knowledge base into Qdrant. See `notebooks/04_qa_rag_pipeline.ipynb` for the ingestion steps.

---

## Running the App

```bash
uvicorn app.main:app --reload
```

The server loads all models on startup (takes ~20–30 seconds the first time). Once ready, open your browser at:

```
http://127.0.0.1:8000
```

The frontend is served directly by FastAPI as static files, so there's no separate frontend server to run.

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe — returns `ready` or `loading` |
| `POST` | `/chat` | Send a message, get a response |

**POST /chat — request body:**
```json
{ "message": "I've been feeling really anxious lately." }
```

**Response:**
```json
{
  "response": "It sounds like you're carrying a lot right now...",
  "intent": "asking_mental_health_question",
  "emotion": "fear",
  "lang_name": "English",
  "lang_code": "en"
}
```

---

## Running Tests

```bash
python tests/test_pipeline.py
```

The test suite covers all five intents and multiple languages (English, Arabic, French, German). Individual component tests are also available in the `tests/` folder.

---

## Notebooks

| Notebook | What it covers |
|---|---|
| `01_language_detection.ipynb` | Data prep, training, and evaluation of the sklearn language detector |
| `02_emotion_classifier.ipynb` | Fine-tuning DistilBERT on the emotion dataset |
| `03_intent_classifier.ipynb` | Prompt design and testing for LLM-based intent classification |
| `04_qa_rag_pipeline.ipynb` | Qdrant setup, dataset ingestion, and RAG pipeline testing |

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI |
| LLM inference | OpenRouter (gpt-oss-20b / gpt-oss-120b) |
| Vector store | Qdrant Cloud |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Emotion model | DistilBERT (fine-tuned) |
| Language detection | scikit-learn |
| Translation | deep-translator (Google Translate) |
| Frontend | Vanilla HTML / CSS / JS |

---

## Notes

- The chatbot is designed for **supportive conversation only** — it is not a substitute for professional mental health care.
- If a user mentions self-harm or a crisis, the system prompt instructs the LLM to encourage reaching out to a crisis helpline.
- All models are loaded once at startup and kept in memory, so response latency is low after the initial boot period.