# main.py
import requests
import datetime
import textwrap
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Literal

from lindenberg_data import pdf_documents

# ---- CONFIG ----
MODEL = "swiss-ai/apertus-8b-instruct"
HISTORY_LENGTH = 5
CONTEXT_LEN = 3

import os

APERTUS_KEY = os.environ.get("APERTUS_KEY")
if not APERTUS_KEY:
    raise RuntimeError("APERTUS_KEY environment variable is not set")

INSTRUCTIONS = "Du bist ein neutraler Informationsassistent fuer das Windpark Lindenberg Projekt. Antworte kurz, sachlich und nur auf Basis der vorhandenen Dokumente. Wenn keine Informationen vorhanden sind, sage dies deutlich."

# ---- DATA MODELS ----
Role = Literal["user", "assistant"]

class Message(BaseModel):
    role: Role
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

# ---- HELPER FUNCTIONS ----
def improved_search(query, documents, max_results=3):
    """Simple search scoring similar to your Streamlit version."""
    if not documents:
        return []

    query_words = query.lower().split()
    scored_docs = []

    for doc in documents:
        content_lower = doc["content"].lower()
        score = 0

        for word in query_words:
            if word in content_lower:
                score += content_lower.count(word)

        if query.lower() in content_lower:
            score += 5

        if score > 0:
            scored_docs.append((score, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs[:max_results]]

def history_to_text(chat_history: List[Message]) -> str:
    return "\n".join(f"[{h.role}]: {h.content}" for h in chat_history)

def build_prompt(question: str, history: List[Message]) -> str:
    documents = pdf_documents

    # Limit history
    recent_history = history[-HISTORY_LENGTH:] if history else []
    recent_history_str = history_to_text(recent_history) if recent_history else None

    # Document context
    relevant_docs = improved_search(question, documents, max_results=CONTEXT_LEN)
    if relevant_docs:
        context_str = "\n\n".join(
            [f"Quelle: {doc['source']}\n{doc['content']}" for doc in relevant_docs]
        )
    else:
        context_str = "Keine relevanten Informationen in den Dokumenten gefunden."

    parts = []
    parts.append(f"<instructions>\n{INSTRUCTIONS}\n</instructions>")
    if context_str:
        parts.append(f"<document_context>\n{context_str}\n</document_context>")
    if recent_history_str:
        parts.append(f"<recent_messages>\n{recent_history_str}\n</recent_messages>")
    parts.append(f"<question>\n{question}\n</question>")

    return "\n\n".join(parts)

def call_apertus(prompt: str) -> str:
    """Call the PublicAI Apertus model and return the full text response."""
    response = requests.post(
        "https://api.publicai.co/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {APERTUS_KEY}",
            "User-Agent": "WindparkChatbot/1.0",
        },
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 400,
            "temperature": 0.1,
            "top_p": 0.9,
        },
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"API Fehler: {response.status_code} - {response.text}"
        )

    result = response.json()
    if "choices" not in result or not result["choices"]:
        return "Keine Antwort vom Apertus-Modell erhalten."

    return result["choices"][0]["message"]["content"]

# ---- FASTAPI APP ----
app = FastAPI()

# ADD CORS MIDDLEWARE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok", "message": "Windpark Lindenberg backend is running."}

@app.post("/chat")
def chat(req: ChatRequest):
    """
    Accepts:
    {
      "messages": [
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "..."},
        ...
      ]
    }
    """
    if not req.messages:
        return {"reply": "Bitte stellen Sie eine Frage zum Windpark Lindenberg."}

    # Last message is assumed to be the new user question
    question = req.messages[-1].content
    history = req.messages[:-1]

    prompt = build_prompt(question, history)
    answer = call_apertus(prompt)
    return {"reply": answer}
