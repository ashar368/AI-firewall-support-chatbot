from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from rapidfuzz import fuzz
from fastapi.responses import FileResponse
import os
from openai import OpenAI
import numpy as np

client = OpenAI(api_key=os.getenv("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"))
                
df = pd.read_csv("faq.csv")
faq_questions = df["Question"].tolist()
faq_answers = df["Answer"].tolist()

# ✅ Generate embeddings for all questions (one-time)
faq_embeddings = []

for q in faq_questions:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=q
    )
    faq_embeddings.append(response.data[0].embedding)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
def find_best_answer(query):

    # ✅ Convert user query to embedding
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = response.data[0].embedding

    best_score = -1
    best_answer = None

    for i, emb in enumerate(faq_embeddings):

        score = cosine_similarity(query_embedding, emb)

        if score > best_score:
            best_score = score
            best_answer = faq_answers[i]

    # ✅ threshold
    if best_score > 0.7:
        return best_answer

    return None
@app.post("/chat")
def chat(req: Request):

    answer = find_best_answer(req.message)

    if answer:
        return {"reply": answer}

    return {"reply": "I couldn’t find a relevant answer. Please rephrase your question."}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def serve_ui():
    return FileResponse("chatbox.html")

df = pd.read_csv("faq.csv")

class Request(BaseModel):
    message: str


def find_top_answers(query, top_n=3):
    results = []

    for q, a in zip(df["Question"], df["Answer"]):
        score = fuzz.partial_ratio(query.lower(), q.lower())
        results.append({"answer": a, "score": score})

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    seen = set()
    final = []

    for r in results:
        if r["score"] > 60 and r["answer"] not in seen:
            final.append(r)
            seen.add(r["answer"])

        if len(final) == top_n:
            break

    return final


@app.post("/chat")
def chat(req: Request):
    matches = find_top_answers(req.message)

    if matches:
        response = "Here are the most relevant answers:\n\n"

        for i, m in enumerate(matches, 1):
            response += f"{i}. {m['answer']} (Confidence: {m['score']}%)\n\n"

        return {"reply": response}

    return {"reply": "I couldn't find a relevant answer. Please rephrase your question."}