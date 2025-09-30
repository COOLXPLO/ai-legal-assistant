from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
from elastic_client import semantic_search
from vertex_ai_client import generate_answer, embed_text

app = FastAPI()

class Query(BaseModel):
    question: str
    top_k: int = 5

@app.post("/qa")
def qa(query: Query):
    # 1) embed question
    q_embedding = embed_text([query.question])[0]
    # 2) search elastic
    hits = semantic_search(q_embedding, top_k=query.top_k)
    # 3) generate using Vertex
    context = [{"case_id": h["source"]["case_id"], "text": h["source"]["text"]} for h in hits]
    answer = generate_answer(query.question, context)
    return {"answer": answer, "sources": [h["source"]["case_id"] for h in hits]}

@app.get("/health")
def health():
    return {"status":"ok"}
