import os, glob, json
from elastic_client import create_index, index_documents
from vertex_ai_client import embed_text
from tqdm import tqdm

def load_files(path="..../sample_data"):
    files = glob.glob(os.path.join(path, "*.txt"))
    docs = []
    for f in files:
        text = open(f, "r", encoding="utf-8").read()
        docs.append({"case_id": os.path.basename(f), "title": os.path.basename(f), "text": text})
    return docs

def chunk_and_embed(doc_texts, chunk_size=800):
    # naive fixed-length chunking by words
    chunks = []
    for d in doc_texts:
        words = d["text"].split()
        for i in range(0, len(words), chunk_size):
            chunk_text = " ".join(words[i:i+chunk_size])
            chunks.append({"case_id": d["case_id"] + f"_{i//chunk_size}", "title": d["title"], "text": chunk_text})
    # compute embeddings in batches
    texts = [c["text"] for c in chunks]
    embeddings = embed_text(texts)
    for c, emb in zip(chunks, embeddings):
        c["embedding"] = emb
    return chunks

if __name__ == "__main__":
    create_index()
    docs = load_files(path="../sample_data")
    chunks = chunk_and_embed(docs)
    index_documents(chunks)
