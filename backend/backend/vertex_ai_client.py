from google.cloud import aiplatform
import os
import base64

PROJECT = os.getenv("GCP_PROJECT")
REGION = os.getenv("GCP_REGION", "us-central1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "textembedding-gecko@001")  #
GEN_MODEL = os.getenv("GEN_MODEL", "text-bison@001")  

aiplatform.init(project=PROJECT, location=REGION)

def embed_text(texts):
    """
    Return list of embeddings for input text list.
    """
    embedding_client = aiplatform.gapic.PredictionServiceClient()
    endpoint = f"projects/{PROJECT}/locations/{REGION}/publishers/google/models/{EMBED_MODEL}"
    instances = [{"content": t} for t in texts]
    response = embedding_client.predict(endpoint=endpoint, instances=instances)
    embeddings = [resp.predictions[0] if hasattr(resp, "predictions") else resp for resp in response]
   
    return [p['embedding'] for p in response.predictions]

def generate_answer(prompt, context_docs, max_output_tokens=256):
    """
    Construct prompt with retrieved docs and call Vertex generate.
    """
  
    system = "You are a helpful legal assistant. Answer concisely and cite source case_ids in brackets."
    user_prompt = f"{system}\n\nContext:\n"
    for doc in context_docs:
        user_prompt += f"[{doc['case_id']}]: {doc['text']}\n\n"
    user_prompt += f"\nQuestion: {prompt}\nAnswer:"

    client = aiplatform.gapic.PredictionServiceClient()
    endpoint = f"projects/{PROJECT}/locations/{REGION}/publishers/google/models/{GEN_MODEL}"
    instances = [{"content": user_prompt}]
    parameters = {"maxOutputTokens": max_output_tokens}
    response = client.predict(endpoint=endpoint, instances=instances, parameters=parameters)
 
    return response.predictions[0].get("content") if response.predictions else ""
