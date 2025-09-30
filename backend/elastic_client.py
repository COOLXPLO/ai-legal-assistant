from elasticsearch import Elasticsearch, helpers
import os

ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
INDEX_NAME = os.getenv("ES_INDEX", "legal_docs_v1")

es = Elasticsearch(ES_HOST)

def create_index():
    # mapping for dense_vector and metadata
    if es.indices.exists(INDEX_NAME):
        print("Index exists")
        return
    mapping = {
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "title": {"type": "keyword"},
                "case_id": {"type": "keyword"},
                "embedding": {"type": "dense_vector", "dims": 1536}
            }
        }
    }
    es.indices.create(index=INDEX_NAME, body=mapping)
    print("Index created:", INDEX_NAME)

def index_documents(docs):
    actions = [
        {
            "_index": INDEX_NAME,
            "_id": doc["case_id"],
            "_source": doc
        } for doc in docs
    ]
    helpers.bulk(es, actions)
    print(f"Indexed {len(actions)} docs")

def semantic_search(embedding, top_k=5):
    # uses cosine similarity via script_score
    query = {
      "size": top_k,
      "query": {
        "script_score": {
          "query": {"match_all": {}},
          "script": {
            "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
            "params": {"query_vector": embedding}
          }
        }
      }
    }
    res = es.search(index=INDEX_NAME, body=query)
    hits = res["hits"]["hits"]
    return [{"id":h["_id"], "score":h["_score"], "source":h["_source"]} for h in hits]
