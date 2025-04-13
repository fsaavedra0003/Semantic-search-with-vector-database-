from pymilvus import Collection, connections
from sklearn.preprocessing import normalize
import numpy as np
import yaml
from transformers import BertTokenizer, BertModel

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Load Milvus config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Connect to Milvus
def connect_milvus():
    connections.connect("default", host=config['milvus']['host'], port=config['milvus']['port'])

# Generate BERT embeddings
def generate_embedding(query):
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).numpy()  # Use mean pooling
    return normalize(embedding)

# Search in Milvus collection
def search_in_milvus(query, collection_name="semantic_search"):
    collection = Collection(name=collection_name)
    embedding = generate_embedding(query)
    
    # Perform the search
    results = collection.search(
        data=embedding.tolist(),
        anns_field="embedding",
        param={"metric_type": "COSINE"},
        limit=5
    )
    
    for result in results[0]:
        print(f"ID: {result.id}, Distance: {result.distance}")

def main():
    query = "What is semantic search?"
    connect_milvus()
    search_in_milvus(query)

if __name__ == "__main__":
    main()
