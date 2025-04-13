
#### **`src/embedder.py`**

```python
import torch
from transformers import BertTokenizer, BertModel
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType
from sklearn.preprocessing import normalize
import numpy as np
import yaml
import os

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Load Milvus config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set up Milvus connection
from pymilvus import connections

def connect_milvus():
    connections.connect("default", host=config['milvus']['host'], port=config['milvus']['port'])

# Define Milvus collection schema
def create_milvus_collection():
    fields = [
        FieldSchema(name="text", dtype=DataType.STRING, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    ]
    schema = CollectionSchema(fields, description="Semantic Search")
    collection = Collection(name=config['milvus']['collection_name'], schema=schema)
    return collection

# Generate BERT embeddings
def generate_embeddings(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()  # Use mean pooling
    return normalize(embeddings)  # Normalize embeddings

# Insert embeddings into Milvus
def insert_embeddings(collection, texts, embeddings):
    data = [
        texts,
        embeddings.tolist()
    ]
    collection.insert(data)
    collection.load()

def main():
    texts = []
    with open('data/sample_data.txt', 'r') as f:
        texts = f.readlines()
    
    embeddings = generate_embeddings(texts)
    connect_milvus()
    collection = create_milvus_collection()
    insert_embeddings(collection, texts, embeddings)

if __name__ == "__main__":
    main()
