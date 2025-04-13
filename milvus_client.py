from pymilvus import Collection, connections
import yaml

# Load Milvus config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Connect to Milvus server
def connect_milvus():
    connections.connect("default", host=config['milvus']['host'], port=config['milvus']['port'])

def get_milvus_collection(collection_name):
    collection = Collection(name=collection_name)
    return collection
