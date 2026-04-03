from sentence_transformers import SentenceTransformer

# Specify a local path to store the model
model = SentenceTransformer("BAAI/bge-m3", cache_dir="~/models_cache/")
