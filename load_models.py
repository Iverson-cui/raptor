"""
This file is used to pre-load bge-m3 embedding model on the server.
For now(Apr 3) the model loading process is very slow.
"""

from sentence_transformers import SentenceTransformer

# Specify a local path to store the model
model = SentenceTransformer("BAAI/bge-m3")
