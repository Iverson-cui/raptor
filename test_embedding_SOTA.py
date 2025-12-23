from sentence_transformers import SentenceTransformer
import torch

# 1. Load the model (Automatically uses MPS on your Mac)
# This model is ~149M parameters (~300MB on disk/RAM)
model = SentenceTransformer(
    "nomic-ai/modernbert-embed-base", trust_remote_code=True, local_files_only=True
)

# 2. Your 10 sentences
# For Nomic, if you aren't doing Query-vs-Doc search,
# 'search_query:' is the standard prefix for general similarity.
sentences = [
    "search_query: Linear algebra is the foundation of machine learning.",
    "search_query: Vectors can be added and scaled in a vector space.",
    "search_query: Eigenvalues are critical for understanding stability.",
    "search_query: The cat is sleeping on the warm radiator.",
    "search_query: Neural networks use matrix multiplication at every layer.",
    "search_query: Python is a popular language for data science.",
    "search_query: Singular Value Decomposition is used for compression.",
    "search_query: The weather is quite sunny for December.",
    "search_query: Transformers changed the field of NLP forever.",
    "search_query: A matrix is a rectangular array of numbers.",
]

# 3. Generate Embeddings
# 'normalize_embeddings=True' ensures vectors have a magnitude of 1
embeddings = model.encode(sentences, normalize_embeddings=True)

# 4. Print the result
print(f"Total Sentences: {len(embeddings)}")
print(f"Vector Dimensions: {embeddings.shape[1]}")  # Output: 768
print("-" * 30)

for i, vec in enumerate(embeddings):
    # Printing just the first 3 dimensions for readability
    print(f"Sentence {i+1} Vector: {vec[:3]} ... [Total 768 dims]")
