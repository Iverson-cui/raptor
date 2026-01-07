import os
import sys
import logging
import tiktoken
import numpy as np

# Ensure the raptor package is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from raptor import RetrievalAugmentationConfig
from raptor.kmeans_tree_builder import KMeansTreeBuilder, KMeansTreeConfig
from raptor.utils import split_text, get_embeddings
from raptor.EmbeddingModels import SBertEmbeddingModel, BaseEmbeddingModel
from raptor.tree_structures import Node

# Setup basic logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


def run_kmeans_test():
    # 1. Read the demo text
    demo_file_path = os.path.join(os.path.dirname(__file__), "../demo/sample.txt")
    if not os.path.exists(demo_file_path):
        logging.error(f"Demo file not found at {demo_file_path}")
        return

    with open(demo_file_path, "r", encoding="utf-8") as f:
        text = f.read()

    logging.info(f"Loaded text from {demo_file_path} (Length: {len(text)} chars)")

    # 2. Split into chunks (Leaf Nodes)
    # Using the same tokenizer as the default config (cl100k_base)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    chunk_size = 256
    chunks = split_text(text, tokenizer, chunk_size)

    logging.info(f"Split text into {len(chunks)} chunks (Max Tokens: {chunk_size})")

    print("\n" + "=" * 50)
    print("OBTAINED CHUNKS:")
    print("=" * 50)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}:\n{chunk}\n{'-'*20}")

    # 3. Configure KMeans Tree Builder
    # We need to manually setup the config to use our local embedding model and settings

    # Initialize Embedding Model (force CPU for local test compatibility if needed,
    # but let it auto-detect based on env)
    # Using SBert directly to ensure we control the instance
    # Note: On server with GPU, this will run on GPU if device is set.
    # For this script, we'll let the class defaults handle it, or specify if needed.
    # The SBertEmbeddingModel defaults to 'nomic-ai/modernbert-embed-base'

    embedding_model = SBertEmbeddingModel(
        device="mps"
    )  # Force CPU for safety in this specific test script if uncertain, or "cuda" if desired.
    # NOTE: The user asked to use existing files. The existing `RetrievalAugmentationConfig`
    # handles model instantiation. Let's use that to be cleaner.

    n_clusters = 5

    # We create the specific config for the builder
    # Note: We need to pass the embedding model explicitly to avoid re-initialization overhead/issues
    builder_config = KMeansTreeConfig(
        tokenizer=tokenizer,
        max_tokens=chunk_size,
        n_clusters=n_clusters,
        cluster_embedding_model="EMB",  # This must match the key in embedding_models
        embedding_models={"EMB": embedding_model},
    )

    logging.info(f"Initializing KMeansTreeBuilder with {n_clusters} clusters...")
    tree_builder = KMeansTreeBuilder(builder_config)

    # 4. Manually trigger the clustering process
    # The `build_from_text` method in TreeBuilder does: split -> create leaf nodes -> construct_tree
    # We already split the text to print it. Let's reuse `build_from_text` to verify the full flow
    # OR we can manually create nodes and call construct_tree to be more granular.

    # Let's use `build_from_text` as it's the standard entry point and easier to verify standard behavior.
    # It will re-split the text, but that's fine.

    logging.info("Running build_from_text (this performs embedding and clustering)...")
    tree = tree_builder.build_from_text(text)

    # 5. Extract and Print Centroids
    # The centroids are stored in Layer 1 of the tree.
    # Layer 0 = Leaves (Chunks)
    # Layer 1 = Clusters (Roots of the clusters)

    if 1 not in tree.layer_to_nodes:
        logging.error(
            "Layer 1 (Clusters) was not created. Text might be too short for the requested number of clusters."
        )
        return

    cluster_nodes = tree.layer_to_nodes[1]

    print("\n" + "=" * 50)
    print(f"GENERATED CLUSTERS (Layer 1) - Total: {len(cluster_nodes)}")
    print("=" * 50)

    for node in cluster_nodes:
        # The text of a cluster node in KMeansTreeBuilder is just "Cluster X Centroid"
        # The actual centroid vector is in node.embeddings['EMB']
        centroid_vector = node.embeddings["EMB"]
        # Convert to numpy for cleaner printing (truncate)
        vec_preview = np.array(centroid_vector)[:5]  # Just show first 5 dims

        print(f"Node Index: {node.index}")
        print(f"Label: {node.text}")
        print(f"Children Indices (Leaf Nodes): {node.children}")
        print(
            f"Centroid Vector (First 5 dims): {vec_preview} ... [Dim: {len(centroid_vector)}]"
        )
        print("-" * 20)


if __name__ == "__main__":
    run_kmeans_test()
