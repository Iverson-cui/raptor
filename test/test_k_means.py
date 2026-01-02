import logging
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from raptor.RetrievalAugmentation import (
    RetrievalAugmentation,
    RetrievalAugmentationConfig,
)
from raptor.EmbeddingModels import MpnetBaseCosModel, SBertEmbeddingModel
from raptor.QAModels import UnifiedQAModel
from raptor.SummarizationModels import (
    BaseSummarizationModel,
    QwenLocalSummarizationModel,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


class MockSummarizationModel(BaseSummarizationModel):
    def summarize(self, context, max_tokens=150):
        # Return a simple truncation as summary to allow the process to continue without API keys
        return "SUMMARY: " + context[:max_tokens] + "..."


def test_k_means_comparison():
    print("=" * 50)
    print("TESTING K-MEANS RAPTOR IMPLEMENTATION")
    print("=" * 50)

    # Load sample text
    with open("demo/sample.txt", "r") as f:
        text = f.read()

    # Initialize Models
    print("Initializing Models...")
    embedding_model = MpnetBaseCosModel()
    # Using UnifiedQAModel (flan-t5-small) for local inference
    qa_model = UnifiedQAModel()
    summarization_model = MockSummarizationModel()

    question = "What did the father bring for Cinderella?"
    print(f"\nQUERY: {question}\n")

    # ----------------------------------------------------------------------
    # 1. K-Means System Test
    # ----------------------------------------------------------------------
    print("-" * 30)
    print("1. Running K-Means System")
    print("-" * 30)

    config_kmeans = RetrievalAugmentationConfig(
        tree_builder_type="kmeans",
        tree_retriever_type="kmeans",
        tb_n_clusters=5,  # 5 clusters
        tr_top_k_clusters=2,  # Search in top 2 closest clusters
        tr_top_k=5,  # Return top 5 leaf nodes
        embedding_model=embedding_model,
        qa_model=qa_model,
    )

    ra_kmeans = RetrievalAugmentation(config_kmeans)

    print("Building K-Means Tree...")
    ra_kmeans.add_documents(text)

    tree_kmeans = ra_kmeans.tree
    print(f"K-Means Tree Layers: {tree_kmeans.num_layers}")

    # Retrieve to inspect details
    print("Retrieving...")
    context_kmeans, layer_info_kmeans = ra_kmeans.retrieve(
        question, return_layer_information=True
    )

    # Answer
    print("Answering...")
    answer_kmeans = ra_kmeans.answer_question(question)

    print("\n--- K-Means Results ---")
    print(f"Answer: {answer_kmeans}")
    print(f"Retrieved {len(layer_info_kmeans)} chunks.")
    print("Top retrieved chunks info:")
    for info in layer_info_kmeans:
        node = tree_kmeans.leaf_nodes[info["node_index"]]
        preview = node.text[:50].replace("\n", " ")
        print(
            f" - Node {info['node_index']} (Layer {info['layer_number']}): {preview}..."
        )

    # ----------------------------------------------------------------------
    # 2. Original System Test (Comparison)
    # ----------------------------------------------------------------------
    print("\n" + "-" * 30)
    print("2. Running Original RAPTOR System")
    print("-" * 30)

    config_original = RetrievalAugmentationConfig(
        tree_builder_type="cluster",
        tree_retriever_type="tree",
        # Use mock summarizer to avoid API requirement
        summarization_model=QwenLocalSummarizationModel(),
        embedding_model=embedding_model,
        qa_model=qa_model,
        tb_max_tokens=100,  # Matches what we probably used implicitly
        tb_num_layers=3,  # Limit layers for speed
    )

    ra_original = RetrievalAugmentation(config_original)

    print("Building Original RAPTOR Tree...")
    ra_original.add_documents(text)

    tree_original = ra_original.tree
    print(f"Original Tree Layers: {tree_original.num_layers}")

    # Answer
    print("Answering...")
    answer_original = ra_original.answer_question(question)

    print("\n--- Original RAPTOR Results ---")
    print(f"Answer: {answer_original}")

    # Compare
    print("\n" + "=" * 50)
    print("COMPARISON")
    print("=" * 50)
    print(f"Query: {question}")
    print(f"K-Means Answer:  {answer_kmeans}")
    print(f"Original Answer: {answer_original}")
    print("=" * 50)


if __name__ == "__main__":
    test_k_means_comparison()
