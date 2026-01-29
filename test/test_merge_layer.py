import os
import sys
import logging
import pickle

# Ensure the raptor package is accessible from the test directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor.EmbeddingModels import SBertEmbeddingModel
from raptor.QAModels import UnifiedQAModel
from raptor.SummarizationModels import BaseSummarizationModel

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


class MockSummarizationModel(BaseSummarizationModel):
    def summarize(self, context, max_tokens=150):
        return context[:max_tokens]


def test_merge_tree_builder():
    logging.info("Starting Merge Tree Builder Test...")

    # Load sample text
    with open("demo/sample.txt", "r") as f:
        text = f.read()

    # Initialize Models (Local CPU)
    embedding_model = SBertEmbeddingModel(
        device="cpu", model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"
    )
    qa_model = UnifiedQAModel()

    # Configure for Merge Tree
    # We use small clusters for this small text
    n_clusters = 3
    chunk_size = 100

    RAC = RetrievalAugmentationConfig(
        tree_builder_type="merge",  # New type
        tree_retriever_type="kmeans",
        tb_n_clusters=n_clusters,
        tb_merge_top_k_clusters=2,  # Search top 2 clusters
        tb_merge_top_k_chunks=5,  # Check 5 candidates
        qa_model=qa_model,
        embedding_model=embedding_model,
        summarization_model=MockSummarizationModel(),
        tb_max_tokens=chunk_size,
    )

    logging.info("Building RAPTOR Tree with Merge Layer...")
    RA = RetrievalAugmentation(config=RAC)
    # create the tree
    RA.add_documents(text, use_multithreading=False)

    # Inspect Tree Structure
    tree = RA.tree

    logging.info(f"Tree Num Layers: {tree.num_layers}")

    # Check Layer 0 (Leaves)
    layer0 = tree.layer_to_nodes.get(0, [])
    logging.info(f"Layer 0 (Leaves) count: {len(layer0)}")

    # Check Layer 1 (Merged)
    layer1 = tree.layer_to_nodes.get(1, [])
    logging.info(f"Layer 1 (Merged) count: {len(layer1)}")

    # check if the lengths of layer0 and layer1 are equal
    if len(layer1) != len(layer0):
        logging.warning(
            "Layer 1 count should match Layer 0 count (since we merge each node)."
        )

    # Verify Layer 1 children are from Layer 0
    if layer1:
        sample_node = layer1[0]
        children = sample_node.children
        logging.info(f"Sample Layer 1 Node Children Indices: {children}")
        # Verify indices exist in Layer 0
        for child_idx in children:
            if child_idx not in tree.leaf_nodes:
                logging.error(f"Child {child_idx} not found in leaf nodes!")

    # Check Layer 2 (Clusters)
    layer2 = tree.layer_to_nodes.get(2, [])
    logging.info(f"Layer 2 (Clusters) count: {len(layer2)}")

    if len(layer2) > n_clusters:
        logging.warning(
            f"Layer 2 count {len(layer2)} > requested clusters {n_clusters}"
        )

    # Verify Layer 2 children are from Layer 1
    if layer2:
        sample_cluster = layer2[0]
        children = sample_cluster.children
        logging.info(f"Sample Layer 2 Node Children Indices: {children}")
        # Indices for layer 1 nodes are offset by len(layer0)
        # We can check if they are in layer 1 nodes list
        l1_indices = {n.index for n in layer1}
        for child_idx in children:
            if child_idx not in l1_indices:
                logging.error(
                    f"Child {child_idx} of Layer 2 node not found in Layer 1!"
                )

    # Test Retrieval
    question = "What did Cinderella's father give her?"
    logging.info(f"Testing Retrieval for: {question}")

    answer = RA.answer_question(question=question)
    logging.info(f"Answer: {answer}")

    if tree.num_layers == 2:
        logging.info(
            "Test Passed: Tree has correct depth (0->1->2 means num_layers=2 in current logic)."
        )
    else:
        logging.error(f"Test Failed: Expected num_layers=2, got {tree.num_layers}")


if __name__ == "__main__":
    test_merge_tree_builder()
