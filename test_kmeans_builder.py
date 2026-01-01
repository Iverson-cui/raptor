import logging
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from raptor.kmeans_tree_builder import KMeansTreeBuilder, KMeansTreeConfig
from raptor.EmbeddingModels import MpnetBaseCosModel
from raptor.tree_structures import Tree

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_kmeans_builder():
    # Load sample text
    with open('demo/sample.txt', 'r') as f:
        text = f.read()

    # Use MpnetBaseCosModel as it is a standard choice and likely to work
    # We avoid BGEM3 if it's large, but Mpnet is reasonable.
    embedding_model = MpnetBaseCosModel()
    
    # Configure Builder
    n_clusters = 3
    config = KMeansTreeConfig(
        n_clusters=n_clusters,
        embedding_models={"Mpnet": embedding_model},
        cluster_embedding_model="Mpnet",
        # Mock summarization model not needed as we don't summarize
    )

    builder = KMeansTreeBuilder(config)

    # Build Tree
    logging.info("Building tree...")
    tree = builder.build_from_text(text)

    # Assertions
    logging.info("Verifying tree structure...")
    
    # Check layers
    # Layer 0: Leaves
    # Layer 1: Clusters
    print(f"Tree Layers: {tree.num_layers}")
    print(f"Layer to Nodes keys: {tree.layer_to_nodes.keys()}")
    
    assert 0 in tree.layer_to_nodes, "Layer 0 (leaves) missing"
    assert 1 in tree.layer_to_nodes, "Layer 1 (clusters) missing"
    
    leaves = tree.layer_to_nodes[0]
    clusters = tree.layer_to_nodes[1]
    
    print(f"Number of leaves: {len(leaves)}")
    print(f"Number of clusters: {len(clusters)}")
    
    assert len(leaves) > 0, "No leaves created"
    assert len(clusters) == n_clusters or len(clusters) == len(leaves), "Incorrect number of clusters" 
    # Note: len(clusters) might be < n_clusters if few leaves, but sample.txt is long enough.
    
    # Check cluster children
    all_children = set()
    for cluster in clusters:
        # Check parent node structure
        assert cluster.text.startswith("Cluster"), "Parent text format incorrect"
        assert len(cluster.embeddings) > 0, "Parent embeddings missing"
        assert len(cluster.children) > 0, "Cluster has no children"
        all_children.update(cluster.children)
        
    # Verify all leaves are covered
    leaf_indices = {node.index for node in leaves}
    assert all_children == leaf_indices, "Not all leaves are assigned to a cluster"

    print("Test Passed Successfully!")

if __name__ == "__main__":
    test_kmeans_builder()
