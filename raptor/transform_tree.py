"""
If you want to use this file to transform tree, don't need to specify input and output file flag:
    python raptor/transform_tree.py pkl_files/squad_128.pkl pkl_files/squad_128m256v2.pkl --n_clusters 250 --v2 --merge_k_chunks 3 --merge_k_clusters 500
"""

import argparse
import pickle
import logging
import os
import sys

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from raptor.merge_tree_builder import MergeTreeBuilder, MergeTreeConfig
from raptor.tree_structures import Tree
from raptor.EmbeddingModels import BGEM3Model, SBertEmbeddingModel


logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def transform_tree(
    input_path, output_path, n_clusters, merge_top_k_clusters, merge_top_k_chunks
):
    """
    Transform a k-mean tree into a merge tree version 1 based on its leaf nodes.

    :param input_path: input pkl tree path
    :param output_path: Description
    :param n_clusters: Description
    :param merge_top_k_clusters: Description
    :param merge_top_k_chunks: Description
    """
    # load k-mean tree
    logging.info(f"Loading tree from {input_path}")
    try:
        with open(input_path, "rb") as f:
            old_tree = pickle.load(f)
    except FileNotFoundError:
        logging.error(f"Input file not found: {input_path}")
        return
    except Exception as e:
        logging.error(f"Error loading tree: {e}")
        return

    if not isinstance(old_tree, Tree):
        raise ValueError("Loaded object is not a Tree instance")

    logging.info(f"Loaded tree with {old_tree.num_layers} layers.")

    # Extract Layer 0 nodes because in k-mean tree, Layer 0 are the leaf nodes
    if 0 not in old_tree.layer_to_nodes:
        raise ValueError("Tree does not have Layer 0 (leaf nodes).")

    leaf_nodes_list = old_tree.layer_to_nodes[0]
    logging.info(f"Found {len(leaf_nodes_list)} leaf nodes.")

    # Prepare data structures for the new builder
    # We want to discard old upper layers, so we start fresh from Layer 0
    new_layer_to_nodes = {0: leaf_nodes_list}
    new_all_tree_nodes = {node.index: node for node in leaf_nodes_list}

    # Determine embedding model from the first node
    # we use the first embedding keys of the first node
    # Since our application only contains one embedding model per node, this is sufficient
    first_node = leaf_nodes_list[0]
    embedding_model_name = list(first_node.embeddings.keys())[0]
    logging.info(f"Detected embedding model in nodes: {embedding_model_name}")

    # device = (
    #     "mps"
    #     if torch.backends.mps.is_available()
    #     else "cuda" if torch.cuda.is_available() else "cpu"
    # )
    device = "cuda:0"
    logging.info(f"Using device: {device}")

    if "BGEM3" in embedding_model_name:
        embedding_model = BGEM3Model(device=device)
        embedding_models = {embedding_model_name: embedding_model}
    elif "SBert" in embedding_model_name or "MultiQA" in embedding_model_name:
        # Assuming SBert based on common naming
        embedding_model = SBertEmbeddingModel(device=device)
        embedding_models = {embedding_model_name: embedding_model}
    else:
        logging.warning(
            f"Unknown embedding model '{embedding_model_name}'. Defaulting to BGEM3 instantiation mapped to this name."
        )
        embedding_model = BGEM3Model(device=device)
        embedding_models = {embedding_model_name: embedding_model}

    # Configure MergeTreeBuilder
    builder_config = MergeTreeConfig(
        n_clusters=n_clusters,
        merge_top_k_clusters=merge_top_k_clusters,
        merge_top_k_chunks=merge_top_k_chunks,
        tokenizer=None,
        max_tokens=100,
        num_layers=5,
        threshold=0.5,
        top_k=5,
        selection_mode="top_k",
        summarization_length=100,
        summarization_model=None,  # Default DeepSeek? If needed, user can modify.
        embedding_models=embedding_models,
        cluster_embedding_model=embedding_model_name,
    )

    builder = MergeTreeBuilder(builder_config)

    # Construct the tree
    # current_level_nodes should be Dict[int, Node] of the layer we are building ON TOP OF.
    current_level_nodes = new_all_tree_nodes.copy()  # Layer 0 nodes

    # construct tree
    logging.info("Starting Merge Tree construction...")
    builder.construct_tree(current_level_nodes, new_all_tree_nodes, new_layer_to_nodes)

    # Result
    # construct_tree updates new_all_tree_nodes and new_layer_to_nodes in place

    # Determine root nodes (highest layer)
    max_layer = max(new_layer_to_nodes.keys())
    root_nodes = {node.index: node for node in new_layer_to_nodes[max_layer]}

    new_tree = Tree(
        all_nodes=new_all_tree_nodes,
        root_nodes=root_nodes,
        leaf_nodes=leaf_nodes_list,  # Kept original list
        num_layers=max_layer + 1,  # 0-indexed layers, so count is max_layer + 1
        layer_to_nodes=new_layer_to_nodes,
    )

    logging.info(f"New tree constructed with {new_tree.num_layers} layers.")

    # save new tree to the output path
    logging.info(f"Saving to {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(new_tree, f)
    logging.info("Done.")


def transform_tree_v2(
    input_path, output_path, n_clusters, merge_top_k_clusters, merge_top_k_chunks
):
    """
    Transform a k-mean tree into a merge tree version 2 (Exclusive Merge) based on its leaf nodes.
    This version uses the updated MergeTreeBuilder which implements the exclusive merge strategy:
    - Calculates 'hotness' (index count) for all nodes.
    - Merges nodes starting from the hottest.
    - Removes merged nodes from the pool (Exclusive).
    """
    # load k-mean tree
    logging.info(f"Loading tree from {input_path}")
    try:
        with open(input_path, "rb") as f:
            old_tree = pickle.load(f)
    except FileNotFoundError:
        logging.error(f"Input file not found: {input_path}")
        return
    except Exception as e:
        logging.error(f"Error loading tree: {e}")
        return

    if not isinstance(old_tree, Tree):
        raise ValueError("Loaded object is not a Tree instance")

    logging.info(f"Loaded tree with {old_tree.num_layers} layers.")

    # Extract Layer 0 nodes
    if 0 not in old_tree.layer_to_nodes:
        raise ValueError("Tree does not have Layer 0 (leaf nodes).")

    leaf_nodes_list = old_tree.layer_to_nodes[0]
    logging.info(f"Found {len(leaf_nodes_list)} leaf nodes.")

    # Prepare data structures
    new_layer_to_nodes = {0: leaf_nodes_list}
    new_all_tree_nodes = {node.index: node for node in leaf_nodes_list}

    # Determine embedding model
    first_node = leaf_nodes_list[0]
    embedding_model_name = list(first_node.embeddings.keys())[0]
    logging.info(f"Detected embedding model: {embedding_model_name}")

    device = "cuda:0"
    logging.info(f"Using device: {device}")

    if "BGEM3" in embedding_model_name:
        embedding_model = BGEM3Model(device=device)
        embedding_models = {embedding_model_name: embedding_model}
    elif "SBert" in embedding_model_name or "MultiQA" in embedding_model_name:
        embedding_model = SBertEmbeddingModel(device=device)
        embedding_models = {embedding_model_name: embedding_model}
    else:
        logging.warning(
            f"Unknown embedding model '{embedding_model_name}'. Defaulting to BGEM3."
        )
        embedding_model = BGEM3Model(device=device)
        embedding_models = {embedding_model_name: embedding_model}

    # Configure MergeTreeBuilder
    builder_config = MergeTreeConfig(
        n_clusters=n_clusters,
        merge_top_k_clusters=merge_top_k_clusters,
        merge_top_k_chunks=merge_top_k_chunks,
        tokenizer=None,
        max_tokens=100,
        num_layers=5,
        threshold=0.5,
        top_k=5,
        selection_mode="top_k",
        summarization_length=100,
        summarization_model=None,
        embedding_models=embedding_models,
        cluster_embedding_model=embedding_model_name,
    )

    builder = MergeTreeBuilder(builder_config)

    # Construct the tree
    current_level_nodes = new_all_tree_nodes.copy()  # Layer 0 nodes

    logging.info("Starting Merge Tree V2 (Exclusive Merge) construction...")
    builder.construct_tree(
        current_level_nodes,
        new_all_tree_nodes,
        new_layer_to_nodes,
        use_existing_index_counts=True,
    )

    # Determine root nodes
    max_layer = max(new_layer_to_nodes.keys())
    root_nodes = {node.index: node for node in new_layer_to_nodes[max_layer]}

    new_tree = Tree(
        all_nodes=new_all_tree_nodes,
        root_nodes=root_nodes,
        leaf_nodes=leaf_nodes_list,
        num_layers=max_layer + 1,
        layer_to_nodes=new_layer_to_nodes,
    )

    logging.info(f"New tree V2 constructed with {new_tree.num_layers} layers.")

    logging.info(f"Saving to {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(new_tree, f)
    logging.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transform a KMeans Tree to a Merge Tree"
    )
    parser.add_argument("input_tree", help="Path to input pickle file")
    parser.add_argument("output_tree", help="Path to output pickle file")
    parser.add_argument("--n_clusters", type=int, default=5, help="Number of clusters")
    parser.add_argument(
        "--merge_k_clusters", type=int, default=5, help="Merge top K clusters"
    )
    parser.add_argument(
        "--merge_k_chunks", type=int, default=10, help="Merge top K chunks"
    )
    parser.add_argument(
        "--v2", action="store_true", help="Use Version 2 (Exclusive Merge)"
    )

    args = parser.parse_args()

    if args.v2:
        transform_tree_v2(
            args.input_tree,
            args.output_tree,
            args.n_clusters,
            args.merge_k_clusters,
            args.merge_k_chunks,
        )
    else:
        transform_tree(
            args.input_tree,
            args.output_tree,
            args.n_clusters,
            args.merge_k_clusters,
            args.merge_k_chunks,
        )
