import os
import sys
import logging
import time
import argparse
import random
import numpy as np
from datasets import load_dataset, concatenate_datasets

# Ensure the raptor package is accessible from the test directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor.kmeans_tree_builder import KMeansTreeConfig
from raptor.QAModels import UnifiedQAModel
from raptor.EmbeddingModels import SBertEmbeddingModel, BGEM3Model
from raptor.SummarizationModels import BaseSummarizationModel
from raptor.utils import (
    distances_from_embeddings,
    indices_of_nearest_neighbors_from_distances,
    get_embeddings,
)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


class MockSummarizationModel(BaseSummarizationModel):
    def summarize(self, context, max_tokens=150):
        return context[:max_tokens]


def get_dataset_processors(dataset_name):
    """
    Returns extraction function for the specified dataset.
    We only need context extraction for this experiment.
    """
    if dataset_name == "squad":

        def extract_contexts(item):
            return [item["context"]]

    elif dataset_name == "squad_v2":

        def extract_contexts(item):
            return [item["context"]]

    elif dataset_name == "trivia_qa":

        def extract_contexts(item):
            contexts = []
            # only retrieve entity_pages
            if "entity_pages" in item:
                for text in item["entity_pages"].get("wiki_context", []):
                    if text.strip():
                        contexts.append(text)
            return contexts

    else:
        # Default fallback for other datasets if needed, generic text field
        def extract_contexts(item):
            return [item.get("context", item.get("text", ""))]

    return extract_contexts


def run_brute_force_search(
    target_node, leaf_nodes, embedding_model_name, distance_metric, top_k_chunks
):
    """
    Performs brute force search to find closest chunks to the target node.
    """
    logging.info("-" * 40)
    logging.info("Method 1: Brute Force Search")

    start_time = time.time()

    target_embedding = target_node.embeddings[embedding_model_name]

    # Calculate distance to ALL other leaf nodes
    all_leaf_embeddings = get_embeddings(leaf_nodes, embedding_model_name)

    # distances_from_embeddings expects list of list
    dists = distances_from_embeddings(
        target_embedding, all_leaf_embeddings, distance_metric
    )
    sorted_indices = indices_of_nearest_neighbors_from_distances(dists)

    # Filter out self (distance ~ 0)
    bf_results = []
    for idx in sorted_indices:
        node = leaf_nodes[idx]
        if node.index == target_node.index:
            continue
        bf_results.append((node, dists[idx]))
        if len(bf_results) >= top_k_chunks:
            break

    bf_time = time.time() - start_time
    logging.info(f"Brute Force Time: {bf_time:.6f} seconds")

    print(f"\nTop {top_k_chunks} Closest Chunks (Brute Force):")
    for i, (node, dist) in enumerate(bf_results):
        print(
            f"{i+1}. Node {node.index} | Dist: {dist:.4f} | Text: {node.text[:50]}..."
        )

    return bf_results


def run_cluster_based_search(
    target_node,
    tree,
    embedding_model_name,
    distance_metric,
    top_k_clusters,
    top_k_chunks,
):
    """
    Performs cluster-based search to find closest chunks to the target node.
    """
    logging.info("-" * 40)
    logging.info("Method 2: Cluster-based Search")

    start_time = time.time()

    target_embedding = target_node.embeddings[embedding_model_name]

    # 1. Get Clusters (Layer 1)
    if 1 not in tree.layer_to_nodes:
        logging.warning(
            "No clusters found (tree too small?). Falling back to brute force logic."
        )
        return []

    cluster_nodes = tree.layer_to_nodes[1]
    cluster_embeddings = get_embeddings(cluster_nodes, embedding_model_name)

    # Distance from target chunk to cluster centroids
    cluster_dists = distances_from_embeddings(
        target_embedding, cluster_embeddings, distance_metric
    )
    sorted_cluster_indices = indices_of_nearest_neighbors_from_distances(cluster_dists)

    # Select top K clusters
    top_clusters = [cluster_nodes[i] for i in sorted_cluster_indices[:top_k_clusters]]

    # Gather candidate chunks from these clusters
    candidate_indices = set()
    for cluster in top_clusters:
        candidate_indices.update(cluster.children)

    candidate_nodes = [
        tree.leaf_nodes[idx] for idx in candidate_indices if idx in tree.leaf_nodes
    ]

    # Calculate distance only to candidates
    cluster_results = []
    if candidate_nodes:
        candidate_embeddings = get_embeddings(candidate_nodes, embedding_model_name)
        candidate_dists = distances_from_embeddings(
            target_embedding, candidate_embeddings, distance_metric
        )
        sorted_candidate_indices = indices_of_nearest_neighbors_from_distances(
            candidate_dists
        )

        for idx in sorted_candidate_indices:
            node = candidate_nodes[idx]
            if node.index == target_node.index:
                continue
            cluster_results.append((node, candidate_dists[idx]))
            if len(cluster_results) >= top_k_chunks:
                break

    cb_time = time.time() - start_time
    logging.info(f"Cluster-based Time: {cb_time:.6f} seconds")

    print(f"\nTop {top_k_chunks} Closest Chunks (Cluster-based):")
    for i, (node, dist) in enumerate(cluster_results):
        print(
            f"{i+1}. Node {node.index} | Dist: {dist:.4f} | Text: {node.text[:50]}..."
        )

    return cluster_results


def run_experiment(
    dataset_name="squad",
    local_test=True,
    chunk_size=128,
    n_clusters=50,
    top_k_clusters=5,
    top_k_chunks=10,
    target_chunk_idx=None,
    distance_metric="cosine",
    context_limit=None,
):
    logging.info(
        f"Starting experiment: Dataset={dataset_name}, Local={local_test}, Metric={distance_metric}"
    )

    # 1. Load Data
    splits = ["validation"] if local_test else ["train", "validation"]
    loaded_splits = []

    try:
        for split in splits:
            if dataset_name == "trivia_qa":
                # if trivia_qa, only use validation split to keep size manageable
                loaded_splits.append(
                    load_dataset("trivia_qa", "rc", split="validation")
                )  # TriviaQA train is huge
            else:
                loaded_splits.append(load_dataset(dataset_name, split=split))
    except Exception as e:
        logging.error(f"Error loading dataset {dataset_name}: {e}")
        return

    if len(loaded_splits) > 1:
        dataset = concatenate_datasets(loaded_splits)
    else:
        dataset = loaded_splits[0]

    extract_contexts_fn = get_dataset_processors(dataset_name)

    all_contexts = []
    seen_contexts = set()

    # how many chunks to consider for the whole database
    max_contexts = 200 if local_test else (context_limit if context_limit else 50000)

    logging.info(f"Gathering contexts (Limit: {max_contexts})...")
    for item in dataset:
        contexts = extract_contexts_fn(item)
        for ctx in contexts:
            if ctx not in seen_contexts:
                all_contexts.append(ctx)
                seen_contexts.add(ctx)
                if len(all_contexts) >= max_contexts:
                    break
        if len(all_contexts) >= max_contexts:
            break

    full_corpus = "\n\n".join(all_contexts)
    logging.info(f"Corpus prepared with {len(all_contexts)} unique contexts.")

    # 2. Initialize Models
    if local_test:
        embedding_device = "mps"
        embedding_model = SBertEmbeddingModel(device=embedding_device)
        # embedding_model = BGEM3Model(device=embedding_device)
        qa_model = UnifiedQAModel()  # Not really used but required by config
    else:
        embedding_device = "cuda:0"
        embedding_model = BGEM3Model(device=embedding_device)
        qa_model = UnifiedQAModel()

    # 3. Build Tree
    logging.info("Building RAPTOR Tree...")
    # Adjust n_clusters if too small for local test
    if local_test:
        n_clusters = min(n_clusters, 5)

    RAC = RetrievalAugmentationConfig(
        tree_builder_type="kmeans",
        tree_retriever_type="kmeans",
        tb_n_clusters=n_clusters,
        qa_model=qa_model,
        embedding_model=embedding_model,
        summarization_model=MockSummarizationModel(),
        tb_max_tokens=chunk_size,
    )

    RA = RetrievalAugmentation(config=RAC)
    RA.add_documents(full_corpus, use_multithreading=not local_test)

    leaf_nodes = list(RA.tree.leaf_nodes.values())
    if not leaf_nodes:
        logging.error("No leaf nodes created!")
        return

    # 4. Select Target Chunk
    if target_chunk_idx is None or target_chunk_idx >= len(leaf_nodes):
        target_chunk_idx = 0  # Default to first

    target_node = leaf_nodes[target_chunk_idx]

    logging.info(f"Target Chunk Index: {target_node.index}")
    logging.info(f"Target Chunk Text (truncated): {target_node.text[:100]}...")

    embedding_model_name = RAC.tree_builder_config.cluster_embedding_model

    # Run Brute Force Search
    bf_results = run_brute_force_search(
        target_node=target_node,
        leaf_nodes=leaf_nodes,
        embedding_model_name=embedding_model_name,
        distance_metric=distance_metric,
        top_k_chunks=top_k_chunks,
    )

    # Run Cluster-based Search
    cluster_results = run_cluster_based_search(
        target_node=target_node,
        tree=RA.tree,
        embedding_model_name=embedding_model_name,
        distance_metric=distance_metric,
        top_k_clusters=top_k_clusters,
        top_k_chunks=top_k_chunks,
    )

    # Validation: Check overlap
    bf_ids = {n.index for n, _ in bf_results}
    cb_ids = {n.index for n, _ in cluster_results}
    overlap = len(bf_ids.intersection(cb_ids))
    print(f"\nOverlap between methods: {overlap}/{top_k_chunks}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Chunk Distances in RAPTOR")
    parser.add_argument(
        "--local", action="store_true", help="Run in local mode with SQuAD"
    )
    parser.add_argument(
        "--server", action="store_true", help="Run in server mode with TriviaQA"
    )
    # these 4 hyperparameters can be adjusted
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--n_clusters", type=int, default=50)
    parser.add_argument("--top_k_clusters", type=int, default=5)
    parser.add_argument("--top_k_chunks", type=int, default=10)
    parser.add_argument(
        "--metric", type=str, default="cosine", choices=["cosine", "L2", "L1"]
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of contexts (for server mode)",
    )

    args = parser.parse_args()

    if args.server:
        # Server defaults
        run_experiment(
            dataset_name="trivia_qa",
            local_test=False,
            chunk_size=args.chunk_size,
            n_clusters=(
                args.n_clusters if args.n_clusters != 50 else 200
            ),  # Default higher for server
            top_k_clusters=args.top_k_clusters,
            top_k_chunks=args.top_k_chunks,
            distance_metric=args.metric,
            context_limit=args.limit,
        )
    else:
        # Local defaults (or explicit local flag)
        run_experiment(
            dataset_name="squad",
            local_test=True,
            chunk_size=args.chunk_size,
            n_clusters=args.n_clusters,
            top_k_clusters=args.top_k_clusters,
            top_k_chunks=args.top_k_chunks,
            distance_metric=args.metric,
            context_limit=args.limit,
        )
