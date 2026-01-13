import logging
from typing import List, Tuple

from .Retrievers import BaseRetriever
from .tree_structures import Tree, Node
from .tree_retriever import TreeRetrieverConfig
from .utils import (
    distances_from_embeddings,
    get_embeddings,
    indices_of_nearest_neighbors_from_distances,
    get_text,
)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class KMeansRetrieverConfig(TreeRetrieverConfig):
    def __init__(self, top_k_clusters=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.top_k_clusters = top_k_clusters

    def log_config(self):
        base_summary = super().log_config()
        kmeans_summary = f"""
        Top K Clusters: {self.top_k_clusters}
        """
        return base_summary + kmeans_summary


class KMeansRetriever(BaseRetriever):
    """
    Docstring for KMeansRetriever

    top_k_clusters: how many clusters to consider during retrieval
    """

    def __init__(self, config, tree: Tree) -> None:
        self.tree = tree
        self.embedding_model = config.embedding_model
        self.context_embedding_model = config.context_embedding_model
        self.top_k_clusters = config.top_k_clusters
        self.top_k = config.top_k
        self.tokenizer = config.tokenizer

        logging.info(
            f"Successfully initialized KMeansRetriever with Config {config.log_config()}"
        )

    def create_embedding(self, text: str) -> List[float]:
        return self.embedding_model.create_embedding(text)

    def retrieve(
        self,
        query: str,
        start_layer: int = None,  # Ignored, kept for compatibility
        num_layers: int = None,  # Ignored, kept for compatibility
        top_k: int = 10,
        max_tokens: int = 5000,
        collapse_tree: bool = True,  # Ignored, specific logic used
        return_layer_information: bool = False,
    ) -> str:

        query_embedding = self.create_embedding(query)

        # 1. Get Cluster Centroids (Layer 1)
        if 1 not in self.tree.layer_to_nodes:
            logging.warning(
                "Layer 1 (Clusters) not found. Falling back to simple search on leaves."
            )
            cluster_nodes = []
        else:
            cluster_nodes = self.tree.layer_to_nodes[1]

        # 2. Select closest clusters
        if cluster_nodes:
            # find cluster embeddings
            cluster_embeddings = get_embeddings(
                cluster_nodes, self.context_embedding_model
            )
            # find distances from query to each cluster centroid
            cluster_distances = distances_from_embeddings(
                query_embedding, cluster_embeddings
            )
            # find indices of closest clusters
            cluster_indices = indices_of_nearest_neighbors_from_distances(
                cluster_distances
            )

            # top_k_clusters is a parameter specifically designed for KMeansRetriever
            # In RAPTOR, no such parameter exists because brute force is used
            selected_cluster_indices = cluster_indices[: self.top_k_clusters]
            # selected_clusters is a list of Node objects
            selected_clusters = [cluster_nodes[i] for i in selected_cluster_indices]

            # 3. Gather chunks from selected clusters
            # candidate_chunks_indices is a set of indexes of leaf nodes from candidate clusters
            candidate_chunks_indices = set()
            for cluster in selected_clusters:
                # add all of indexes of the children of all clusters
                candidate_chunks_indices.update(cluster.children)

            # Retrieve actual Node objects for these indices
            # Since leaf nodes are in self.tree.leaf_nodes (dict index->Node)
            # a list of candidate nodes
            # candidate_chunks is a list of Node objects from candidate clusters
            candidate_chunks = [
                self.tree.leaf_nodes[idx]
                for idx in candidate_chunks_indices
                if idx in self.tree.leaf_nodes
            ]
        else:
            # Fallback if no clusters (e.g. tree too small)
            candidate_chunks = list(self.tree.leaf_nodes.values())

        # 4. Brute force compare query with candidate chunks
        if not candidate_chunks:
            return "", [] if return_layer_information else ""

        chunk_embeddings = get_embeddings(
            candidate_chunks, self.context_embedding_model
        )
        chunk_distances = distances_from_embeddings(query_embedding, chunk_embeddings)
        chunk_indices = indices_of_nearest_neighbors_from_distances(chunk_distances)

        # 5. Pick top-K closest chunks
        final_selected_nodes = []
        total_tokens = 0
        print(f"Selecting up to {self.top_k} chunks with max {max_tokens} tokens.")
        for idx in chunk_indices[: self.top_k]:
            print(f"Considering chunk index {idx}.")
            node = candidate_chunks[idx]
            node_tokens = len(self.tokenizer.encode(node.text))

            if total_tokens + node_tokens > max_tokens:
                break

            final_selected_nodes.append(node)
            print(
                f"Selected chunk index {idx} with {node_tokens} tokens. Total so far: {total_tokens + node_tokens} tokens and {len(final_selected_nodes)} chunks."
            )
            total_tokens += node_tokens

        print(
            f"Final selection: {len(final_selected_nodes)} chunks with total {total_tokens} tokens."
        )
        context = get_text(final_selected_nodes)

        if return_layer_information:
            # Leaves are always Layer 0
            layer_information = [node.index for node in final_selected_nodes]
            return context, layer_information

        return context
