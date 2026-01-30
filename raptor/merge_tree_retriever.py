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


class MergeTreeRetrieverConfig(TreeRetrieverConfig):
    def __init__(self, top_k_clusters=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.top_k_clusters = top_k_clusters

    def log_config(self):
        base_summary = super().log_config()
        merge_summary = f"""
        Top K Clusters: {self.top_k_clusters}
        """
        return base_summary + merge_summary


class MergeTreeRetriever(BaseRetriever):
    """
    Retriever for Merge Tree structure (3 layers).
    Layer 2: Cluster Centroids (Root)
    Layer 1: Merged Chunks (Target for retrieval)
    Layer 0: Original Chunks (Untouched)

    Logic:
    1. Select closest clusters from Layer 2.
    2. Select closest merged chunks (Layer 1) from the children of selected clusters.
    """

    def __init__(self, config, tree: Tree) -> None:
        self.tree = tree
        self.embedding_model = config.embedding_model
        self.context_embedding_model = config.context_embedding_model
        self.top_k_clusters = config.top_k_clusters
        self.top_k = config.top_k
        self.tokenizer = config.tokenizer

        logging.info(
            f"Successfully initialized MergeTreeRetriever with Config {config.log_config()}"
        )

    def create_embedding(self, text: str) -> List[float]:
        return self.embedding_model.create_embedding(text)

    def retrieve(
        self,
        query: str,
        start_layer: int = None,  # Ignored
        num_layers: int = None,  # Ignored
        top_k: int = 10,
        max_tokens: int = 5000,
        collapse_tree: bool = True,  # Ignored
        return_layer_information: bool = False,
    ) -> str:
        """
        This function retrieves relevant chunks based on the query using the Merge Tree structure.
        Retrieval is done in two main steps:
        1. Select closest clusters from Layer 2 (Cluster Centroids).
        2. From the selected clusters, gather Layer 1 nodes (Merged Chunks) and select the closest ones to the query.

        The first step is done in retrieve but the second step is handled by helper function _select_chunks.

        :param self: Description
        :param query: Description
        :type query: str
        :param start_layer: Description
        :type start_layer: int
        :param num_layers: Description
        :type num_layers: int
        :param top_k: Description
        :type top_k: int
        :param max_tokens: Description
        :type max_tokens: int
        :param collapse_tree: Description
        :type collapse_tree: bool
        :param return_layer_information: Description
        :type return_layer_information: bool
        :return: Description
        :rtype: str
        """

        query_embedding = self.create_embedding(query)

        # 1. Get Cluster Centroids (Layer 2)
        # In Merge Tree, clusters are at Layer 2
        if 2 not in self.tree.layer_to_nodes:
            logging.warning(
                "Layer 2 (Clusters) not found in Merge Tree. Falling back to Layer 1 or Layer 0 search."
            )
            # Fallback strategy: if Layer 2 missing, maybe it's just Layer 1 and 0?
            if 1 in self.tree.layer_to_nodes:
                cluster_nodes = (
                    []
                )  # No clusters, treat Layer 1 as "leaves" for brute force?
                candidate_chunks = list(self.tree.layer_to_nodes[1])
            else:
                candidate_chunks = list(self.tree.leaf_nodes.values())

            # Skip cluster selection if we fell back
            if 2 not in self.tree.layer_to_nodes:
                return self._select_chunks(
                    query_embedding,
                    candidate_chunks,
                    max_tokens,
                    return_layer_information,
                )

        else:
            cluster_nodes = self.tree.layer_to_nodes[2]

        # 2. Select closest clusters
        # find cluster embeddings
        cluster_embeddings = get_embeddings(cluster_nodes, self.context_embedding_model)
        # find distances from query to each cluster centroid
        cluster_distances = distances_from_embeddings(
            query_embedding, cluster_embeddings
        )
        # find indices of closest clusters
        cluster_indices = indices_of_nearest_neighbors_from_distances(cluster_distances)

        # In retrieval stage, how many clusters to consider is determined by parameter self.top_k_clusters
        selected_cluster_indices = cluster_indices[: self.top_k_clusters]
        selected_clusters = [cluster_nodes[i] for i in selected_cluster_indices]

        # 3. Gather chunks (Layer 1 nodes) from selected clusters
        candidate_chunks_indices = set()
        for cluster in selected_clusters:
            # cluster.children contains indices of Layer 1 nodes
            candidate_chunks_indices.update(cluster.children)

        # Retrieve actual Node objects for these indices
        # We look up in all_nodes. Ideally we know they are in Layer 1.
        candidate_chunks = [
            self.tree.all_nodes[idx]
            for idx in candidate_chunks_indices
            if idx in self.tree.all_nodes
        ]

        # 4 & 5. Select closest chunks from candidates
        return self._select_chunks(
            query_embedding, candidate_chunks, max_tokens, return_layer_information
        )

    def _select_chunks(
        self, query_embedding, candidate_chunks, max_tokens, return_layer_information
    ):
        """
        This functions handles selecting closest chunks from candidate chunks.
        """
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

        # how many chunks to retrieve is determined by self.top_k
        for idx in chunk_indices[: self.top_k]:
            node = candidate_chunks[idx]
            node_tokens = len(self.tokenizer.encode(node.text))

            if total_tokens + node_tokens > max_tokens:
                break

            final_selected_nodes.append(node)
            total_tokens += node_tokens

        context = get_text(final_selected_nodes)

        if return_layer_information:
            layer_information = [
                (node.index, len(self.tokenizer.encode(node.text)))
                for node in final_selected_nodes
            ]
            return context, layer_information

        return context
