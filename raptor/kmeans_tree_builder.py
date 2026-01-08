import logging
import numpy as np
from typing import Dict, List, Set
import faiss

from .tree_builder import TreeBuilder, TreeBuilderConfig
from .tree_structures import Node, Tree
from .utils import get_embeddings, get_node_list

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class KMeansTreeConfig(TreeBuilderConfig):
    """
    config for KMeans Tree Builder

    :var Clusters: Description
    """

    def __init__(self, n_clusters=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # n_clusters: number of clusters to form
        self.n_clusters = n_clusters

    def log_config(self):
        base_summary = super().log_config()
        kmeans_summary = f"""
        N Clusters: {self.n_clusters}
        """
        return base_summary + kmeans_summary


class KMeansTreeBuilder(TreeBuilder):
    """
    Docstring for KMeansTreeBuilder
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        if not isinstance(config, KMeansTreeConfig):
            raise ValueError("config must be an instance of KMeansTreeConfig")
        self.n_clusters = config.n_clusters
        logging.info(
            f"Successfully initialized KMeansTreeBuilder with Config {config.log_config()}"
        )

    def construct_tree(
        self,
        current_level_nodes: Dict[int, Node],
        all_tree_nodes: Dict[int, Node],
        layer_to_nodes: Dict[int, List[Node]],
        use_multithreading: bool = True,
    ) -> Dict[int, Node]:

        logging.info("Using KMeans TreeBuilder (Faiss Accelerated)")

        # Get nodes and their embeddings
        node_list = get_node_list(current_level_nodes)
        if not node_list:
            return {}

        embeddings = get_embeddings(node_list, self.cluster_embedding_model)

        # Convert to numpy float32 for Faiss
        embeddings_np = np.array(embeddings, dtype=np.float32)
        n_samples, d = embeddings_np.shape

        # Check if we have enough samples for k-means
        n_clusters = min(self.n_clusters, n_samples)

        if n_samples == 0:
            return {}

        # Perform KMeans clustering using Faiss
        # spherical=True is recommended for Cosine Similarity (normalized embeddings)
        use_gpu = faiss.get_num_gpus() if (faiss.get_num_gpus() > 0) else 0

        logging.info(
            f"Starting Faiss KMeans (GPU={use_gpu}, spherical=True, k={n_clusters})"
        )

        # d: dimension of the vectors to cluster
        # k: number of clusters
        # spherical means normalize vectors to unit length before clustering
        # nredo: run it multiple times and keep the best
        # gpu=True means using all of gpus available, gpu=3 means using only 3 gpus.
        kmeans = faiss.Kmeans(
            d=d,
            k=n_clusters,
            niter=20,
            nredo=3,
            verbose=True,
            spherical=True,
            seed=42,
            gpu=use_gpu,
        )

        kmeans.train(embeddings_np)

        # Get cluster centers (centroids)
        # centroids is a numpy array of shape (n_clusters, d)
        centroids = kmeans.centroids

        # After kmeans.train, we only get n_clusters cluster centroids
        # BUT we are not assigning every embedding to a cluster yet.
        # To do that, first We search the index (centroids) to find the nearest cluster for each point.
        # index.search returns (distances, indices)
        # indices contains the cluster ID for each sample
        _, labels = kmeans.index.search(embeddings_np, 1)
        # flatten the cluster index to get 1d array
        # this labels is used later to build cluster nodes
        labels = labels.flatten()

        new_level_nodes = {}
        next_node_index = len(all_tree_nodes)  # Start indexing after existing nodes

        # Create parent nodes for each cluster
        for i in range(n_clusters):
            # Identify children for this cluster
            # node_list is sorted by index (from get_node_list), so we can map back
            # find which leaf nodes belong to this cluster
            children_indices = {
                node_list[j].index for j in range(n_samples) if labels[j] == i
            }

            # If a cluster is empty (rare but possible in K-means), skip it
            if not children_indices:
                continue

            # Create parent node
            # Text is a placeholder that doesn't affect the cluster embedding vectors
            # embedding vectors are directly retrieved from centroids[i]
            parent_node = Node(
                text=f"Cluster {i} Centroid",
                index=next_node_index,
                children=children_indices,
                embeddings={self.cluster_embedding_model: centroids[i].tolist()},
            )

            new_level_nodes[next_node_index] = parent_node
            next_node_index += 1

        # Update layer_to_nodes
        # Layer 0 (leaves) is already there. Layer 1 (roots/clusters) is what we just built.
        layer_to_nodes[1] = list(new_level_nodes.values())

        # Update all_tree_nodes
        all_tree_nodes.update(new_level_nodes)

        # Set num_layers to 1 (meaning 1 layer of abstraction/clustering on top of leaves)
        self.num_layers = 1

        return new_level_nodes
