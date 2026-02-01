import logging
import numpy as np
from typing import Dict, List, Set, Tuple
from sklearn.cluster import KMeans
import faiss

from .tree_builder import TreeBuilder, TreeBuilderConfig
from .kmeans_tree_builder import KMeansTreeBuilder, KMeansTreeConfig
from .tree_structures import Node
from .utils import (
    get_embeddings,
    get_node_list,
    distances_from_embeddings,
    indices_of_nearest_neighbors_from_distances,
)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class MergeTreeConfig(KMeansTreeConfig):
    """
    Config for MergeTreeBuilder.
    Inherits from KMeansTreeConfig.
    """

    def __init__(
        self,
        merge_top_k_clusters=5,
        merge_top_k_chunks=10,  # Number of candidates to check, though we pick top 1 to merge
        *args,
        **kwargs,
    ):
        """
        Docstring for __init__

        :param self: Description
        :param merge_top_k_clusters: number of top clusters to consider for merging
        :param merge_top_k_chunks: Number of candidates to check, though we pick top 1 to merge
        :param args: Description
        :param kwargs: Description

        TODO: make them configurable
        merge_top_k_* parameters are extra parameters for MergeTreeBuilder. Now they are defaulted.
        """
        super().__init__(*args, **kwargs)
        self.merge_top_k_clusters = merge_top_k_clusters
        self.merge_top_k_chunks = merge_top_k_chunks

    def log_config(self):
        base_summary = super().log_config()
        merge_summary = f"""
        Merge Top K Clusters: {self.merge_top_k_clusters}
        Merge Top K Chunks: {self.merge_top_k_chunks}
        """
        return base_summary + merge_summary


class MergeTreeBuilder(KMeansTreeBuilder):
    """
    MergeTreeBuilder implements a 3-layer tree construction:
    Layer 0: Original Chunks
    Layer 1: Merged Chunks (merged with nearest neighbor)
    Layer 2: Clusters of Layer 1
    """

    def __init__(self, config) -> None:
        # We initialize with KMeansTreeBuilder's init logic
        # but check for MergeTreeConfig
        if not isinstance(config, MergeTreeConfig):
            # If passed a KMeansTreeConfig by accident, we can try to adapt or raise error
            # For safety, let's assume strict typing or provide defaults if missing
            raise ValueError("config must be an instance of MergeTreeConfig")

        super().__init__(config)
        self.merge_top_k_clusters = config.merge_top_k_clusters
        # merge_top_k_chunks is currently unused and only merge_top_k_chunks is used
        self.merge_top_k_chunks = config.merge_top_k_chunks

        logging.info(
            f"Successfully initialized MergeTreeBuilder with Config {config.log_config()}"
        )

    def _perform_kmeans(self, embeddings_np, n_clusters):
        """
        Helper to run KMeans and return centroids and labels.
        Switches between FAISS (Server/GPU) and Sklearn (Local/CPU).
        Args:
            embeddings_np: np.ndarray of shape (n_samples, n_features) containing embeddings of leaf nodes
            n_clusters: int, number of clusters where each element is an integer indicating which cluster each input embedding belongs to.

        Returns:
            centroids: np.ndarray of shape (n_clusters, n_features)
            labels: np.ndarray of shape (n_samples,)
        """
        try:
            num_gpus = faiss.get_num_gpus()
        except Exception:
            num_gpus = 0

        logging.info(
            f"Server mode detected (GPUs={num_gpus}). Using FAISS for clustering."
        )
        # Explicitly normalize for FAISS spherical clustering
        # We copy to avoid modifying the original array in place if it's used elsewhere,
        # though here it's passed by value (but numpy array is ref).
        # To be safe for the caller, let's normalize a copy or just normalize in place if we know it's safe.
        # In construct_tree, we create embeddings_np_layer0 specifically for this call.

        # Note: faiss.normalize_L2 is in-place.
        # If we don't want to affect the caller's array, we should copy.
        # embeddings_for_faiss is the copy
        embeddings_for_faiss = embeddings_np.copy()
        # normalize the copy vectors
        faiss.normalize_L2(embeddings_for_faiss)

        kmeans = faiss.Kmeans(
            d=embeddings_for_faiss.shape[1],
            k=n_clusters,
            niter=20,
            nredo=1,
            verbose=False,
            spherical=True,
            seed=42,
            gpu=num_gpus,
            min_points_per_centroid=1,
        )
        kmeans.train(embeddings_for_faiss)
        # assigns each embedding to its nearest centroid
        _, labels = kmeans.index.search(embeddings_for_faiss, 1)
        return kmeans.centroids, labels.flatten()

    def construct_tree(
        self,
        current_level_nodes: Dict[int, Node],
        all_tree_nodes: Dict[int, Node],
        layer_to_nodes: Dict[int, List[Node]],
        use_multithreading: bool = True,
        use_existing_index_counts: bool = False,
    ) -> Dict[int, Node]:

        logging.info("Using Merge TreeBuilder")

        # --- Step 1: Prepare Layer 0 Embeddings ---
        node_list_layer0 = get_node_list(current_level_nodes)
        if not node_list_layer0:
            return {}

        # extract embeddings from each node
        # embeddings_layer0 contains a list of embeddings (list of list)
        embeddings_layer0 = get_embeddings(
            node_list_layer0, self.cluster_embedding_model
        )
        # convert embeddings to np array
        embeddings_np_layer0 = np.array(embeddings_layer0, dtype=np.float32)
        # n_samples is the number of leaf nodes
        n_samples = len(node_list_layer0)

        # --- Step 2: Build Auxiliary Clusters for Search ---
        # We need clusters to perform "cluster based search"
        # We use self.n_clusters as the target number of clusters for this aux step too
        # TODO: The 2 clustering use the same n_clusters parameters because we have the same chunks
        n_clusters_aux = min(self.n_clusters, n_samples)
        if n_clusters_aux == 0:
            return {}

        logging.info(
            f"Step 1: Building auxiliary clusters (k={n_clusters_aux}) for neighbor search..."
        )
        aux_centroids, aux_labels = self._perform_kmeans(
            embeddings_np_layer0, n_clusters_aux
        )

        # Organize nodes into these aux clusters
        # cluster_to_node_indices is a dict mapping cluster index to list of node indices in that cluster
        # because we haven't assign nodes to clusters yet
        cluster_to_node_indices = {i: [] for i in range(n_clusters_aux)}
        for idx, label in enumerate(aux_labels):
            cluster_to_node_indices[label].append(idx)  # idx in node_list_layer0

        # --- Step 3: Create Layer 1 (Merged Nodes) ---
        logging.info("Step 2: Merging chunks with nearest neighbors...")
        new_level_nodes = {}
        next_node_index = len(all_tree_nodes)

        # Pre-calculate distances from all nodes to all centroids
        # shape: (n_samples, n_clusters)
        dists_to_centroids = []
        # for every leaf node, compute distances to each cluster centroid
        # calculate the distances of every layer 0 nodes to each auxiliary centroid
        logging.info(f"Calculating distances for {n_samples} nodes to {n_clusters_aux} centroids...")
        for i in range(n_samples):
            dists = distances_from_embeddings(
                embeddings_layer0[i], aux_centroids, distance_metric="cosine"
            )
            dists_to_centroids.append(dists)

            # Log progress every 100 nodes
            if (i + 1) % 100 == 0:
                logging.info(f"Distance calculation progress: {i + 1}/{n_samples} nodes processed")

        logging.info(f"Distance calculation complete for all {n_samples} nodes")

        # TODO: This code is said to be more efficient but needs testing
        # # ...existing code...
        # # Pre-calculate distances from all nodes to all centroids
        # # shape: (n_samples, n_clusters)
        # if distance_metric == "cosine":
        #     # Normalize embeddings for cosine similarity
        #     embeddings_norm = embeddings_np_layer0 / np.linalg.norm(
        #         embeddings_np_layer0, axis=1, keepdims=True
        #     )
        #     centroids_norm = aux_centroids / np.linalg.norm(
        #         aux_centroids, axis=1, keepdims=True
        #     )
        #     # Cosine similarity via dot product, then convert to distance
        #     similarities = embeddings_norm @ centroids_norm.T
        #     dists_to_centroids = 1 - similarities  # Convert similarity to distance
        # else:
        #     # Fallback to loop for non-cosine metrics
        #     dists_to_centroids = []
        #     for i in range(n_samples):
        #         dists = distances_from_embeddings(
        #             embeddings_layer0[i],
        #             aux_centroids.tolist(),
        #             distance_metric=distance_metric,
        #         )
        #         dists_to_centroids.append(dists)
        #     dists_to_centroids = np.array(dists_to_centroids)
        # # ...existing code...

        # --- Pre-calculate Neighbors and Index Counts (Conditional) ---
        if not use_existing_index_counts:
            logging.info("Pre-calculating neighbors to determine index counts...")

            # target_to_neighbor is a dict mapping target node index(int) to its best neighbor Node
            target_to_neighbor = {}
            # index_counts is a dict mapping node index(int) to how many times it was chosen as neighbor(int)
            index_counts = {node.index: 0 for node in node_list_layer0}

            # for every node in layer 0
            for i, target_node in enumerate(node_list_layer0):
                # obtain i-th embedding
                target_embedding = embeddings_layer0[i]

                # 3a. Find top K clusters
                # dists_to_centroids[i] is array of distances to each cluster
                cluster_dists = dists_to_centroids[i]
                sorted_cluster_indices = np.argsort(
                    cluster_dists
                )  # Ascending order (smallest distance first)

                # In this step we choose self.merge_top_k_clusters clusters to search for neighbors
                top_clusters = sorted_cluster_indices[: self.merge_top_k_clusters]

                # 3b. Gather candidate nodes
                candidate_indices_local = []
                # append all of the node indices in the top clusters in candidate_indices_local
                for c_idx in top_clusters:
                    # gather node indices in the candidate clusters
                    candidate_indices_local.extend(cluster_to_node_indices[c_idx])

                # Filter candidates (exclude self)
                candidate_indices_local = [
                    idx for idx in candidate_indices_local if idx != i
                ]

                best_neighbor_node = None

                if candidate_indices_local:
                    # 3c. Find nearest neighbor among candidates
                    # candidate_embeddings is a list of embeddings
                    # includes all of the node embeddings in candidate_indices_local
                    candidate_embeddings = [
                        embeddings_layer0[idx] for idx in candidate_indices_local
                    ]

                    # distances_from_embeddings expects list of list as second arg
                    # returns list of distances
                    dists_candidates = distances_from_embeddings(
                        target_embedding, candidate_embeddings, distance_metric="cosine"
                    )

                    # Find min distance
                    min_dist_idx = np.argmin(dists_candidates)
                    # find min distance index
                    best_neighbor_local_idx = candidate_indices_local[min_dist_idx]
                    # find min distance node
                    best_neighbor_node = node_list_layer0[best_neighbor_local_idx]
                else:
                    # Fallback if no candidates (e.g. single node in top clusters? unlikely)
                    logging.warning(
                        f"No candidates found for node {target_node.index}. Duplicating node."
                    )
                    best_neighbor_node = target_node

                # after finding the best neighbor node, update 2 dicts
                target_to_neighbor[target_node.index] = best_neighbor_node
                index_counts[best_neighbor_node.index] += 1

                if (i + 1) % 100 == 0:
                    logging.info(f"Pre-calculated {i + 1}/{n_samples} neighbors")

            # Update nodes.attribute with index counts
            for node in node_list_layer0:
                node.index_count = index_counts[node.index]
        else:
            logging.info("Using existing index_counts from nodes. Skipping pre-calculation.")
            # Ensure all nodes have index_count
            for node in node_list_layer0:
                if not hasattr(node, "index_count"):
                    node.index_count = 0

        # Sort nodes by index_count descending (Hot nodes first)
        sorted_nodes = sorted(
            node_list_layer0, key=lambda n: n.index_count, reverse=True
        )

        logging.info(
            "Step 2: Merging chunks (sorted by index frequency, exclusive merge)..."
        )

        # Mapping to access embeddings/dists by node.index
        # it's like 42:0 103:1 7:2 etc.
        node_index_to_matrix_idx = {
            node.index: i for i, node in enumerate(node_list_layer0)
        }
        used_indices = set()

        # Iterate over sorted nodes to create merged nodes
        for i, target_node in enumerate(sorted_nodes):
            if target_node.index in used_indices:
                continue

            # We need to find a partner from UNUSED nodes
            matrix_idx = node_index_to_matrix_idx[target_node.index]
            # obtain the embedding vector of the target node
            target_embedding = embeddings_layer0[matrix_idx]

            # obtain the cluster distance of the target node
            cluster_dists = dists_to_centroids[matrix_idx]
            # sort the cluster distances to get top clusters
            sorted_cluster_indices = np.argsort(cluster_dists)
            # obtain the top clusters
            top_clusters = sorted_cluster_indices[: self.merge_top_k_clusters]

            # obtain the candidate chunks by extending node indices in the top clusters
            candidate_matrix_indices = []
            for c_idx in top_clusters:
                candidate_matrix_indices.extend(cluster_to_node_indices[c_idx])

            # Filter candidates: Must not be used, must not be self
            valid_candidates_matrix_idxs = [
                idx
                for idx in candidate_matrix_indices
                if node_list_layer0[idx].index not in used_indices
                and node_list_layer0[idx].index != target_node.index
            ]

            best_neighbor_node = None

            if valid_candidates_matrix_idxs:
                # Find best among valid candidates
                # gather candidate embedding vectors
                candidate_embeddings = [
                    embeddings_layer0[idx] for idx in valid_candidates_matrix_idxs
                ]
                # calculate distance embeddings
                dists_candidates = distances_from_embeddings(
                    target_embedding, candidate_embeddings, distance_metric="cosine"
                )
                # find closest from candidate_embeddings
                min_dist_idx = np.argmin(dists_candidates)
                # we cannot directly use min_dist_idx in the node_list_layer0 because a node's position is not necessarily its index
                # So we need to use matrix indexes to do conversion
                best_neighbor_matrix_idx = valid_candidates_matrix_idxs[min_dist_idx]
                best_neighbor_node = node_list_layer0[best_neighbor_matrix_idx]

                # Merge
                merged_text = f"{target_node.text} {best_neighbor_node.text}"
                children = {target_node.index, best_neighbor_node.index}

                # Mark partner as used
                used_indices.add(best_neighbor_node.index)

            else:
                # No partner found (orphan or all neighbors taken)
                # Promote as single node
                # logging.info(f"Node {target_node.index} has no available neighbors. Promoting as single node.")
                merged_text = target_node.text
                children = {target_node.index}

            # Create the new node (Merged or Single)
            index, new_node = self.create_node(next_node_index, merged_text, children)
            new_level_nodes[index] = new_node
            next_node_index += 1

            # Mark self as used
            used_indices.add(target_node.index)

            if (len(new_level_nodes)) % 100 == 0:
                logging.info(f"Created {len(new_level_nodes)} Layer 1 nodes so far...")

        # Register Layer 1
        layer_to_nodes[1] = list(new_level_nodes.values())
        all_tree_nodes.update(new_level_nodes)

        # --- Step 4: Create Layer 2 (Clusters of Merged Nodes) ---
        logging.info("Step 3: Clustering merged nodes (Layer 2)...")

        # Get embeddings of Layer 1
        node_list_layer1 = get_node_list(new_level_nodes)
        embeddings_layer1 = get_embeddings(
            node_list_layer1, self.cluster_embedding_model
        )
        embeddings_np_layer1 = np.array(embeddings_layer1, dtype=np.float32)

        n_samples_l1 = len(node_list_layer1)
        # use half of self.n_clusters for Layer 2
        n_clusters_final = min(self.n_clusters // 2, n_samples_l1)

        if n_clusters_final > 0:
            centroids_final, labels_final = self._perform_kmeans(
                embeddings_np_layer1, n_clusters_final
            )

            # Create Layer 2 Nodes (Cluster Centroids)
            layer2_nodes = {}
            for i in range(n_clusters_final):
                children_indices = {
                    node_list_layer1[j].index
                    for j in range(n_samples_l1)
                    if labels_final[j] == i
                }

                if not children_indices:
                    continue

                # Create parent node for the cluster centroid
                parent_node = Node(
                    text=f"Cluster {i} Centroid",
                    index=next_node_index,
                    children=children_indices,
                    embeddings={
                        self.cluster_embedding_model: centroids_final[i].tolist()
                    },
                )

                layer2_nodes[next_node_index] = parent_node
                next_node_index += 1

            # Register Layer 2
            layer_to_nodes[2] = list(layer2_nodes.values())
            all_tree_nodes.update(layer2_nodes)
            self.num_layers = 2  # 0->1->2

            return layer2_nodes
        else:
            return new_level_nodes
