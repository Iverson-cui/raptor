import logging
import pickle
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List, Set

from .cluster_utils import ClusteringAlgorithm, RAPTOR_Clustering
from .tree_builder import TreeBuilder, TreeBuilderConfig
from .tree_structures import Node, Tree
from .utils import (distances_from_embeddings, get_children, get_embeddings,
                    get_node_list, get_text,
                    indices_of_nearest_neighbors_from_distances, split_text)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


# ClusterTreeBuilder and ClusterTreeConfig is just one type of TreeBuilder and TreeBuilderConfig
# TreeBuilder and TreeBuilderConfig are base classes and support multiple child classes, i.e. multiple types of tree builders. Cluster is one of them, you can create your own ways of building trees
class ClusterTreeConfig(TreeBuilderConfig):
    def __init__(
        self,
        reduction_dimension=10,
        clustering_algorithm=RAPTOR_Clustering,  # Default to RAPTOR clustering
        clustering_params={},  # Pass additional params as a dict
        *args,
        **kwargs,
    ):
        # ClusterTreeConfig has 3 extra parameters compared to TreeBuilderConfig base class
        super().__init__(*args, **kwargs)
        self.reduction_dimension = reduction_dimension
        self.clustering_algorithm = clustering_algorithm
        self.clustering_params = clustering_params

    def log_config(self):
        base_summary = super().log_config()
        cluster_tree_summary = f"""
        Reduction Dimension: {self.reduction_dimension}
        Clustering Algorithm: {self.clustering_algorithm.__name__}
        Clustering Parameters: {self.clustering_params}
        """
        return base_summary + cluster_tree_summary


class ClusterTreeBuilder(TreeBuilder):
    def __init__(self, config) -> None:
        super().__init__(config)

        if not isinstance(config, ClusterTreeConfig):
            raise ValueError("config must be an instance of ClusterTreeConfig")
        self.reduction_dimension = config.reduction_dimension
        self.clustering_algorithm = config.clustering_algorithm
        self.clustering_params = config.clustering_params

        logging.info(
            f"Successfully initialized ClusterTreeBuilder with Config {config.log_config()}"
        )

    def construct_tree(
        self,
        current_level_nodes: Dict[int, Node],
        all_tree_nodes: Dict[int, Node],
        layer_to_nodes: Dict[int, List[Node]],
        use_multithreading: bool = False,
    ) -> Dict[int, Node]:
        """
        Step 4 (Layer-by-Layer Construction):
        This method iterates to build the tree from the bottom up.

        Process for each layer:
        1. Clustering: Group 'current_level_nodes' based on semantic similarity (embeddings).
        2. Summarization: For each cluster, generate a summary text.
        3. Node Creation: Create a new parent node containing the summary.
        4. Linkage: The new node becomes the parent of the clustered nodes.
        5. Repeat: The new nodes become 'current_level_nodes' for the next layer.

        This function returns the root nodes of the constructed tree, while leaf nodes are modified in-place.

        When called initially, current_level_nodes and all_tree_nodes contain the same memory. Later all_tree_nodes is updated in-place and current_level_nodes rebinds to another local variables.
        """
        logging.info("Using Cluster TreeBuilder")

        # next_node_index helps assign unique indices to new parent nodes no matter which layer the node is in
        next_node_index = len(all_tree_nodes)

        def process_cluster(
            cluster, new_level_nodes, next_node_index, summarization_length, lock
        ):
            """
            Given a list of nodes in a cluster, summarize their texts and create a new parent node.

             :param cluster: List[Node] that we want to summarize into a parent node
             :param new_level_nodes: a shared dictionary to store newly created parent nodes
             :param next_node_index: The index to assign to the newly created parent node
             :param summarization_length: maximum tokens for summarization
             :param lock: Ensures thread-safe access to new_level_nodes when multithreading is enabled
            """
            # concatenate all texts from the cluster into node_texts for summarization
            node_texts = get_text(cluster)

            # logging.info(f"Summarization model type: {type(self.summarization_model)}")
            # logging.info(f"Node Texts: {node_texts}")

            # defaults to GPT3Turbo to summarize
            summarized_text = self.summarize(
                context=node_texts,
                max_tokens=summarization_length,
            )

            # # 将待检查的变量赋值给临时变量，方便操作
            # obj = summarized_text

            # # 打印详细信息
            # print(
            #     f"DEBUG CHECK -> Type: {type(obj)}, "
            #     f"Length: {len(obj) if obj is not None else 'N/A'}, "
            #     f"Value: {repr(obj)[:100]}..."
            # )  # 只显示前100个字符防止刷屏
            # logging.info(
            #     f"Node Texts Length: {len(self.tokenizer.encode(node_texts))}, Summarized Text Length: {len(self.tokenizer.encode(summarized_text))}"
            # )

            # create new parent node with child nodes the cluster
            __, new_parent_node = self.create_node(
                next_node_index, summarized_text, {node.index for node in cluster}
            )

            with lock:
                new_level_nodes[next_node_index] = new_parent_node

        for layer in range(self.num_layers):

            # every layer will have its own set of new level nodes
            new_level_nodes = {}

            logging.info(f"Constructing Layer {layer}")

            node_list_current_layer = get_node_list(current_level_nodes)

            # If we cannot reduce further, early terminate the tree construction
            if len(node_list_current_layer) <= self.reduction_dimension + 1:
                self.num_layers = layer
                logging.info(
                    f"Stopping Layer construction: Cannot Create More Layers. Total Layers in tree: {layer}"
                )
                break

            # Clustering step: Groups semantically similar nodes together
            # In the paper it adopts UMAP and GMM methods
            clusters = self.clustering_algorithm.perform_clustering(
                node_list_current_layer,
                self.cluster_embedding_model,
                reduction_dimension=self.reduction_dimension,
                **self.clustering_params,
            )

            lock = Lock()

            summarization_length = self.summarization_length
            logging.info(f"Summarization Length: {summarization_length}")

            if use_multithreading:
                with ThreadPoolExecutor() as executor:
                    for cluster in clusters:
                        executor.submit(
                            process_cluster,
                            cluster,
                            new_level_nodes,
                            next_node_index,
                            summarization_length,
                            lock,
                        )
                        next_node_index += 1
                    executor.shutdown(wait=True)

            else:
                # perform summarization and parent node creation for every cluster
                for cluster in clusters:
                    process_cluster(
                        cluster,
                        new_level_nodes,
                        next_node_index,
                        summarization_length,
                        lock,
                    )
                    next_node_index += 1

            # update layer_to_nodes to include new level nodes
            layer_to_nodes[layer + 1] = list(new_level_nodes.values())
            current_level_nodes = new_level_nodes
            # update all_tree_nodes to include new level nodes
            all_tree_nodes.update(new_level_nodes)

            # tree = Tree(
            #     all_tree_nodes,
            #     layer_to_nodes[layer + 1],
            #     layer_to_nodes[0],
            #     layer + 1,
            #     layer_to_nodes,
            # )

        return current_level_nodes
