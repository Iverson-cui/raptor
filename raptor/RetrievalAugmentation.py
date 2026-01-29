import logging
import pickle

from .cluster_tree_builder import ClusterTreeBuilder, ClusterTreeConfig
from .kmeans_tree_builder import KMeansTreeBuilder, KMeansTreeConfig
from .merge_tree_builder import MergeTreeBuilder, MergeTreeConfig
from .kmeans_retriever import KMeansRetriever, KMeansRetrieverConfig
from .EmbeddingModels import BaseEmbeddingModel
from .QAModels import BaseQAModel, GPT3TurboQAModel, QwenQAModel, UnifiedQAModel
from .SummarizationModels import BaseSummarizationModel
from .tree_builder import TreeBuilder, TreeBuilderConfig
from .tree_retriever import TreeRetriever, TreeRetrieverConfig
from .tree_structures import Node, Tree

# Define a dictionary to map supported tree builders to their respective configs
supported_tree_builders = {
    "cluster": (ClusterTreeBuilder, ClusterTreeConfig),
    "kmeans": (KMeansTreeBuilder, KMeansTreeConfig),
    "merge": (MergeTreeBuilder, MergeTreeConfig),
}

# Define a dictionary to map supported retrievers to their respective configs
supported_retrievers = {
    "tree": (TreeRetriever, TreeRetrieverConfig),
    "kmeans": (KMeansRetriever, KMeansRetrieverConfig),
}
# TODO: add a merge tree retriever
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class RetrievalAugmentationConfig:

    def __init__(
        self,
        tree_builder_config=None,
        tree_retriever_config=None,  # Change from default instantiation
        # question answering model
        qa_model=None,
        # global embedding model
        embedding_model=None,
        # global summarization model
        summarization_model=None,
        tree_builder_type="cluster",
        tree_retriever_type="tree",  # Default to tree retriever
        # New parameters for TreeRetrieverConfig and TreeBuilderConfig
        # TreeRetrieverConfig arguments
        tr_tokenizer=None,
        tr_threshold=0.5,
        tr_top_k=5,
        tr_selection_mode="top_k",
        # query embedding model
        tr_context_embedding_model="BGEM3",
        # tree retriever local embedding model
        tr_embedding_model=None,
        tr_num_layers=None,
        tr_start_layer=None,
        tr_top_k_clusters=3,  # specifically for KMeansRetriever
        # TreeBuilderConfig arguments
        tb_tokenizer=None,
        tb_max_tokens=100,
        tb_num_layers=5,
        tb_threshold=0.5,
        tb_top_k=5,
        tb_selection_mode="top_k",
        tb_summarization_length=100,
        # tree builder local summarization model
        tb_summarization_model=None,
        # tree builder local embedding models
        tb_embedding_models=None,
        # which embedding to use for clustering in tree builder
        tb_cluster_embedding_model="BGEM3",
        tb_n_clusters=5,  # specifically for KMeansTreeBuilder
        tb_merge_top_k_clusters=10,  # specifically for MergeTreeBuilder
        tb_merge_top_k_chunks=3,  # specifically for MergeTreeBuilder
    ):
        # Validate tree_builder_type
        if tree_builder_type not in supported_tree_builders:
            raise ValueError(
                f"tree_builder_type must be one of {list(supported_tree_builders.keys())}"
            )

        if tree_retriever_type not in supported_retrievers:
            raise ValueError(
                f"tree_retriever_type must be one of {list(supported_retrievers.keys())}"
            )

        print("Start initializing RetrievalAugmentation...")
        # Validate qa_model
        print("Validating QA model...")
        # if qa_model is None, skip this step and default to UnifiedQAModel later
        if qa_model is not None and not isinstance(qa_model, BaseQAModel):
            raise ValueError("qa_model must be an instance of BaseQAModel")
        elif qa_model is not None:
            print(f"QA model set to {qa_model}")
        else:
            print("QA model not provided in config")

        print("Validating embedding model...")
        if embedding_model is not None and not isinstance(
            embedding_model, BaseEmbeddingModel
        ):
            raise ValueError(
                "embedding_model must be an instance of BaseEmbeddingModel"
            )
        # embedding_model is the universal way to set embedding model for both tree builder and retriever
        # if embedding_model is None, 5 model arguments are None and default values are set later
        elif embedding_model is not None:
            if tb_embedding_models is not None:
                raise ValueError(
                    "Only one of 'tb_embedding_models' or 'embedding_model' should be provided, not both."
                )
            # if embedding_model is set, 4 models below is set to align with it
            tb_embedding_models = {"EMB": embedding_model}
            tr_embedding_model = embedding_model
            tb_cluster_embedding_model = "EMB"
            tr_context_embedding_model = "EMB"
        else:
            print("Embedding model not provided in config")
        print("Validating summarization model...")
        if summarization_model is not None and not isinstance(
            summarization_model, BaseSummarizationModel
        ):
            raise ValueError(
                "summarization_model must be an instance of BaseSummarizationModel"
            )

        # The same logic applies to summarization_model, except that summarization_model only has one child
        elif summarization_model is not None:
            if tb_summarization_model is not None:
                raise ValueError(
                    "Only one of 'tb_summarization_model' or 'summarization_model' should be provided, not both."
                )
            tb_summarization_model = summarization_model
        else:
            print("Summarization model not provided in config")

        # Set TreeBuilderConfig
        # supported_tree_builders is a dict
        tree_builder_class, tree_builder_config_class = supported_tree_builders[
            tree_builder_type
        ]
        # based on the ways we choose to build the tree, we choose the corresponding TreeBuilderConfig class
        # the same arguments are passed to different TreeBuilderConfig classes
        print("Setting TreeBuilderConfig...")
        if tree_builder_config is None:
            # Prepare kwargs for config init
            config_kwargs = {
                "tokenizer": tb_tokenizer,
                "max_tokens": tb_max_tokens,
                "num_layers": tb_num_layers,
                "threshold": tb_threshold,
                "top_k": tb_top_k,
                "selection_mode": tb_selection_mode,
                "summarization_length": tb_summarization_length,
                "summarization_model": tb_summarization_model,
                "embedding_models": tb_embedding_models,
                "cluster_embedding_model": tb_cluster_embedding_model,
            }
            if tree_builder_type == "kmeans":
                config_kwargs["n_clusters"] = tb_n_clusters

            if tree_builder_type == "merge":
                config_kwargs["n_clusters"] = tb_n_clusters
                config_kwargs["merge_top_k_clusters"] = tb_merge_top_k_clusters
                config_kwargs["merge_top_k_chunks"] = tb_merge_top_k_chunks

            tree_builder_config = tree_builder_config_class(**config_kwargs)

        elif not isinstance(tree_builder_config, tree_builder_config_class):
            raise ValueError(
                f"tree_builder_config must be a direct instance of {tree_builder_config_class} for tree_builder_type '{tree_builder_type}'"
            )

        # Set TreeRetrieverConfig
        print("Setting TreeRetrieverConfig...")
        retriever_class, retriever_config_class = supported_retrievers[
            tree_retriever_type
        ]

        if tree_retriever_config is None:
            config_kwargs = {
                "tokenizer": tr_tokenizer,
                "threshold": tr_threshold,
                "top_k": tr_top_k,
                "selection_mode": tr_selection_mode,
                "context_embedding_model": tr_context_embedding_model,
                "embedding_model": tr_embedding_model,
                "num_layers": tr_num_layers,
                "start_layer": tr_start_layer,
            }
            if tree_retriever_type == "kmeans":
                config_kwargs["top_k_clusters"] = tr_top_k_clusters

            tree_retriever_config = retriever_config_class(**config_kwargs)

        elif not isinstance(tree_retriever_config, retriever_config_class):
            raise ValueError(
                f"tree_retriever_config must be an instance of {retriever_config_class}"
            )

        # Assign the created configurations to the instance
        print("2 config done")
        self.tree_builder_config = tree_builder_config
        self.tree_retriever_config = tree_retriever_config
        self.qa_model = qa_model or QwenQAModel()
        self.tree_builder_type = tree_builder_type
        self.tree_retriever_type = tree_retriever_type

    def log_config(self):
        config_summary = """
        RetrievalAugmentationConfig:
            {tree_builder_config}
            
            {tree_retriever_config}
            
            QA Model: {qa_model}
            Tree Builder Type: {tree_builder_type}
            Tree Retriever Type: {tree_retriever_type}
        """.format(
            tree_builder_config=self.tree_builder_config.log_config(),
            tree_retriever_config=self.tree_retriever_config.log_config(),
            qa_model=self.qa_model,
            tree_builder_type=self.tree_builder_type,
            tree_retriever_type=self.tree_retriever_type,
        )
        return config_summary


class RetrievalAugmentation:
    """
    A Retrieval Augmentation class that combines the TreeBuilder and TreeRetriever classes.
    Enables adding documents to the tree, retrieving information, and answering questions.
    """

    def __init__(self, config=None, tree=None):
        """
        Initializes a RetrievalAugmentation instance with the specified configuration.
        Args:
            config (RetrievalAugmentationConfig): The configuration for the RetrievalAugmentation instance.
            tree: The tree instance or the path to a pickled tree file.
        """

        if config is None:
            config = RetrievalAugmentationConfig()
        if not isinstance(config, RetrievalAugmentationConfig):
            raise ValueError(
                "config must be an instance of RetrievalAugmentationConfig"
            )

        self.config = config
        # Check if tree is a string (indicating a path to a pickled tree)
        if isinstance(tree, str):
            try:
                with open(tree, "rb") as file:
                    self.tree = pickle.load(file)
                if not isinstance(self.tree, Tree):
                    raise ValueError("The loaded object is not an instance of Tree")
            except Exception as e:
                raise ValueError(f"Failed to load tree from {tree}: {e}")
        elif isinstance(tree, Tree) or tree is None:
            self.tree = tree
        else:
            raise ValueError(
                "tree must be an instance of Tree, a path to a pickled Tree, or None"
            )

        tree_builder_class = supported_tree_builders[config.tree_builder_type][0]
        self.tree_builder = tree_builder_class(config.tree_builder_config)
        self.qa_model = config.qa_model
        # self.retrievers is a list of retriever instances
        self.retrievers = {}

        if self.tree is not None:
            retriever_class = supported_retrievers[config.tree_retriever_type][0]
            self.retriever = retriever_class(config.tree_retriever_config, self.tree)
            self.retrievers["default"] = self.retriever
        else:
            self.retriever = None

        logging.info(
            f"Successfully initialized RetrievalAugmentation with Config {config.log_config()}"
        )

    def add_retriever(self, name, config, retriever_type=None):
        if self.tree is None:
            raise ValueError(
                "The tree has not been initialized. Please call 'add_documents' first."
            )

        if retriever_type is None:
            retriever_type = self.config.tree_retriever_type

        if retriever_type not in supported_retrievers:
            raise ValueError(f"Unsupported retriever type: {retriever_type}")

        retriever_class, config_class = supported_retrievers[retriever_type]

        if not isinstance(config, config_class):
            raise ValueError(
                f"config must be an instance of {config_class} for retriever_type '{retriever_type}'"
            )

        self.retrievers[name] = retriever_class(config, self.tree)
        logging.info(f"Successfully added retriever '{name}'.")

    def add_documents(self, docs, use_multithreading=True):
        """
        Adds documents to the tree and creates a TreeRetriever instance.

        Step 1 (Tree Building Entry): This is the high-level entry point.
        It delegates the heavy lifting of building the hierarchical tree
        from the raw text to the configured TreeBuilder (e.g., ClusterTreeBuilder).
        """
        if self.tree is not None:
            user_input = input(
                "Warning: Overwriting existing tree. Did you mean to call 'add_to_existing' instead? (y/n): "
            )
            if user_input.lower() == "y":
                # self.add_to_existing(docs)
                return

        # The build_from_text method orchestrates the entire tree construction process:
        # 1. Splitting text into chunks (Leaf Nodes)
        # 2. Recursively clustering and summarizing (Higher-level Nodes)
        self.tree = self.tree_builder.build_from_text(text=docs, use_multithreading=use_multithreading)
        # builder and retriever is connected by this self.tree
        retriever_class = supported_retrievers[self.config.tree_retriever_type][0]
        self.retriever = retriever_class(self.config.tree_retriever_config, self.tree)
        self.retrievers["default"] = self.retriever

    def retrieve(
        self,
        question,
        retriever_name="default",
        start_layer: int = None,
        num_layers: int = None,
        top_k: int = 10,
        max_tokens: int = 5000,
        collapse_tree: bool = True,
        return_layer_information: bool = True,
    ):
        """
        Retrieves information and answers a question using the TreeRetriever instance.

        Args:
            question (str): The question to answer.
            retriever_name (str): The name of the retriever to use. Defaults to "default".
            start_layer (int): The layer to start from. Defaults to self.start_layer.
            num_layers (int): The number of layers to traverse. Defaults to self.num_layers.
            max_tokens (int): The maximum number of tokens. Defaults to 3500.
            use_all_information (bool): Whether to retrieve information from all nodes. Defaults to False.

        Returns:
            str: The context from which the answer can be found.

        Raises:
            ValueError: If the TreeRetriever instance has not been initialized.
        """
        if retriever_name not in self.retrievers:
            raise ValueError(f"Retriever '{retriever_name}' not found.")

        # pick the retriever by name
        retriever = self.retrievers[retriever_name]

        # use that picked retriever to retrieve contexts and return
        return retriever.retrieve(
            question,
            start_layer,
            num_layers,
            top_k,
            max_tokens,
            collapse_tree,
            return_layer_information,
        )

    def answer_question(
        self,
        question,
        retriever_name="default",
        top_k: int = 10,
        start_layer: int = None,
        num_layers: int = None,
        max_tokens: int = 5000,
        collapse_tree: bool = False,
        return_layer_information: bool = False,
    ):
        """
        Retrieves information and answers a question using the TreeRetriever instance.

        Args:
            question (str): The question to answer.
            retriever_name (str): The name of the retriever to use. Defaults to "default".
            start_layer (int): The layer to start from. Defaults to self.start_layer.
            num_layers (int): The number of layers to traverse. Defaults to self.num_layers.
            max_tokens (int): The maximum number of tokens. Defaults to 3500. This means that the context provided to answer a question will not exceed max_tokens.
            use_all_information (bool): Whether to retrieve information from all nodes. Defaults to False.

        Returns:
            str: The answer to the question.

        Raises:
            ValueError: If the TreeRetriever instance has not been initialized.
        """
        # if return_layer_information:
        # use the retriever with retriever_name
        context, layer_information = self.retrieve(
            question,
            retriever_name,
            start_layer,
            num_layers,
            top_k,
            max_tokens,
            collapse_tree,
            True,
        )
        # print("Retrieved Context: ", context)
        answer = self.qa_model.answer_question(context, question)

        if return_layer_information:
            return answer, layer_information

        return answer

    def save(self, path):
        if self.tree is None:
            raise ValueError("There is no tree to save.")
        with open(path, "wb") as file:
            pickle.dump(self.tree, file)
        logging.info(f"Tree successfully saved to {path}")
