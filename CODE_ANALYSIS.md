# Codebase Analysis: RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)

This repository implements RAPTOR, a system for building a hierarchical tree structure from text documents to enable efficient and context-aware information retrieval. The system recursively clusters and summarizes text chunks to create a multi-layered tree, which is then used for answering queries.

## 📂 Project Structure Overview

The core logic resides in the `raptor/` directory. The system is designed with a modular architecture separating tree construction, retrieval, and model interactions (Embeddings, QA, Summarization).
****
### 1. 🧠 Core Controller
*   **`raptor/RetrievalAugmentation.py`**
    *   **Role**: The main entry point and orchestrator.
    *   **Functionality**:
        *   Manages the configuration (`RetrievalAugmentationConfig`).
        *   Initializes the `TreeBuilder` and `TreeRetriever`.
        *   Provides high-level methods: `add_documents()` (to build the tree) and `answer_question()` (to retrieve context and generate answers).
        *   Handles saving/loading the tree state.

### 2. 🌲 Tree Construction (The "RAPTOR" Process)
*   **`raptor/tree_structures.py`**
    *   **Role**: Defines the data structures.
    *   **Classes**:
        *   `Node`: Represents a node in the tree (text, vector embeddings, children indices).
        *   `Tree`: Represents the entire hierarchy, tracking all nodes, root nodes, leaf nodes, and layer mappings.

*   **`raptor/tree_builder.py`**
    *   **Role**: Base class for tree construction.
    *   **Functionality**:
        *   Splits input text into chunks (leaf nodes).
        *   Manages tokenizer and model configurations.
        *   Defines the interface `construct_tree`.
        *   Provides helper methods like `create_node` and `summarize`.

*   **`raptor/cluster_tree_builder.py`**
    *   **Role**: The concrete implementation of RAPTOR's tree building logic.
    *   **Functionality**:
        *   Inherits from `TreeBuilder`.
        *   Recursively builds the tree from the bottom up.
        *   Uses **Clustering** to group semantically similar nodes.
        *   Uses **Summarization** to create parent nodes that abstract the content of their children.
        *   Stops when the number of nodes reduces to a defined dimension.

*   **`raptor/cluster_utils.py`**
    *   **Role**: Clustering logic utilities.
    *   **Functionality**:
        *   Implements `RAPTOR_Clustering`.
        *   Uses **UMAP** for dimensionality reduction.
        *   Uses **Gaussian Mixture Models (GMM)** for soft clustering.
        *   Includes logic to recursively re-cluster if a cluster is too large (`max_length_in_cluster`).

### 3. 🔍 Retrieval Mechanisms
*   **`raptor/Retrievers.py`**
    *   **Role**: Abstract base class for retrievers.

*   **`raptor/tree_retriever.py`**
    *   **Role**: The primary retriever for the RAPTOR tree.
    *   **Functionality**:
        *   `retrieve()`: Supports two modes:
            1.  **Collapsed Tree**: Flattens the tree and searches all nodes (roots, intermediates, leaves) simultaneously using embedding similarity.
            2.  **Tree Traversal**: (Implemented but `collapse_tree=True` seems to be the default/preferred) Navigates the tree layer by layer.
        *   Selects the top-k most relevant nodes/context for a given query.

*   **`raptor/FaissRetriever.py`**
    *   **Role**: A flat retrieval alternative.
    *   **Functionality**:
        *   Uses **FAISS** (Facebook AI Similarity Search) for efficient vector search.
        *   Can build an index from raw text chunks or existing leaf nodes.
        *   Used effectively as a baseline or for simple flat retrieval tasks.

### 4. 🤖 Model Wrappers
These files abstract interactions with external APIs (like OpenAI) or local models (HuggingFace).

*   **`raptor/EmbeddingModels.py`**
    *   `OpenAIEmbeddingModel`: Uses OpenAI's `text-embedding-ada-002` (or similar).
    *   `SBertEmbeddingModel`: Uses SentenceTransformers (e.g., `multi-qa-mpnet-base-cos-v1`).

*   **`raptor/SummarizationModels.py`**
    *   `GPT3TurboSummarizationModel`: Uses `gpt-3.5-turbo` to summarize text clusters.
    *   `GPT3SummarizationModel`: Uses legacy `text-davinci-003`.

*   **`raptor/QAModels.py`**
    *   `GPT3TurboQAModel` / `GPT4QAModel`: Uses OpenAI Chat models to answer questions based on retrieved context.
    *   `UnifiedQAModel`: Uses the T5-based UnifiedQA model (local execution).

### 5. 🛠️ Utilities
*   **`raptor/utils.py`**
    *   **Role**: General helper functions.
    *   **Functionality**: Text splitting, embedding distance calculations (cosine), handling node/text conversions.

### 6. 📄 Demo
*   **`demo.ipynb`**
    *   **Role**: Usage example.
    *   **Functionality**: Demonstrates how to initialize `RetrievalAugmentation`, add documents (e.g., the Cinderella story), and ask questions.

---
**Summary of Flow:**
1.  **Input**: Raw text (e.g., `sample.txt`).
2.  **Build (`add_documents`)**:
    *   Text -> Chunks -> Leaf Nodes (Embeddings).
    *   Leaf Nodes -> Cluster (GMM/UMAP) -> Summarize -> Layer 1 Nodes.
    *   Layer 1 -> Cluster -> Summarize -> Layer 2... (until root).
3.  **Query (`answer_question`)**:
    *   Question -> Embedding.
    *   Search Tree (e.g., all nodes) -> Top-k relevant nodes.
    *   Top-k Context + Question -> QA Model -> **Answer**.
