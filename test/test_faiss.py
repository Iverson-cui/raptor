import os
import sys
import logging
import warnings

# Ensure the raptor package is accessible from the test directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from raptor import FaissRetriever, FaissRetrieverConfig
from raptor.EmbeddingModels import SBertEmbeddingModel, BGEM3Model

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Setup basic logging for debugging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

def test_faiss_retriever(local_test=True):
    logging.info(f"Starting FaissRetriever test. Mode: {'Local Test' if local_test else 'Server Test'}")

    # 1. Load Data
    sample_path = os.path.join(os.path.dirname(__file__), "../demo/sample.txt")
    logging.info(f"Loading sample text from {sample_path}...")
    try:
        with open(sample_path, "r") as f:
            text = f.read()
    except FileNotFoundError:
        logging.error(f"Sample file not found at {sample_path}")
        return

    # 2. Initialize Models
    if local_test:
        logging.info("Initializing LOCAL model: SBERT Embedding Model...")
        embedding_model = SBertEmbeddingModel()
    else:
        logging.info("Initializing SERVER model: BGEM3 Embedding Model...")
        embedding_model = BGEM3Model()

    # 3. Configure Retriever
    logging.info("Configuring FaissRetriever...")
    retriever_config = FaissRetrieverConfig(
        max_tokens=100,
        max_context_tokens=1000,
        use_top_k=True,
        top_k=3,
        embedding_model=embedding_model,
    )

    # 4. Initialize Retriever
    retriever = FaissRetriever(retriever_config)

    # 5. Build Index
    logging.info("Building Faiss index from text...")
    retriever.build_from_text(text)

    # 6. Sanity Check
    logging.info("Running sanity check...")
    try:
        retriever.sanity_check(num_samples=2)
        logging.info("Sanity check passed!")
    except AssertionError as e:
        logging.error(f"Sanity check failed: {e}")
    except Exception as e:
        logging.error(f"An error occurred during sanity check: {e}")

    # 7. Test Retrieval
    query = "What happened to the golden shoe?"
    logging.info(f"Testing retrieval with query: '{query}'")
    
    retrieved_context = retriever.retrieve(query)
    
    logging.info("-" * 30)
    logging.info("Retrieved Context:")
    logging.info("-" * 30)
    print(retrieved_context)
    logging.info("-" * 30)

    if retrieved_context:
        logging.info("FaissRetriever test completed successfully.")
    else:
        logging.warning("No context retrieved!")

if __name__ == "__main__":
    # Default to local_test=True for verification run
    test_faiss_retriever(local_test=True)