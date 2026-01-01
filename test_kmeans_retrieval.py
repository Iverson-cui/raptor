import logging
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from raptor.RetrievalAugmentation import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor.EmbeddingModels import MpnetBaseCosModel
from raptor.QAModels import BaseQAModel

# Setup logging
logging.basicConfig(level=logging.INFO)

class MockQAModel(BaseQAModel):
    def answer_question(self, context, question):
        return f"Question: {question}\nContext: {context[:200]}..." # Return preview

def test_kmeans_retrieval():
    # Load sample text
    with open('demo/sample.txt', 'r') as f:
        text = f.read()

    # Use MpnetBaseCosModel
    embedding_model = MpnetBaseCosModel()
    
    # Configure RetrievalAugmentation
    # Using KMeans for both builder and retriever
    config = RetrievalAugmentationConfig(
        tree_builder_type="kmeans",
        tree_retriever_type="kmeans",
        tb_n_clusters=3,        # 3 clusters for builder
        tr_top_k_clusters=2,    # Search in top 2 closest clusters
        tr_top_k=5,             # Return top 5 leaf nodes
        embedding_model=embedding_model,
        qa_model=MockQAModel()
    )

    ra = RetrievalAugmentation(config)

    # Build Tree
    logging.info("Adding documents (building tree)...")
    ra.add_documents(text)
    
    # Verify Tree Structure
    tree = ra.tree
    print(f"Tree Layers: {tree.num_layers}")
    assert tree.num_layers == 1
    assert 1 in tree.layer_to_nodes
    
    # Test Retrieval
    question = "What did the father bring for Cinderella?"
    logging.info(f"Asking question: {question}")
    
    # We expect retrieval to work
    answer = ra.answer_question(question)
    print(f"Answer from MockQA:\n{answer}")
    
    # Detailed Retrieval Check
    # We can also call retrieve directly to inspect context
    context = ra.retrieve(question, return_layer_information=False)
    
    # The context should likely contain "hazel", "twig", "branch" or "hat"
    keywords = ["hazel", "twig", "branch", "hat"]
    found = any(k in context.lower() for k in keywords)
    
    if found:
        logging.info("Relevant keywords found in context.")
    else:
        logging.warning("Relevant keywords NOT found in context. Context might be poor.")
        print("Context preview:", context[:500])
        
    assert len(context) > 0, "Retrieved context is empty"
    
    print("Test Passed Successfully!")

if __name__ == "__main__":
    test_kmeans_retrieval()
