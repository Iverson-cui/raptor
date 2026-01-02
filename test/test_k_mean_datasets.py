import os
import sys
import logging
import torch
import warnings
import time
import evaluate
from datasets import load_dataset

# Ensure the raptor package is accessible from the test directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor.QAModels import UnifiedQAModel, QwenQAModel
from raptor.EmbeddingModels import SBertEmbeddingModel, BGEM3Model, MpnetBaseCosModel
from raptor.SummarizationModels import (
    DeepSeekSummarizationModel,
    QwenLocalSummarizationModel,
    BaseSummarizationModel,
)

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Setup basic logging for debugging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


class MockSummarizationModel(BaseSummarizationModel):
    def summarize(self, context, max_tokens=150):
        return context[:max_tokens]


def evaluate_k_means_on_squad(local_test=True):
    """
    Evaluates K-Means RAPTOR on the SQuAD dataset.

    Args:
        local_test (bool): If True, runs on a small subset (10 contexts, 20 questions).
                           If False, runs on the entire validation dataset.
    """
    print("local_test =", local_test)
    logging.info(
        f"Starting SQuAD evaluation (K-Means). Mode: {'Local Test' if local_test else 'Full Dataset'}"
    )

    logging.info("Loading SQuAD dataset validation split...")
    dataset = load_dataset("squad", split="validation")

    if local_test == 0:
        # QA Model gets GPU 0 and 1
        qa_memory_map = {
            0: "0GiB",
            1: "0GiB",
            2: "0GiB",
            3: "0GiB",
            4: "40GiB",
            5: "40GiB",
            6: "40GiB",
        }
        embedding_device = "cuda:3"

    # Define slicing parameters
    if local_test:
        num_contexts_target = 25
        num_eval_questions_target = 50
    else:
        num_contexts_target = 250
        num_eval_questions_target = 500

    # 1. Collect contexts to build the unified tree
    logging.info("Gathering contexts...")
    all_contexts = []
    seen_contexts = set()
    for item in dataset:
        if item["context"] not in seen_contexts:
            all_contexts.append(item["context"])
            seen_contexts.add(item["context"])
        if len(all_contexts) >= num_contexts_target:
            break

    # 2. Collect questions to evaluate that match the gathered contexts
    logging.info("Gathering evaluation questions...")
    eval_items = []
    for item in dataset:
        if item["context"] in seen_contexts:
            eval_items.append(item)
        if len(eval_items) >= num_eval_questions_target:
            break

    logging.info(f"Tree construction corpus: {len(all_contexts)} unique contexts.")
    logging.info(f"Evaluation target: {len(eval_items)} questions.")

    # Initialize Models
    # K-Means doesn't strictly use SummarizationModel for tree building (centroids),
    # but RetrievalAugmentation might check for it. We'll pass a Mock one just in case
    # or rely on default if not using 'cluster' builder.
    # Actually, K-Means builder doesn't use summarization.

    if local_test:
        logging.info("Initializing LOCAL models: UnifiedQA (QA) & Mpnet (Embedding)...")
        qa_model = UnifiedQAModel()  # Defaults to flan-t5-small
        # Use Mpnet as it's reliable locally
        embedding_model = SBertEmbeddingModel()
    else:
        logging.info("Initializing SERVER models: Qwen (QA) & BGEM3 (Embedding)...")
        qa_model = QwenQAModel(max_memory=qa_memory_map, device_map="auto")
        embedding_model = BGEM3Model(device="cuda:3")

    # Configure for K-Means
    # Adjust clusters based on dataset size
    # For 25 contexts, maybe 5 clusters is enough.
    # For 250 contexts, maybe 20.
    n_clusters = 5 if local_test else 20

    RAC = RetrievalAugmentationConfig(
        tree_builder_type="kmeans",
        tree_retriever_type="kmeans",
        tb_n_clusters=n_clusters,
        tr_top_k_clusters=2,  # Search top 2 clusters
        tr_top_k=5,  # Return top 5 chunks
        qa_model=qa_model,
        embedding_model=embedding_model,
        # summarization_model not needed for KMeans builder logic, but to satisfy config init checks if any
        summarization_model=MockSummarizationModel(),
    )

    RA = RetrievalAugmentation(config=RAC)

    # Concatenate all contexts into one large corpus
    full_corpus = "\n\n".join(all_contexts)

    logging.info("Building K-Means RAPTOR tree...")
    start_time = time.time()
    # Build tree from the corpus
    RA.add_documents(full_corpus)
    elapsed = time.time() - start_time
    logging.info(f"Tree built successfully in {elapsed:.2f} seconds.")

    # Verify Tree
    if hasattr(RA, "tree"):
        logging.info(f"Tree Layers: {RA.tree.num_layers}")
        if 1 in RA.tree.layer_to_nodes:
            logging.info(
                f"Number of clusters (Layer 1): {len(RA.tree.layer_to_nodes[1])}"
            )

    logging.info("Loading SQuAD evaluation metric via 'evaluate' library...")
    squad_metric = evaluate.load("squad")

    predictions = []
    references = []

    logging.info(f"Starting Q&A evaluation loop for {len(eval_items)} questions...")

    for i, item in enumerate(eval_items):
        question = item["question"]

        # answer_question returns (answer, layer_info) if return_layer_information is True (default for answer_question?)
        # Let's check RetrievalAugmentation.answer_question default: return_layer_information=True
        response = RA.answer_question(question=question)

        # Unpack if tuple
        if isinstance(response, tuple):
            pred_answer = response[0]
        else:
            pred_answer = response

        predictions.append({"id": item["id"], "prediction_text": str(pred_answer)})
        references.append({"id": item["id"], "answers": item["answers"]})

        # Periodic checkpoint logs
        log_freq = 5 if local_test else 100
        if (i + 1) % log_freq == 0:
            logging.info(f"Checkpoint: Processed {i + 1}/{len(eval_items)} questions.")

        if i < 2:  # Log first 2 predictions
            logging.info(
                f"Sample {i+1} - Q: {question} | Pred: {pred_answer} | Gold: {item['answers']['text']}"
            )

    logging.info("Computing final F1 and Exact Match scores...")
    results = squad_metric.compute(predictions=predictions, references=references)

    print("\n" + "=" * 50)
    print(f"Final Evaluation Results (K-Means RAPTOR, local_test={local_test})")
    print(f"Total Questions: {len(eval_items)}")
    print(f"Average F1: {results['f1']:.2f}")
    print(f"Average EM: {results['exact_match']:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    # Placeholder to ensure environment won't crash on default initializers
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "not_used"

    # Defaulting to local_test=True for verification run in this environment
    evaluate_k_means_on_squad(local_test=True)
