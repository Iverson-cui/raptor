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
from raptor.EmbeddingModels import SBertEmbeddingModel, BGEM3Model
from raptor.SummarizationModels import DeepSeekSummarizationModel

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Setup basic logging for debugging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


def evaluate_on_squad(local_test=True):
    """
    Evaluates RAPTOR on the SQuAD dataset.

    Args:
        local_test (bool): If True, runs on a small subset (10 contexts, 20 questions).
                           If False, runs on the entire validation dataset.
    """
    logging.info(
        f"Starting SQuAD evaluation. Mode: {'Local Test' if local_test else 'Full Dataset'}"
    )

    logging.info("Loading SQuAD dataset validation split...")
    dataset = load_dataset("squad", split="validation")

    # Define slicing parameters
    if local_test:
        num_contexts_target = 25
        num_eval_questions_target = 50
    else:
        num_contexts_target = float("inf")
        num_eval_questions_target = float("inf")

    # 1. Collect contexts to build the unified tree
    logging.info("Gathering contexts...")
    all_contexts = []
    seen_contexts = set()
    for item in dataset:
        if item["context"] not in seen_contexts:
            all_contexts.append(item["context"])
            seen_contexts.add(item["context"])
        # if we've reached the target number of contexts, stop
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

    # Ensure API Key is available for DeepSeek as requested
    if not os.environ.get("DEEPSEEK_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        logging.warning(
            "Warning: No API keys found for summarization model. Ensure DEEPSEEK_API_KEY is set."
        )

    # Summarization model stays DeepSeek for both modes as requested
    summarization_model = DeepSeekSummarizationModel()

    if local_test:
        logging.info("Initializing LOCAL models: UnifiedQA (QA) & SBert (Embedding)...")
        qa_model = UnifiedQAModel()
        embedding_model = SBertEmbeddingModel()
    else:
        logging.info("Initializing SERVER models: Qwen (QA) & BGEM3 (Embedding)...")
        # Initialize QwenQAModel - assuming default path or SERVER_MODEL_PATH env var is set on server
        qa_model = QwenQAModel()
        # Initialize BGEM3Model for embeddings
        embedding_model = BGEM3Model()

    RAC = RetrievalAugmentationConfig(
        summarization_model=summarization_model,
        qa_model=qa_model,
        embedding_model=embedding_model,
    )

    RA = RetrievalAugmentation(config=RAC)

    # Concatenate all contexts into one large corpus
    full_corpus = "\n\n".join(all_contexts)

    logging.info("Building RAPTOR tree (recursive clustering & summarization)...")
    start_time = time.time()
    # Build tree from the corpus
    RA.add_documents(full_corpus)
    elapsed = time.time() - start_time
    logging.info(f"Tree built successfully in {elapsed:.2f} seconds.")

    logging.info("Loading SQuAD evaluation metric via 'evaluate' library...")
    squad_metric = evaluate.load("squad")

    predictions = []
    references = []

    logging.info(f"Starting Q&A evaluation loop for {len(eval_items)} questions...")

    for i, item in enumerate(eval_items):
        question = item["question"]

        # answer_question returns (answer, layer_info)
        response = RA.answer_question(question=question)
        pred_answer = response[0] if isinstance(response, tuple) else response

        predictions.append({"id": item["id"], "prediction_text": str(pred_answer)})
        references.append({"id": item["id"], "answers": item["answers"]})

        # Periodic checkpoint logs
        log_freq = 5 if local_test else 100
        if (i + 1) % log_freq == 0:
            logging.info(f"Checkpoint: Processed {i + 1}/{len(eval_items)} questions.")

        if i < 1:  # Log the first prediction for visual sanity check
            logging.info(
                f"Sample - Q: {question} | Pred: {pred_answer} | Gold: {item['answers']['text']}"
            )

    logging.info("Computing final F1 and Exact Match scores...")
    results = squad_metric.compute(predictions=predictions, references=references)

    print("\n" + "=" * 50)
    print(f"Final Evaluation Results (local_test={local_test})")
    print(f"Total Questions: {len(eval_items)}")
    print(f"Average F1: {results['f1']:.2f}")
    print(f"Average EM: {results['exact_match']:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    # Placeholder to ensure environment won't crash on default initializers
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "not_used"

    # Defaulting to local_test=True for verification run
    evaluate_on_squad(local_test=True)
