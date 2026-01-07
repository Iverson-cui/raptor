import os
import sys
import logging
import torch
import warnings
import time
import evaluate
import argparse
from datasets import load_dataset, concatenate_datasets

# Ensure the raptor package is accessible from the test directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor.QAModels import UnifiedQAModel, QwenQAModel, DeepSeekQAModel
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


def get_dataset_processors(dataset_name):
    """
    Returns extraction and processing functions for the specified dataset.
    """
    if dataset_name == "squad":

        def extract_contexts(item):
            return [item["context"]]

        def process_item(item):
            return {
                "id": str(item["id"]),
                "question": item["question"],
                "answers": item["answers"],
            }

    elif dataset_name == "hotpot_qa":

        def extract_contexts(item):
            contexts = []
            # item['context'] is {'title': [...], 'sentences': [[...], [...]]}
            for sentences_list in item["context"]["sentences"]:
                para = " ".join(sentences_list).strip()
                if para:
                    contexts.append(para)
            return contexts

        def process_item(item):
            return {
                "id": str(item["id"]),
                "question": item["question"],
                "answers": {"text": [item["answer"]], "answer_start": [-1]},
            }

    elif dataset_name == "ms_marco":

        def extract_contexts(item):
            # item['passages'] is {'passage_text': [...], ...}
            return [p for p in item["passages"]["passage_text"] if p.strip()]

        def process_item(item):
            return {
                "id": str(item["query_id"]),
                "question": item["query"],
                "answers": {"text": item["answers"], "answer_start": [-1]},
            }

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return extract_contexts, process_item


def evaluate_k_means_on_dataset(
    dataset_name="squad",
    model_name="qwen",
    local_test=True,
    tb_max_tokens=None,
    print_summary=True,
    answer_without_context=False,
):
    """
    Evaluates K-Means RAPTOR on the specified dataset.
    """
    print(f"Dataset: {dataset_name}, Model: {model_name}, local_test: {local_test}")
    logging.info(
        f"Starting evaluation (K-Means). Mode: {'Local Test' if local_test else 'Full Dataset'}"
    )

    # Load dataset object (concatenated splits if needed)
    splits = ["validation"] if local_test else ["train", "validation"]
    loaded_splits = []
    for split in splits:
        try:
            if dataset_name == "squad":
                loaded_splits.append(load_dataset("squad", split=split))
            elif dataset_name == "hotpot_qa":
                loaded_splits.append(
                    load_dataset("hotpot_qa", "distractor", split=split)
                )
            elif dataset_name == "ms_marco":
                loaded_splits.append(load_dataset("ms_marco", "v1.1", split=split))
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
        except Exception as e:
            logging.warning(f"Could not load split '{split}' for {dataset_name}: {e}")

    if not loaded_splits:
        raise ValueError(f"Failed to load any splits for {dataset_name}")

    if len(loaded_splits) > 1:
        dataset = concatenate_datasets(loaded_splits)
    else:
        dataset = loaded_splits[0]

    # Get processors
    extract_contexts_fn, process_item_fn = get_dataset_processors(dataset_name)

    # Configuration for models
    if not local_test:
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
    else:
        embedding_device = "mps"

    # Define slicing parameters
    if local_test:
        num_eval_questions_target = 5
        # In local test, we gather contexts only for these 5 questions
    else:
        # Load WHOLE dataset for tree building as requested
        # For full run, we iterate the whole dataset
        num_eval_questions_target = 200
        max_contexts_to_process = None

    # Collect Data (Synchronized Loop)
    logging.info("Gathering data (contexts and questions)...")
    all_contexts = []
    eval_items = []
    seen_contexts = set()

    # Iterate through the dataset
    for i, item in enumerate(dataset):
        # 1. Extract and store contexts
        current_contexts = extract_contexts_fn(item)
        for ctx in current_contexts:
            if ctx not in seen_contexts:
                all_contexts.append(ctx)
                seen_contexts.add(ctx)

        # 2. Store eval item if needed
        if len(eval_items) < num_eval_questions_target:
            processed_item = process_item_fn(item)
            # Only add if there is at least one ground truth answer to avoid max() error in metrics
            if processed_item["answers"]["text"]:
                eval_items.append(processed_item)

        # 3. Stop condition
        if local_test:
            # In local test, stop immediately after getting enough questions.
            if len(eval_items) >= num_eval_questions_target:
                break
        else:
            # In full test, apply safety break if context limit is set
            if (
                max_contexts_to_process is not None
                and len(all_contexts) >= max_contexts_to_process
            ):
                logging.info(
                    f"Reached max context limit ({max_contexts_to_process}). Stopping data collection."
                )
                break

    logging.info(f"Tree construction corpus: {len(all_contexts)} unique contexts.")
    logging.info(f"Evaluation target: {len(eval_items)} questions.")

    # Initialize Models
    if local_test:
        logging.info("Initializing LOCAL models: UnifiedQA (QA) & SBert (Embedding)...")
        qa_model = UnifiedQAModel()
        embedding_model = SBertEmbeddingModel(device=embedding_device)
    else:
        logging.info(
            f"Initializing SERVER models: {model_name} (QA) & SBert (Embedding)..."
        )

        if model_name.lower() == "qwen":
            qa_model = QwenQAModel(max_memory=qa_memory_map, device_map="auto")
        elif model_name.lower() == "deepseek":
            qa_model = DeepSeekQAModel(max_memory=qa_memory_map, device_map="auto")
        else:
            logging.warning(f"Unknown server model {model_name}, defaulting to Qwen.")
            qa_model = QwenQAModel(max_memory=qa_memory_map, device_map="auto")

        # embedding_model = SBertEmbeddingModel(device=embedding_device)
        embedding_model = BGEM3Model(device=embedding_device)
        # embedding_model = SBertEmbeddingModel(
        #     device=embedding_device,
        #     target_gpus=["cuda:3", "cuda:4", "cuda:5", "cuda:6"],
        # )

    # Configure for K-Means
    # Configure parameters based on mode and dataset
    if local_test:
        default_tokens = 200
        n_clusters = 5
        tr_top_k_clusters = 2
        tr_top_k = 5
    else:
        if dataset_name == "squad":
            default_tokens = 256
            n_clusters = 210
            tr_top_k_clusters = 15
            tr_top_k = 10
        elif dataset_name in ["hotpot_qa", "ms_marco"]:
            default_tokens = 256
            n_clusters = 2500
            tr_top_k_clusters = 50
            tr_top_k = 15
        else:
            # Fallback defaults
            default_tokens = 256
            n_clusters = 50
            tr_top_k_clusters = 5
            tr_top_k = 10

    if tb_max_tokens is None:
        tb_max_tokens = default_tokens

    logging.info(
        f"Configuring RAPTOR with: n_clusters={n_clusters}, tb_max_tokens={tb_max_tokens}, tr_top_k_clusters={tr_top_k_clusters}, tr_top_k={tr_top_k}"
    )

    RAC = RetrievalAugmentationConfig(
        tree_builder_type="kmeans",
        tree_retriever_type="kmeans",
        tb_n_clusters=n_clusters,
        tr_top_k_clusters=tr_top_k_clusters,
        tr_top_k=tr_top_k,
        qa_model=qa_model,
        embedding_model=embedding_model,
        summarization_model=MockSummarizationModel(),
        tb_max_tokens=tb_max_tokens,
    )

    RA = RetrievalAugmentation(config=RAC)

    # Concatenate all contexts into one large corpus
    logging.info("Joining contexts into full corpus...")
    full_corpus = "\n\n".join(all_contexts)

    logging.info("Building K-Means RAPTOR tree...")
    start_time = time.time()
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

    logging.info("Loading evaluation metric...")
    squad_metric = evaluate.load("squad")

    predictions = []
    references = []

    logging.info(f"Starting Q&A evaluation loop for {len(eval_items)} questions...")

    for i, item in enumerate(eval_items):
        question = item["question"]

        try:
            if answer_without_context:
                if hasattr(qa_model, "answer_question_without_contexts"):
                    pred_answer = qa_model.answer_question_without_contexts(
                        context=None, question=question
                    )
                else:
                    # Fallback for models without specific no-context method
                    pred_answer = qa_model.answer_question(
                        context="", question=question
                    )
            else:
                response = RA.answer_question(question=question)

                if isinstance(response, tuple):
                    pred_answer = response[0]
                else:
                    pred_answer = response
        except Exception as e:
            logging.error(f"Error answering question {i}: {e}")
            pred_answer = ""

        predictions.append({"id": item["id"], "prediction_text": str(pred_answer)})
        references.append({"id": item["id"], "answers": item["answers"]})

        log_freq = 1 if local_test else 10
        if (i + 1) % log_freq == 0:
            logging.info(f"Checkpoint: Processed {i + 1}/{len(eval_items)} questions.")

        if i < 2:  # Log first 2 predictions
            logging.info(
                f"Sample {i+1} - Q: {question} | Pred: {pred_answer} | Gold: {item['answers']['text']}"
            )

    logging.info("Computing final F1 and Exact Match scores...")
    results = squad_metric.compute(predictions=predictions, references=references)

    if print_summary:
        print("\n" + "=" * 50)
        print(f"Final Evaluation Results")
        print(f"Dataset: {dataset_name}, Model: {model_name}, Chunk size: {tb_max_tokens}")
        # print(f"Total Questions: {len(eval_items)}")
        print(f"Average F1: {results['f1']:.2f}")
        print(f"Average EM: {results['exact_match']:.2f}")
        print("=" * 50)

    return results


if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "not_used"

    parser = argparse.ArgumentParser(
        description="Evaluate RAPTOR K-Means on various datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="squad",
        choices=["squad", "hotpot_qa", "ms_marco"],
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen",
        choices=["qwen", "deepseek", "unifiedqa"],
        help="QA Model to use (unifiedqa is for local test)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run in local test mode (small subset, CPU)",
    )
    parser.add_argument(
        "--fulltest",
        action="store_true",
        help="Run full test with multiple tb_max_tokens (128, 256, 512, 1024, 2048).",
    )
    parser.add_argument(
        "--no_context",
        action="store_true",
        help="Answer questions directly without retrieving context from the tree.",
    )

    args = parser.parse_args()

    # Logic for full test
    if args.fulltest:
        tokens_list = [128, 256, 512, 1024, 2048]
        all_results = []
        print(
            f"Starting FULL TEST on dataset={args.dataset}, model={args.model}, local={args.local}"
        )
        for tokens in tokens_list:
            print(f"\n--- Running for tb_max_tokens={tokens} ---")
            try:
                res = evaluate_k_means_on_dataset(
                    dataset_name=args.dataset,
                    model_name=args.model,
                    local_test=args.local,
                    tb_max_tokens=tokens,
                    print_summary=False,
                    answer_without_context=args.no_context,
                )
                all_results.append((tokens, res))
            except Exception as e:
                logging.error(f"Failed run for tokens={tokens}: {e}")
                all_results.append((tokens, None))

        print("\n" + "=" * 60)
        print("FULL TEST SUMMARY RESULTS")
        print("=" * 60)
        print(f"{'Max Tokens':<15} | {'F1':<10} | {'EM':<10}")
        print("-" * 60)
        for tokens, res in all_results:
            if res:
                print(
                    f"{tokens:<15} | {res['f1']:.2f}{'':<6} | {res['exact_match']:.2f}"
                )
            else:
                print(f"{tokens:<15} | {'FAILED':<10} | {'FAILED':<10}")
        print("=" * 60)

    # If user runs without arguments, default to local test with squad/qwen (or safe defaults)
    elif len(sys.argv) == 1:
        # Default run behavior if no args provided (backward compatibility / safety)
        print("No arguments provided. Defaulting to SQuAD local test.")
        evaluate_k_means_on_dataset(
            dataset_name="squad", model_name="unifiedqa", local_test=True
        )
    else:
        evaluate_k_means_on_dataset(
            dataset_name=args.dataset,
            model_name=args.model,
            local_test=args.local,
            answer_without_context=args.no_context,
        )
