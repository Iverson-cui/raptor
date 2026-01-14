import os
import sys
import logging
import torch
import warnings
import time
import evaluate
import argparse
import json
from datasets import load_dataset, concatenate_datasets

# Ensure the raptor package is accessible from the test directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from raptor.QAModels import UnifiedQAModel, QwenQAModel, DeepSeekQAModel

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Setup basic logging for debugging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


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

    elif dataset_name == "trivia_qa":

        def extract_contexts(item):
            contexts = []
            if "entity_pages" in item:
                for text in item["entity_pages"].get("wiki_context", []):
                    if text.strip():
                        contexts.append(text)
            return contexts

        def process_item(item):
            answer_val = item["answer"]["value"]
            aliases = item["answer"]["aliases"]
            all_answers = [answer_val] + aliases
            all_answers = list(set([a for a in all_answers if a]))
            return {
                "id": str(item["question_id"]),
                "question": item["question"],
                "answers": {
                    "text": all_answers,
                    "answer_start": [-1] * len(all_answers),
                },
            }

    else:
        # Simplified for other datasets if needed, but the user specifically asked for squad and trivia_qa
        raise ValueError(f"Unsupported dataset in perfect recall: {dataset_name}")

    return extract_contexts, process_item


def truncate_context(context, tokenizer, max_tokens):
    """
    Truncates the context to the specified number of tokens.
    """
    if tokenizer is None:
        # Naive character-based truncation if no tokenizer is available (fallback)
        # Assuming average 4 chars per token
        return context[: max_tokens * 4]

    tokens = tokenizer.encode(context, add_special_tokens=False)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        return tokenizer.decode(tokens)
    return context


def evaluate_perfect_recall(
    dataset_name="squad",
    model_name="unifiedqa",
    local_test=True,
    max_context_tokens=10000,
    num_questions=None,
):
    print(
        f"Dataset: {dataset_name}, Model: {model_name}, local_test: {local_test}, max_context_tokens: {max_context_tokens}"
    )

    # Set number of questions if not specified
    if num_questions is None:
        num_questions = 10 if local_test else 200

    # Load dataset
    try:
        if dataset_name == "squad":
            dataset = load_dataset("squad", split="validation")
        elif dataset_name == "trivia_qa":
            dataset = load_dataset("trivia_qa", "rc", split="validation")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    except Exception as e:
        logging.error(f"Could not load dataset {dataset_name}: {e}")
        return

    extract_contexts_fn, process_item_fn = get_dataset_processors(dataset_name)

    # Initialize Model
    if local_test:
        logging.info("Initializing LOCAL model: UnifiedQA...")
        qa_model = UnifiedQAModel()
    else:
        logging.info(f"Initializing SERVER model: {model_name}...")
        qa_memory_map = {
            0: "0GiB",
            1: "0GiB",
            2: "0GiB",
            3: "0GiB",
            4: "0GiB",
            5: "0GiB",
            6: "45GiB",
        }
        if model_name.lower() == "qwen":
            qa_model = QwenQAModel(max_memory=qa_memory_map, device_map="auto")
        elif model_name.lower() == "deepseek":
            qa_model = DeepSeekQAModel(max_memory=qa_memory_map, device_map="auto")
        else:
            qa_model = QwenQAModel(max_memory=qa_memory_map, device_map="auto")

    tokenizer = getattr(qa_model, "tokenizer", None)
    squad_metric = evaluate.load("squad")

    predictions = []
    references = []

    count = 0
    for item in dataset:
        if count >= num_questions:
            break

        processed_item = process_item_fn(item)
        if not processed_item["answers"]["text"]:
            continue

        contexts = extract_contexts_fn(item)
        if not contexts:
            continue

        full_context = "\n\n".join(contexts)

        # Truncate context if limit is provided
        if max_context_tokens:
            full_context = truncate_context(full_context, tokenizer, max_context_tokens)

        question = processed_item["question"]

        try:
            pred_answer = qa_model.answer_question(full_context, question)
        except Exception as e:
            logging.error(f"Error answering question {processed_item['id']}: {e}")
            pred_answer = ""

        predictions.append(
            {"id": processed_item["id"], "prediction_text": str(pred_answer)}
        )
        references.append(
            {"id": processed_item["id"], "answers": processed_item["answers"]}
        )

        count += 1
        if count % (1 if local_test else 10) == 0:
            logging.info(f"Processed {count}/{num_questions} questions.")

    results = squad_metric.compute(predictions=predictions, references=references)

    print("\n" + "=" * 50)
    print(f"Perfect Recall Evaluation Results")
    print(f"Dataset: {dataset_name}, Model: {model_name}")
    print(f"Max Context Tokens: {max_context_tokens}")
    print(f"Average F1: {results['f1']:.2f}")
    print(f"Average EM: {results['exact_match']:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Perfect Recall (All Context) on datasets."
    )
    parser.add_argument(
        "--dataset", type=str, default="squad", choices=["squad", "trivia_qa"]
    )
    parser.add_argument(
        "--model", type=str, default="qwen", choices=["qwen", "deepseek", "unifiedqa"]
    )
    parser.add_argument(
        "--local", action="store_true", help="Run in local mode (squad, 10 questions)"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=5000, help="Max tokens for context"
    )
    parser.add_argument(
        "--num_questions", type=int, help="Override number of questions to process"
    )

    args = parser.parse_args()

    # Adjust defaults based on local flag if not explicitly provided
    dataset = args.dataset
    num_q = args.num_questions

    if args.local:
        if not args.dataset:
            dataset = "squad"
        if num_q is None:
            num_q = 10
        model = "unifiedqa"
    else:
        if not args.dataset:
            dataset = "trivia_qa"
        if num_q is None:
            num_q = 200
        model = args.model

    evaluate_perfect_recall(
        dataset_name=dataset,
        model_name=model,
        local_test=args.local,
        max_context_tokens=args.max_tokens,
        num_questions=num_q,
    )
