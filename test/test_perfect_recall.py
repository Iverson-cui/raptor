import os
import sys
import logging
import torch
import warnings
import time
import evaluate
import argparse
import json
import tiktoken
from datasets import load_dataset, concatenate_datasets

# Ensure the raptor package is accessible from the test directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from raptor.QAModels import UnifiedQAModel, QwenQAModel, DeepSeekQAModel
from raptor.EmbeddingModels import BGEM3Model, SBertEmbeddingModel
from raptor.utils import (
    split_text,
    distances_from_embeddings,
    indices_of_nearest_neighbors_from_distances,
)

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
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return extract_contexts, process_item


def evaluate_perfect_recall(
    dataset_name="squad",
    model_name="unifiedqa",
    local_test=True,
    chunk_size=256,
    top_k=5,
    num_questions=None,
):
    print(f"Dataset: {dataset_name}, Model: {model_name}, local_test: {local_test}")
    print(f"Chunk Size: {chunk_size}, Top-K: {top_k}")

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

    # Initialize Tokenizer (for chunking)
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Initialize Models
    if local_test:
        logging.info("Initializing LOCAL models: UnifiedQA & BGEM3...")
        qa_model = UnifiedQAModel()
        embedding_device = "mps" if torch.backends.mps.is_available() else "cpu"
        embedding_model = BGEM3Model(device=embedding_device)
    else:
        logging.info(f"Initializing SERVER models: {model_name} & BGEM3...")
        qa_memory_map = {
            0: "45GiB",
            1: "0GiB",
            2: "0GiB",
            3: "0GiB",
            4: "0GiB",
            5: "0GiB",
            6: "0GiB",
        }
        embedding_device = "cuda:0"  # As per test_k_mean_datasets.py

        if model_name.lower() == "qwen":
            qa_model = QwenQAModel(max_memory=qa_memory_map, device_map="auto")
        elif model_name.lower() == "deepseek":
            qa_model = DeepSeekQAModel(max_memory=qa_memory_map, device_map="auto")
        else:
            qa_model = QwenQAModel(max_memory=qa_memory_map, device_map="auto")

        embedding_model = BGEM3Model(device=embedding_device)

    squad_metric = evaluate.load("squad")

    predictions = []
    references = []

    count = 0
    # Use a loop that respects num_questions
    for i, item in enumerate(dataset):
        if count >= num_questions:
            break

        processed_item = process_item_fn(item)
        if not processed_item["answers"]["text"]:
            continue

        contexts = extract_contexts_fn(item)
        if not contexts:
            continue

        # 1. Join all ground truth contexts
        full_context = "\n\n".join(contexts)

        # 2. Chunk the context
        chunks = split_text(full_context, tokenizer, max_tokens=chunk_size)
        if not chunks:
            continue

        # 3. Embed chunks and question
        try:
            # Batch embedding for chunks
            chunk_embeddings = embedding_model.create_embedding(chunks)
            question_embedding = embedding_model.create_embedding(
                processed_item["question"]
            )
        except Exception as e:
            logging.error(f"Embedding failed for question {processed_item['id']}: {e}")
            continue

        # 4. Calculate distances and retrieve Top-K
        distances = distances_from_embeddings(
            question_embedding, chunk_embeddings, distance_metric="cosine"
        )
        sorted_indices = indices_of_nearest_neighbors_from_distances(distances)

        # Select top-k indices (smallest distance = closest)
        top_k_indices = sorted_indices[:top_k]
        top_k_chunks = [chunks[idx] for idx in top_k_indices]

        # Join retrieved chunks
        retrieved_context = "\n\n".join(top_k_chunks)

        # 5. Answer Question
        question = processed_item["question"]
        try:
            pred_answer = qa_model.answer_question(retrieved_context, question)
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
    print(f"Perfect Recall (Retrieval) Evaluation Results")
    print(f"Dataset: {dataset_name}, Model: {model_name}")
    print(f"Chunk Size: {chunk_size}, Top-K: {top_k}")
    print(f"Average F1: {results['f1']:.2f}")
    print(f"Average EM: {results['exact_match']:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Perfect Recall (Retrieval) on datasets."
    )
    parser.add_argument(
        "--dataset", type=str, default="squad", choices=["squad", "trivia_qa"]
    )
    parser.add_argument(
        "--model", type=str, default="qwen", choices=["qwen", "deepseek", "unifiedqa"]
    )
    parser.add_argument("--local", action="store_true", help="Run in local mode")
    parser.add_argument("--chunk_size", type=int, default=256, help="Tokens per chunk")
    parser.add_argument(
        "--top_k", type=int, default=5, help="Number of chunks to retrieve"
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
        chunk_size=args.chunk_size,
        top_k=args.top_k,
        num_questions=num_q,
    )
