import os
import sys
import logging
import argparse
import copy
import time
import numpy as np
from datasets import load_dataset, concatenate_datasets

# Add parent dir to path to import raptor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor.kmeans_retriever import KMeansRetriever, KMeansRetrieverConfig
from raptor.kmeans_tree_builder import KMeansTreeBuilder, KMeansTreeConfig
from raptor.tree_structures import Tree
from raptor.QAModels import QwenQAModel, BaseQAModel
from raptor.EmbeddingModels import BGEM3Model

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


class MockQAModel(BaseQAModel):
    def __init__(self):
        pass

    def answer_question(self, context, question):
        return "Mock Answer"


def get_dataset_processors(dataset_name):
    """
    Returns extraction and processing functions for the specified dataset.
    Every dataset is different. The row in a dataset contains different fields.
    So for each dataset, we define two functions:
    1. extract_contexts(item): extracts a list of context strings from the dataset row
    2. process_item(item): extracts the questions and answers
    No matter what dataset, process_item should return a dict with keys:
        - 'id': unique identifier for the question
        - 'question': the question string
        - 'answers': a dict with key 'text' containing a list of ground truth answers
        - 'answer_start' is optional, can be set to -1 if not available
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

    elif dataset_name == "squad_v2":

        def extract_contexts(item):
            return [item["context"]]

        def process_item(item):
            return {
                "id": str(item["id"]),
                "question": item["question"],
                "answers": item["answers"],
            }

    elif dataset_name == "natural_questions":

        def extract_contexts(item):
            # item['document']['tokens'] is {'token': [...], 'is_html': [...]}
            doc_tokens = item["document"]["tokens"]
            token_list = doc_tokens["token"]
            is_html_list = doc_tokens["is_html"]
            # Filter out HTML tokens to get clean context
            valid_tokens = [
                t for t, is_html in zip(token_list, is_html_list) if not is_html
            ]
            return [" ".join(valid_tokens)]

        def process_item(item):
            # Extract answers from annotations
            # item['annotations'] is a list of dicts
            doc_tokens = item["document"]["tokens"]["token"]
            valid_answers = []
            for ann in item["annotations"]:
                # short_answers is {'start_token': [...], 'end_token': [...]}
                starts = ann["short_answers"]["start_token"]
                ends = ann["short_answers"]["end_token"]
                for s, e in zip(starts, ends):
                    ans_str = " ".join(doc_tokens[s:e])
                    if ans_str:
                        valid_answers.append(ans_str)

            valid_answers = list(set(valid_answers))
            return {
                "id": str(item["id"]),
                "question": item["question"]["text"],
                "answers": {
                    "text": valid_answers,
                    "answer_start": [-1] * len(valid_answers),
                },
            }

    elif dataset_name == "trivia_qa":

        def extract_contexts(item):
            contexts = []
            # item['entity_pages'] is {'wiki_context': [...], ...}
            if "entity_pages" in item:
                for text in item["entity_pages"].get("wiki_context", []):
                    if text.strip():
                        contexts.append(text)

            # # item['search_results'] is {'search_context': [...], ...}
            # if "search_results" in item:
            #     for text in item["search_results"].get("search_context", []):
            #         if text.strip():
            #             contexts.append(text)
            return contexts

        def process_item(item):
            # item['answer'] is {'aliases': [...], 'value': ...}
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
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return extract_contexts, process_item


def load_data(dataset_name, split="validation", sample_limit=300, need_contexts=True):
    logging.info(f"Loading dataset: {dataset_name} ({split})")
    try:
        if dataset_name == "squad":
            dataset = load_dataset("squad", split=split)
        elif dataset_name == "squad_v2":
            dataset = load_dataset("squad_v2", split=split)
        elif dataset_name == "trivia_qa":
            dataset = load_dataset("trivia_qa", "rc", split="validation")
        else:
            raise ValueError(f"Unsupported dataset for this script: {dataset_name}")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    extract_contexts_fn, process_item_fn = get_dataset_processors(dataset_name)

    all_contexts = []
    eval_items = []
    seen_contexts = set()

    logging.info("Processing data...")
    for item in dataset:
        # Optimization: Skip context processing if we don't need them (e.g. tree loaded)
        if need_contexts:
            current_contexts = extract_contexts_fn(item)
            has_context = False
            for ctx in current_contexts:
                if ctx not in seen_contexts:
                    all_contexts.append(ctx)
                    seen_contexts.add(ctx)
                has_context = True
        else:
            # If we don't need contexts, we assume we can just take the question
            # But we might want to ensure the question is valid/answerable if that depends on context presence in the original logic
            # For SQuAD, it's fine.
            has_context = True

        # extract questions and answers
        if len(eval_items) < sample_limit and has_context:
            processed = process_item_fn(item)
            if processed["answers"]["text"]:
                eval_items.append(processed)

        # Stop early if we have enough questions and don't need full context corpus
        if not need_contexts and len(eval_items) >= sample_limit:
            break

    logging.info(
        f"Loaded {len(all_contexts)} unique contexts and {len(eval_items)} evaluation questions."
    )
    return all_contexts, eval_items


def run_retrieval(ra_instance, queries, retriever_name="default"):
    results = {}
    logging.info(f"Running retrieval with retriever: {retriever_name}")

    for item in queries:
        qid = item["id"]
        question = item["question"]

        try:
            _, node_indices = ra_instance.retrieve(
                question=question,
                retriever_name=retriever_name,
                return_layer_information=True,
            )
            results[qid] = set(node_indices)
        except Exception as e:
            logging.error(f"Error processing query {qid}: {e}")
            results[qid] = set()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Recall Comparison Test for RAPTOR K-Means"
    )

    # Dataset args
    parser.add_argument("--dataset", type=str, default="squad", help="Dataset name")
    parser.add_argument(
        "--num_questions", type=int, default=300, help="Number of questions to evaluate"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=256, help="Chunk size (tokens)"
    )

    # Baseline args
    parser.add_argument(
        "--base_n_clusters",
        type=int,
        required=True,
        help="Baseline: Number of clusters",
    )
    parser.add_argument(
        "--base_top_k_clusters",
        type=int,
        default=5,
        help="Baseline: Top K clusters to search",
    )
    parser.add_argument(
        "--base_top_k_chunks",
        type=int,
        default=10,
        help="Baseline: Top K chunks to retrieve",
    )
    parser.add_argument(
        "--load_tree",
        type=str,
        default=None,
        help="Path to a pickled tree for baseline",
    )

    # Experiment args
    parser.add_argument("--exp_n_clusters", type=int, required=True, help="Experiment: Number of clusters")
    parser.add_argument("--exp_top_k_clusters", type=int, default=5, help="Experiment: Top K clusters to search")
    parser.add_argument("--exp_top_k_chunks", type=int, default=10, help="Experiment: Top K chunks to retrieve")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for embedding model (e.g., cuda:0, mps, cpu)")

    args = parser.parse_args()

    # 1. Load Data
    # Optimization: If loading a tree, we don't need to parse/collect all contexts from the dataset
    contexts, queries = load_data(
        args.dataset, 
        sample_limit=args.num_questions, 
        need_contexts=(args.load_tree is None)
    )
    full_corpus = "\n\n".join(contexts)

    # Common Embedding Model
    # Using BGEM3 as standard in this repo
    # Use specified device
    embedding_model = BGEM3Model(device=args.device) 
    
    # Use Mock QA Model to avoid loading Qwen path which fails
    mock_qa = MockQAModel()

    logging.info("--- Setting up Baseline System ---")

    ra1_config = RetrievalAugmentationConfig(
        tree_builder_type="kmeans",
        tree_retriever_type="kmeans",
        tb_max_tokens=args.chunk_size,
        tb_n_clusters=args.base_n_clusters,
        tr_top_k_clusters=args.base_top_k_clusters,
        tr_top_k=args.base_top_k_chunks,
        embedding_model=embedding_model,
        qa_model=mock_qa,
    )

    # Initialize Baseline RA (RA1)
    if args.load_tree:
        logging.info(f"Loading Baseline Tree from {args.load_tree}...")
        ra1 = RetrievalAugmentation(config=ra1_config, tree=args.load_tree)
    else:
        ra1 = RetrievalAugmentation(config=ra1_config)
        logging.info("Building Baseline Tree (and creating leaf nodes)...")
        start_time = time.time()
        ra1.add_documents(full_corpus)
        logging.info(f"Baseline Tree built in {time.time() - start_time:.2f}s")

    # Run Baseline Retrieval
    results_baseline = run_retrieval(ra1, queries)

    # 3. Initialize Experiment RA (RA2) reusing Leaf Nodes
    logging.info("--- Setting up Experiment System ---")

    # Extract leaf nodes from RA1
    # RA1.tree.leaf_nodes is a dict {index: Node}
    leaf_nodes = ra1.tree.leaf_nodes
    logging.info(f"Reusing {len(leaf_nodes)} leaf nodes from Baseline.")

    # Check embedding keys in leaf nodes to ensure compatibility
    cluster_embedding_model = "BGEM3"
    if leaf_nodes:
        first_node = next(iter(leaf_nodes.values()))
        available_keys = list(first_node.embeddings.keys())
        logging.info(f"Embeddings found in leaf nodes: {available_keys}")

        # If the tree uses a different key than 'BGEM3' (default in config), update config
        # Assuming we want to use the first available embedding model for clustering
        if "BGEM3" not in available_keys and available_keys:
            cluster_embedding_model = available_keys[0]
            logging.info(
                f"Switching cluster_embedding_model to '{cluster_embedding_model}' based on loaded tree."
            )

    # Prepare data for manual tree construction
    leaf_nodes_2 = copy.deepcopy(leaf_nodes)

    # Setup RA2 Config
    ra2_config = RetrievalAugmentationConfig(
        tree_builder_type="kmeans",
        tree_retriever_type="kmeans",
        tb_max_tokens=args.chunk_size,
        tb_n_clusters=args.exp_n_clusters,
        tr_top_k_clusters=args.exp_top_k_clusters,
        tr_top_k=args.exp_top_k_chunks,
        tb_cluster_embedding_model=cluster_embedding_model,  # Ensure consistency
        tr_context_embedding_model=cluster_embedding_model,  # Ensure consistency
        embedding_model=embedding_model,
        qa_model=mock_qa,
    )

    ra2 = RetrievalAugmentation(config=ra2_config)

    logging.info("Building Experiment Tree (Clustering existing leaves)...")
    start_time = time.time()

    all_nodes_2 = copy.deepcopy(leaf_nodes_2)
    layer_to_nodes_2 = {0: list(leaf_nodes_2.values())}

    root_nodes_2 = ra2.tree_builder.construct_tree(
        current_level_nodes=all_nodes_2,
        all_tree_nodes=all_nodes_2,
        layer_to_nodes=layer_to_nodes_2,
    )

    ra2.tree = Tree(
        all_nodes=all_nodes_2,
        root_nodes=root_nodes_2,
        leaf_nodes=leaf_nodes_2,
        num_layers=ra2.tree_builder.num_layers,
        layer_to_nodes=layer_to_nodes_2,
    )

    retriever_class = KMeansRetriever
    ra2.retriever = retriever_class(ra2_config.tree_retriever_config, ra2.tree)
    ra2.retrievers["default"] = ra2.retriever

    logging.info(f"Experiment Tree built in {time.time() - start_time:.2f}s")

    results_experiment = run_retrieval(ra2, queries)

    logging.info("--- Calculating Statistics ---")

    overlaps = []

    for item in queries:
        qid = item["id"]
        base_set = results_baseline.get(qid, set())
        exp_set = results_experiment.get(qid, set())

        if not base_set:
            continue

        intersection = base_set.intersection(exp_set)
        recall = len(intersection) / len(base_set)
        overlaps.append(recall)

    avg_recall = np.mean(overlaps) if overlaps else 0.0

    print("\n" + "=" * 60)
    print("RECALL COMPARISON TEST SUMMARY")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Questions Evaluated: {len(overlaps)}")
    print(f"Chunk Size: {args.chunk_size}")
    print("-" * 60)
    print(
        f"{ 'Configuration':<15} | {'N Clusters':<10} | {'TopK Clust':<10} | {'TopK Chunks':<10}"
    )
    print(
        f"{ 'Baseline':<15} | {args.base_n_clusters:<10} | {args.base_top_k_clusters:<10} | {args.base_top_k_chunks:<10}"
    )
    print(
        f"{ 'Experiment':<15} | {args.exp_n_clusters:<10} | {args.exp_top_k_clusters:<10} | {args.exp_top_k_chunks:<10}"
    )
    print("-" * 60)
    print(f"Average Overlap (Recall): {avg_recall:.4f} ({avg_recall*100:.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
