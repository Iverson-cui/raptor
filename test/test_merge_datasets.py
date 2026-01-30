"""
if you want to test, run:
python test/test_merge_datasets.py --dataset trivia_qa --freetest --node_info --load_tree path_to_tree.pkl --k_clusters_list x x x --k_chunks_list y y y
"""

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

from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor.merge_tree_retriever import MergeTreeRetrieverConfig
from raptor.QAModels import UnifiedQAModel, QwenQAModel, DeepSeekQAModel
from raptor.EmbeddingModels import SBertEmbeddingModel, BGEM3Model
from raptor.SummarizationModels import BaseSummarizationModel

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
            doc_tokens = item["document"]["tokens"]
            token_list = doc_tokens["token"]
            is_html_list = doc_tokens["is_html"]
            valid_tokens = [
                t for t, is_html in zip(token_list, is_html_list) if not is_html
            ]
            return [" ".join(valid_tokens)]

        def process_item(item):
            doc_tokens = item["document"]["tokens"]["token"]
            valid_answers = []
            for ann in item["annotations"]:
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
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return extract_contexts, process_item


def evaluate_merge_on_dataset(
    dataset_name="squad",
    model_name="qwen",
    local_test=True,
    tb_max_tokens=None,
    n_clusters=None,
    tr_top_k_clusters=None,
    tr_top_k=None,
    merge_k_clusters=None,
    merge_k_chunks=None,
    print_summary=True,
    answer_without_context=False,
    multi_retriever_configs=None,
    node_information=False,
    context_ratio=None,
    load_tree_path=None,
):
    """
    Evaluates Merge Tree RAPTOR on the specified dataset.
    Compared to evaluate_merge_on_dataset, 2 arguments are added:
    merge_k_clusters and merge_k_chunks
    """
    print(f"Dataset: {dataset_name}, Model: {model_name}, local_test: {local_test}")
    logging.info(
        f"Starting evaluation (Merge Tree). Mode: {'Local Test' if local_test else 'Full Dataset'}"
    )

    if multi_retriever_configs:
        first_cfg = multi_retriever_configs[0]
        logging.info("Multi-retriever mode: Using first config for Tree Building.")
        tb_max_tokens = first_cfg.get("tb_max_tokens", tb_max_tokens)
        n_clusters = first_cfg.get("n_clusters", n_clusters)
        tr_top_k_clusters = first_cfg.get("tr_top_k_clusters", tr_top_k_clusters)
        tr_top_k = first_cfg.get("tr_top_k", tr_top_k)
        merge_k_clusters = first_cfg.get("merge_k_clusters", merge_k_clusters)
        merge_k_chunks = first_cfg.get("merge_k_chunks", merge_k_chunks)

    # Load dataset logic
    splits = ["validation"] if local_test else ["train", "validation"]
    loaded_splits = []
    for split in splits:
        try:
            if dataset_name == "squad":
                loaded_splits.append(load_dataset("squad", split=split))
            elif dataset_name == "squad_v2":
                loaded_splits.append(load_dataset("squad_v2", split=split))
            elif dataset_name == "hotpot_qa":
                loaded_splits.append(
                    load_dataset("hotpot_qa", "distractor", split=split)
                )
            elif dataset_name == "ms_marco":
                loaded_splits.append(load_dataset("ms_marco", "v1.1", split=split))
            elif dataset_name == "natural_questions":
                loaded_splits.append(load_dataset("natural_questions", split=split))
            elif dataset_name == "trivia_qa":
                loaded_splits.append(
                    load_dataset("trivia_qa", "rc", split="validation")
                )
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

    extract_contexts_fn, process_item_fn = get_dataset_processors(dataset_name)

    if not local_test:
        qa_memory_map = {
            0: "45GiB",
            1: "0GiB",
            2: "0GiB",
            3: "0GiB",
            4: "0GiB",
            5: "0GiB",
            6: "0GiB",
        }
        embedding_device = "cuda:0"
    else:
        embedding_device = "mps"

    if local_test:
        num_eval_questions_target = 100
        max_contexts_to_process = 200
    else:
        num_eval_questions_target = 200
        max_contexts_to_process = None

    logging.info("Gathering data (contexts and questions)...")
    all_contexts = []
    eval_items = []
    seen_contexts = set()

    for i, item in enumerate(dataset):
        current_contexts = extract_contexts_fn(item)
        has_context = False
        for ctx in current_contexts:
            if ctx not in seen_contexts:
                all_contexts.append(ctx)
                seen_contexts.add(ctx)
            has_context = True

        if len(eval_items) < num_eval_questions_target and has_context:
            processed_item = process_item_fn(item)
            if processed_item["answers"]["text"]:
                eval_items.append(processed_item)

        if local_test:
            if len(eval_items) >= num_eval_questions_target:
                break
        else:
            if (
                max_contexts_to_process is not None
                and len(all_contexts) >= max_contexts_to_process
            ):
                logging.info(
                    f"Reached max context limit ({max_contexts_to_process}). Stopping."
                )
                break

    if context_ratio is not None and 0.0 < context_ratio < 1.0:
        logging.info(
            f"Context ratio set to {context_ratio}. Keeping first {context_ratio*100:.1f}% of contexts."
        )
        cutoff = int(len(all_contexts) * context_ratio)
        all_contexts = all_contexts[:cutoff]

    logging.info(f"Tree construction corpus: {len(all_contexts)} unique contexts.")
    logging.info(f"Evaluation target: {len(eval_items)} questions.")

    if local_test:
        logging.info("Initializing LOCAL models: UnifiedQA (QA) & BGEM3 (Embedding)...")
        qa_model = UnifiedQAModel()
        embedding_model = BGEM3Model(device=embedding_device)
    else:
        logging.info(
            f"Initializing SERVER models: {model_name} (QA) & BGEM3 (Embedding)..."
        )
        if model_name.lower() == "qwen":
            qa_model = QwenQAModel(max_memory=qa_memory_map, device_map="auto")
        elif model_name.lower() == "deepseek":
            qa_model = DeepSeekQAModel(max_memory=qa_memory_map, device_map="auto")
        else:
            logging.warning(f"Unknown server model {model_name}, defaulting to Qwen.")
            qa_model = QwenQAModel(max_memory=qa_memory_map, device_map="auto")
        embedding_model = BGEM3Model(device=embedding_device)

    # Defaults for Merge Tree
    # TODO: These parameters need to be tuned properly for Merge Tree
    if local_test:
        default_tokens = 128
        default_n_clusters = 3
        default_tr_top_k_clusters = 3
        default_tr_top_k = 5
        default_merge_k_clusters = 3
        default_merge_k_chunks = 3
    else:
        default_tokens = 256
        default_n_clusters = 210
        default_tr_top_k_clusters = 15
        default_tr_top_k = 10
        default_merge_k_clusters = 10
        default_merge_k_chunks = 10

    if tb_max_tokens is None:
        tb_max_tokens = default_tokens
    if n_clusters is None:
        n_clusters = default_n_clusters
    if tr_top_k_clusters is None:
        tr_top_k_clusters = default_tr_top_k_clusters
    if tr_top_k is None:
        tr_top_k = default_tr_top_k
    if merge_k_clusters is None:
        merge_k_clusters = default_merge_k_clusters
    if merge_k_chunks is None:
        merge_k_chunks = default_merge_k_chunks

    logging.info(
        f"Configuring RAPTOR (Merge) with: n_clusters={n_clusters}, "
        f"tb_max_tokens={tb_max_tokens}, tr_top_k_clusters={tr_top_k_clusters}, "
        f"tr_top_k={tr_top_k}, merge_k_clusters={merge_k_clusters}, merge_k_chunks={merge_k_chunks}"
    )

    RAC = RetrievalAugmentationConfig(
        tree_builder_type="merge",
        tree_retriever_type="merge",
        tb_n_clusters=n_clusters,
        tb_merge_top_k_clusters=merge_k_clusters,
        tb_merge_top_k_chunks=merge_k_chunks,
        tr_top_k_clusters=tr_top_k_clusters,
        tr_top_k=tr_top_k,
        qa_model=qa_model,
        embedding_model=embedding_model,
        summarization_model=MockSummarizationModel(),
        tb_max_tokens=tb_max_tokens,
    )

    if load_tree_path:
        logging.info(f"Loading tree from {load_tree_path}...")
        # still use RAC configuration
        RA = RetrievalAugmentation(config=RAC, tree=load_tree_path)
    else:
        RA = RetrievalAugmentation(config=RAC)

    if not load_tree_path:
        logging.info("Joining contexts into full corpus...")
        full_corpus = "\n\n".join(all_contexts)

        logging.info("Building Merge RAPTOR tree...")
        start_time = time.time()
        RA.add_documents(full_corpus, use_multithreading=not local_test)
        elapsed = time.time() - start_time
        logging.info(f"Tree built successfully in {elapsed:.2f} seconds.")
    else:
        logging.info("Skipping tree building (tree loaded).")

    retriever_names = ["default"]
    if multi_retriever_configs:
        retriever_names = []
        for idx, cfg in enumerate(multi_retriever_configs):
            name = f"config_{idx}"
            k_clusters = cfg.get("tr_top_k_clusters", tr_top_k_clusters)
            k_chunks = cfg.get("tr_top_k", tr_top_k)

            config = MergeTreeRetrieverConfig(
                top_k_clusters=k_clusters,
                embedding_model=embedding_model,
                top_k=k_chunks,
                context_embedding_model=RA.config.tree_retriever_config.context_embedding_model,
            )
            RA.add_retriever(name, config, retriever_type="merge")
            retriever_names.append(name)

    if hasattr(RA, "tree"):
        logging.info(f"Tree Layers: {RA.tree.num_layers}")

    logging.info("Loading evaluation metric...")
    squad_metric = evaluate.load("squad")

    all_results_dict = {}

    for r_name in retriever_names:
        logging.info(
            f"Starting Q&A evaluation loop for retriever '{r_name}' with {len(eval_items)} questions..."
        )
        predictions = []
        references = []
        node_infos = {}

        for i, item in enumerate(eval_items):
            question = item["question"]
            try:
                if answer_without_context:
                    if hasattr(qa_model, "answer_question_without_contexts"):
                        pred_answer = qa_model.answer_question_without_contexts(
                            context=None, question=question
                        )
                    else:
                        pred_answer = qa_model.answer_question(
                            context="", question=question
                        )
                else:
                    response = RA.answer_question(
                        question=question,
                        retriever_name=r_name,
                        return_layer_information=node_information,
                    )
                    if node_information and isinstance(response, tuple):
                        pred_answer, layer_info = response
                        node_infos[item["id"]] = layer_info
                    else:
                        pred_answer = (
                            response[0] if isinstance(response, tuple) else response
                        )
            except Exception as e:
                logging.error(f"Error answering question {i} with {r_name}: {e}")
                pred_answer = ""

            predictions.append({"id": item["id"], "prediction_text": str(pred_answer)})
            references.append({"id": item["id"], "answers": item["answers"]})

            log_freq = 1 if local_test else 10
            if (i + 1) % log_freq == 0:
                logging.info(
                    f"Checkpoint ({r_name}): Processed {i + 1}/{len(eval_items)} questions."
                )

        if node_information:
            fname = f"node_info_{dataset_name}_{tb_max_tokens}_{r_name}.json"
            try:
                with open(fname, "w") as f:
                    json.dump(node_infos, f, indent=4)
                logging.info(f"Saved node info to {fname}")
            except Exception as e:
                logging.error(f"Failed to save node info: {e}")

        results = squad_metric.compute(predictions=predictions, references=references)
        all_results_dict[r_name] = results

        if print_summary:
            print("\n" + "=" * 50)
            print(f"Final Evaluation Results for retriever '{r_name}'")
            print(
                f"Dataset: {dataset_name}, Model: {model_name}, Chunk size: {tb_max_tokens}"
            )
            print(
                f"n_clusters={n_clusters}, tb_max_tokens={tb_max_tokens}, top_k={tr_top_k}"
            )
            print(f"Average F1: {results['f1']:.2f}")
            print(f"Average EM: {results['exact_match']:.2f}")
            print("=" * 50)

    return all_results_dict if multi_retriever_configs else all_results_dict["default"]


if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "not_used"

    parser = argparse.ArgumentParser(
        description="Evaluate RAPTOR Merge Tree on various datasets."
    )
    # --dataset let us to choose a specific dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="squad",
        choices=[
            "squad",
            "hotpot_qa",
            "ms_marco",
            "squad_v2",
            "natural_questions",
            "trivia_qa",
        ],
    )
    # --model let us to choose a specific QA model
    parser.add_argument(
        "--model", type=str, default="qwen", choices=["qwen", "deepseek", "unifiedqa"]
    )
    # --local: local test or full dataset test
    parser.add_argument("--local", action="store_true")
    # --freetest: run free test with multiple retriever configs
    # chunk_size, n_clusters, merge_k_clusters, merge_k_chunks are specific
    # top_k_clusters and top_k_chunks are given as lists for freetest
    parser.add_argument("--freetest", action="store_true")
    # --no_context: answer without context
    parser.add_argument("--no_context", action="store_true")
    # --node_info: save node information in a given file. Default to False
    parser.add_argument(
        "--node_info", action="store_true", help="Save layer info for each question"
    )
    # defaultly not used
    parser.add_argument(
        "--context_ratio", type=float, help="Ratio of contexts to use (0.0 - 1.0)"
    )
    # provide the path of tree pkl file
    parser.add_argument(
        "--load_tree",
        type=str,
        default=None,
        help="Path to load a pre-built tree (pickle format)",
    )

    # used in freetest. This is set to one set value for multiple retrievers.
    parser.add_argument("--chunk_size", type=int, help="Chunk size for tree building")
    # used in freetest
    parser.add_argument(
        "--n_clusters", type=int, help="Number of clusters for tree building"
    )
    # top-k-clusters. It's int because this is only used in non-freetest.
    parser.add_argument(
        "--top_k_clusters", type=int, help="Top k clusters for retrieval"
    )
    # top-k-chunks. It's int because this is only used in non-freetest.
    parser.add_argument("--top_k", type=int, help="Top k chunks for retrieval")
    # used in freetest.
    parser.add_argument(
        "--merge_k_clusters", type=int, help="Merge tree: top k clusters"
    )
    # used in freetest.
    parser.add_argument("--merge_k_chunks", type=int, help="Merge tree: top k chunks")

    # used in freetest so it's a list
    parser.add_argument(
        "--k_clusters_list",
        type=int,
        nargs="+",
        help="List of top_k_clusters for freetest",
    )
    # used in freetest so it's a list
    parser.add_argument(
        "--k_chunks_list", type=int, nargs="+", help="List of top_k_chunks for freetest"
    )

    args = parser.parse_args()

    if args.freetest:
        print(
            f"Starting FREE TEST on dataset={args.dataset}, model={args.model}, local={args.local}"
        )

        CHUNK_SIZE = args.chunk_size if args.chunk_size is not None else 256
        N_CLUSTERS = args.n_clusters if args.n_clusters is not None else 400
        MERGE_K_CLUSTERS = (
            args.merge_k_clusters if args.merge_k_clusters is not None else 10
        )
        MERGE_K_CHUNKS = args.merge_k_chunks if args.merge_k_chunks is not None else 10

        tr_top_k_clusters_list = (
            args.k_clusters_list
            if args.k_clusters_list is not None
            else [3, 5, 10, 15, 20]
        )
        tr_top_k_chunks_list = (
            args.k_chunks_list
            if args.k_chunks_list is not None
            else [10, 10, 10, 10, 10]
        )

        multi_retriever_configs = []
        for kc, k in zip(tr_top_k_clusters_list, tr_top_k_chunks_list):
            multi_retriever_configs.append(
                {
                    "tb_max_tokens": CHUNK_SIZE,
                    "n_clusters": N_CLUSTERS,
                    "merge_k_clusters": MERGE_K_CLUSTERS,
                    "merge_k_chunks": MERGE_K_CHUNKS,
                    "tr_top_k_clusters": kc,
                    "tr_top_k": k,
                }
            )

        all_results_dict = evaluate_merge_on_dataset(
            dataset_name=args.dataset,
            model_name=args.model,
            local_test=args.local,
            multi_retriever_configs=multi_retriever_configs,
            answer_without_context=args.no_context,
            print_summary=False,
            node_information=args.node_info,
            context_ratio=args.context_ratio,
            load_tree_path=args.load_tree,
        )

        print("\n" + "=" * 100)
        print("FREE TEST SUMMARY RESULTS (MULTI-RETRIEVER) - MERGE TREE")
        print(
            f"Fixed: Chunk Size={CHUNK_SIZE}, N Clusters={N_CLUSTERS}, Merge K Clusters={MERGE_K_CLUSTERS}"
        )
        print("=" * 100)
        print(
            f"{ 'Retriever':<12} | {'TopK_Cl':<8} | {'TopK_Ch':<8} | {'F1':<10} | {'EM':<10}"
        )
        print("-" * 100)
        for idx, (name, res) in enumerate(all_results_dict.items()):
            cfg = multi_retriever_configs[idx]
            if res:
                print(
                    f"{name:<12} | {cfg['tr_top_k_clusters']:<8} | {cfg['tr_top_k']:<8} | {res['f1']:.2f}{'':<6} | {res['exact_match']:.2f}"
                )
            else:
                print(
                    f"{name:<12} | {cfg['tr_top_k_clusters']:<8} | {cfg['tr_top_k']:<8} | {'FAILED':<10} | {'FAILED':<10}"
                )
        print("=" * 100)

    # if not freetest, run single test, single parameter settings
    else:
        evaluate_merge_on_dataset(
            dataset_name=args.dataset,
            model_name=args.model,
            local_test=args.local,
            answer_without_context=args.no_context,
            tb_max_tokens=args.chunk_size,
            n_clusters=args.n_clusters,
            tr_top_k_clusters=args.top_k_clusters,
            tr_top_k=args.top_k,
            merge_k_clusters=args.merge_k_clusters,
            merge_k_chunks=args.merge_k_chunks,
            node_information=args.node_info,
            context_ratio=args.context_ratio,
            load_tree_path=args.load_tree,
        )
