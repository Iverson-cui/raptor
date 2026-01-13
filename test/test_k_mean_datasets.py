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
from raptor.kmeans_retriever import KMeansRetrieverConfig
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
                "answers": {"text": valid_answers, "answer_start": [-1] * len(valid_answers)},
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
                "answers": {"text": all_answers, "answer_start": [-1] * len(all_answers)},
            }

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return extract_contexts, process_item


def evaluate_k_means_on_dataset(
    dataset_name="squad",
    model_name="qwen",
    local_test=True,
    tb_max_tokens=None,
    n_clusters=None,
    tr_top_k_clusters=None,
    tr_top_k=None,
    print_summary=True,
    answer_without_context=False,
    multi_retriever_configs=None,  # New parameter: list of dicts for retrieval benchmarking
    node_information=False,
    context_ratio=None,
):
    """
    Evaluates K-Means RAPTOR on the specified dataset.
    """
    print(f"Dataset: {dataset_name}, Model: {model_name}, local_test: {local_test}")
    logging.info(
        f"Starting evaluation (K-Means). Mode: {'Local Test' if local_test else 'Full Dataset'}"
    )

    # multi_retriever_configs is not None means we are entering free test mode
    # Determine Tree Building Params from the first config if provided
    # since when entering free test mode, the tb_max_tokens and n_clusters are the same for all configs
    if multi_retriever_configs:
        first_cfg = multi_retriever_configs[0]
        logging.info("Multi-retriever mode: Using first config for Tree Building.")
        # Override defaults with the first config's values in free test mode
        tb_max_tokens = first_cfg.get("tb_max_tokens", tb_max_tokens)
        n_clusters = first_cfg.get("n_clusters", n_clusters)
        tr_top_k_clusters = first_cfg.get("tr_top_k_clusters", tr_top_k_clusters)
        tr_top_k = first_cfg.get("tr_top_k", tr_top_k)

    # Load dataset object (concatenated splits if needed)
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
                # hard code to validation split only due to loading time
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

    # Get processors
    extract_contexts_fn, process_item_fn = get_dataset_processors(dataset_name)

    # Configuration for models
    if not local_test:
        qa_memory_map = {
            0: "0GiB",
            1: "45GiB",
            2: "45GiB",
            3: "0GiB",
            4: "0GiB",
            5: "45GiB",
            6: "45GiB",
        }
        embedding_device = "cuda:0"
    else:
        embedding_device = "mps"

    # Define slicing parameters
    if local_test:
        num_eval_questions_target = 100
        max_contexts_to_process = 200
        # In local test, we gather contexts only for these 5 questions
    else:
        # Load WHOLE dataset for tree building as requested
        # For full run, we iterate the whole dataset
        num_eval_questions_target = 150
        # max_contexts_to_process = 100

    # Collect Data (Synchronized Loop)
    logging.info("Gathering data (contexts and questions)...")
    # all_contexts is a list of unique context strings
    all_contexts = []
    # eval_items is a list of processed items containing questions and answers
    eval_items = []
    seen_contexts = set()

    # Iterate through the dataset
    for i, item in enumerate(dataset):
        # 1. Extract and store contexts in this row
        current_contexts = extract_contexts_fn(item)
        has_context = False
        for ctx in current_contexts:
            if ctx not in seen_contexts:
                all_contexts.append(ctx)
                seen_contexts.add(ctx)
            has_context = True

        # 2. Store eval item if needed, i.e. question in this row
        # Only store questions that have corresponding contexts
        if len(eval_items) < num_eval_questions_target and has_context:
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

    if context_ratio is not None and 0.0 < context_ratio < 1.0:
        logging.info(f"Context ratio set to {context_ratio}. Keeping first {context_ratio*100:.1f}% of contexts.")
        cutoff = int(len(all_contexts) * context_ratio)
        all_contexts = all_contexts[:cutoff]

    logging.info(f"Tree construction corpus: {len(all_contexts)} unique contexts.")
    logging.info(f"Evaluation target: {len(eval_items)} questions.")

    # Initialize Models
    if local_test:
        logging.info("Initializing LOCAL models: UnifiedQA (QA) & SBert (Embedding)...")
        qa_model = UnifiedQAModel()
        # embedding_model = SBertEmbeddingModel(device=embedding_device)
        embedding_model = BGEM3Model(device=embedding_device)
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

        embedding_model = BGEM3Model(device=embedding_device)

    # Configure for K-Means
    if local_test:
        default_tokens = 128
        default_n_clusters = 3
        default_tr_top_k_clusters = 3
        default_tr_top_k = 5
    else:
        if dataset_name in ["squad", "squad_v2"]:
            default_tokens = 256
            default_n_clusters = 210
            default_tr_top_k_clusters = 15
            default_tr_top_k = 10
        elif dataset_name in ["hotpot_qa", "ms_marco", "natural_questions", "trivia_qa"]:
            default_tokens = 256
            default_n_clusters = 2500
            default_tr_top_k_clusters = 50
            default_tr_top_k = 15
        else:
            default_tokens = 256
            default_n_clusters = 50
            default_tr_top_k_clusters = 5
            default_tr_top_k = 10

    # these 4 parameters can be directly given by the input args of the function
    # if these 4 are given explicitly, no matter what mode we are in, retrievalAugmentation will use them
    # if not given and if in freetest mode, we use the first config's values
    # if not given and not in freetest mode, we use the defaults defined above
    if tb_max_tokens is None:
        tb_max_tokens = default_tokens
    if n_clusters is None:
        n_clusters = default_n_clusters
    if tr_top_k_clusters is None:
        tr_top_k_clusters = default_tr_top_k_clusters
    if tr_top_k is None:
        tr_top_k = default_tr_top_k

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
    # tree building
    RA.add_documents(full_corpus, use_multithreading=not local_test)
    elapsed = time.time() - start_time
    logging.info(f"Tree built successfully in {elapsed:.2f} seconds.")

    # Multi-retriever setup
    retriever_names = ["default"]
    # if entering free test mode with multiple retriever configs
    if multi_retriever_configs:
        retriever_names = []
        for idx, cfg in enumerate(multi_retriever_configs):
            # retrievers' names are: config_0, config_1, ...
            # if multi_retriever_configs is given, there is no default, just start from config_0
            name = f"config_{idx}"
            # only k_clusters and k_chunks vary here because in free test mode tb_max_tokens and n_clusters are the same for all configs
            k_clusters = cfg.get("tr_top_k_clusters", tr_top_k_clusters)
            k_chunks = cfg.get("tr_top_k", tr_top_k)

            config = KMeansRetrieverConfig(
                top_k_clusters=k_clusters,
                embedding_model=embedding_model,
                top_k=k_chunks,
                context_embedding_model=RA.config.tree_retriever_config.context_embedding_model,
            )
            # self.retriever is a dict of retriever objects
            RA.add_retriever(name, config, retriever_type="kmeans")
            # retriever_names is a list of strings representing all of the retrievers we have
            retriever_names.append(name)

    # Verify Tree
    if hasattr(RA, "tree"):
        logging.info(f"Tree Layers: {RA.tree.num_layers}")
        if 1 in RA.tree.layer_to_nodes:
            logging.info(
                f"Number of clusters (Layer 1): {len(RA.tree.layer_to_nodes[1])}"
            )

    logging.info("Loading evaluation metric...")
    squad_metric = evaluate.load("squad")

    all_results_dict = {}

    # if not in free test mode, this loop runs only once with the default retriever
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
                        print(f"Layer info for question ID {item['id']}: {layer_info}")
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

        # this step will automatically normalize the answers and predictions
        results = squad_metric.compute(predictions=predictions, references=references)
        # if multiple retrievers are given, store results in a dict
        # in default mode, the dict only has one parameters
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

    # if free test mode return a dict, else directly return the single result
    return all_results_dict if multi_retriever_configs else all_results_dict["default"]


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
        choices=["squad", "hotpot_qa", "ms_marco", "squad_v2", "natural_questions", "trivia_qa"],
    )
    parser.add_argument(
        "--model", type=str, default="qwen", choices=["qwen", "deepseek", "unifiedqa"]
    )
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--fulltest", action="store_true")
    parser.add_argument("--freetest", action="store_true")
    parser.add_argument("--no_context", action="store_true")
    parser.add_argument(
        "--node_info", action="store_true", help="Save layer info for each question"
    )
    parser.add_argument(
        "--context_ratio", type=float, help="Ratio of contexts to use (0.0 - 1.0)"
    )

    # New arguments for customizing parameters
    parser.add_argument("--chunk_size", type=int, help="Chunk size for tree building")
    parser.add_argument("--n_clusters", type=int, help="Number of clusters for tree building")
    parser.add_argument("--top_k_clusters", type=int, help="Top k clusters for retrieval")
    parser.add_argument("--top_k", type=int, help="Top k chunks for retrieval")
    parser.add_argument(
        "--k_clusters_list",
        type=int,
        nargs="+",
        help="List of top_k_clusters for freetest (e.g., 3 5 10)",
    )
    parser.add_argument(
        "--k_chunks_list",
        type=int,
        nargs="+",
        help="List of top_k_chunks for freetest (e.g., 10 10 10)",
    )

    args = parser.parse_args()

    if args.fulltest and args.freetest:
        print("ERROR: --fulltest and --freetest cannot be used together.")
        sys.exit(1)

    # full test mode
    if args.fulltest:
        # in full test mode, tree building and retrieving process are both run several times
        all_results = []
        print(
            f"Starting FULL TEST on dataset={args.dataset}, model={args.model}, local={args.local}"
        )

        if args.dataset == "squad":
            # SQuAD-specific hyperparameter configurations (Original Logic)
            configs = [
                {
                    "tb_max_tokens": 128,
                    "n_clusters": 1500,
                    "tr_top_k_clusters": 10,
                    "tr_top_k": 20,
                },
                {
                    "tb_max_tokens": 256,
                    "n_clusters": 400,
                    "tr_top_k_clusters": 5,
                    "tr_top_k": 10,
                },
                {
                    "tb_max_tokens": 512,
                    "n_clusters": 400,
                    "tr_top_k_clusters": 5,
                    "tr_top_k": 5,
                },
                {
                    "tb_max_tokens": 1024,
                    "n_clusters": 200,
                    "tr_top_k_clusters": 3,
                    "tr_top_k": 3,
                },
            ]
            for config in configs:
                print(f"\n--- Running with config: {config} ---")
                try:
                    res = evaluate_k_means_on_dataset(
                        dataset_name=args.dataset,
                        model_name=args.model,
                        local_test=args.local,
                        tb_max_tokens=config["tb_max_tokens"],
                        n_clusters=config["n_clusters"],
                        tr_top_k_clusters=config["tr_top_k_clusters"],
                        tr_top_k=config["tr_top_k"],
                        print_summary=False,
                        answer_without_context=args.no_context,
                        node_information=args.node_info,
                        context_ratio=args.context_ratio,
                    )
                    all_results.append((config["tb_max_tokens"], res))
                except Exception as e:
                    logging.error(f"Failed run for config={config}: {e}")
                    all_results.append((config["tb_max_tokens"], None))
        else:
            # Original fulltest logic for non-SQuAD datasets
            tokens_list = [128, 256, 512, 1024, 2048]
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
                        node_information=args.node_info,
                        context_ratio=args.context_ratio,
                    )
                    all_results.append((tokens, res))
                except Exception as e:
                    logging.error(f"Failed run for tokens={tokens}: {e}")
                    all_results.append((tokens, None))

        print("\n" + "=" * 60)
        print("FULL TEST SUMMARY RESULTS")
        print("=" * 60)
        print(f"{ 'Max Tokens':<15} | {'F1':<10} | {'EM':<10}")
        print("-" * 60)
        for tokens, res in all_results:
            if res:
                print(
                    f"{tokens:<15} | {res['f1']:.2f}{'':<6} | {res['exact_match']:.2f}"
                )
            else:
                print(f"{tokens:<15} | {'FAILED':<10} | {'FAILED':<10}")
        print("=" * 60)

    # free test mode
    elif args.freetest:
        # in free test mode, tree building is done once, retrieval is done multiple times with varied parameters
        print(
            f"Starting FREE TEST on dataset={args.dataset}, model={args.model}, local={args.local}"
        )
        print("Using optimized multi-retriever benchmarking (building tree once).")

        # Consistent Tree Building parameters
        CHUNK_SIZE = args.chunk_size if args.chunk_size is not None else 256
        N_CLUSTERS = args.n_clusters if args.n_clusters is not None else 400

        # Varied Retrieval parameters
        tr_top_k_clusters_list = (
            args.k_clusters_list
            if args.k_clusters_list is not None
            else [3, 5, 10, 15, 20]
        )
        tr_top_k_chunks_list = (
            args.k_chunks_list
            if args.k_chunks_list is not None
            else [10, 10, 10, 10, 10]
        )  # top_k retrieved chunks

        multi_retriever_configs = []
        for kc, k in zip(tr_top_k_clusters_list, tr_top_k_chunks_list):
            multi_retriever_configs.append(
                {
                    "tb_max_tokens": CHUNK_SIZE,
                    "n_clusters": N_CLUSTERS,
                    "tr_top_k_clusters": kc,
                    "tr_top_k": k,
                }
            )

        all_results_dict = evaluate_k_means_on_dataset(
            dataset_name=args.dataset,
            model_name=args.model,
            local_test=args.local,
            multi_retriever_configs=multi_retriever_configs,
            answer_without_context=args.no_context,
            print_summary=False,
            node_information=args.node_info,
            context_ratio=args.context_ratio,
        )

        print("\n" + "=" * 100)
        print("FREE TEST SUMMARY RESULTS (MULTI-RETRIEVER)")
        print(f"Fixed: Chunk Size={CHUNK_SIZE}, N Clusters={N_CLUSTERS}")
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

    elif len(sys.argv) == 1:
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
            tb_max_tokens=args.chunk_size,
            n_clusters=args.n_clusters,
            tr_top_k_clusters=args.top_k_clusters,
            tr_top_k=args.top_k,
            node_information=args.node_info,
            context_ratio=args.context_ratio,
        )
