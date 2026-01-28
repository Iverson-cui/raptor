from datasets import load_dataset
import tiktoken
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
chunk_sizes = [256, 512, 1024]


def inspect_dataset_structure(dataset_name, dataset, num_samples=1):
    """Inspect the structure of a dataset by showing fields and sample rows."""
    print("\n" + "=" * 80)
    print(f"DATASET STRUCTURE: {dataset_name}")
    print("=" * 80)

    # Get column names
    try:
        print(f"\nTotal Rows: {len(dataset)}")
    except TypeError:
        print(f"\nTotal Rows: Unknown (Streaming)")

    fields = None
    if hasattr(dataset, "column_names"):
        print(f"\nFields ({len(dataset.column_names)}): {dataset.column_names}")
        fields = dataset.column_names
    elif hasattr(dataset, "features"):
        print(f"\nFields ({len(dataset.features)}): {list(dataset.features.keys())}")
        fields = list(dataset.features.keys())
    else:
        print(f"\nFields: Unknown (Streaming/No metadata)")

    # Show sample rows
    print(f"\n--- Sample Row(s) ---")

    try:
        iterator = iter(dataset)
        for i in range(num_samples):
            try:
                row = next(iterator)
                print(f"\nRow {i}:")

                # If fields are known, use them, else use row keys
                row_fields = fields if fields else row.keys()

                for field_name in row_fields:
                    value = row.get(field_name)
                    # Truncate long strings/lists for readability
                    if isinstance(value, str):
                        display_value = value[:200] + "..." if len(value) > 200 else value
                    elif isinstance(value, list):
                        display_value = f"[List with {len(value)} items]"
                        if len(value) > 0:
                            display_value += f" First item: {str(value[0])[:100]}..."
                    elif isinstance(value, dict):
                        display_value = f"[Dict with keys: {list(value.keys())}]"
                    else:
                        display_value = value
                    print(f"  {field_name}: {display_value}")
            except StopIteration:
                break
    except Exception as e:
        print(f"Error inspecting rows: {e}")
    print()


def analyze_dataset(dataset_name, dataset, context_field, limit=None):
    """Analyze a dataset and print chunk statistics."""
    print("\n" + "=" * 80)
    print(f"DATASET: {dataset_name}")
    print("=" * 80)

    # Get unique contexts
    unique_contexts = set()
    print("Extracting unique contexts...")

    i = 0
    # Use the column directly if possible for speed, but iterate for dicts/lists
    for i, item in enumerate(dataset):
        if limit and i >= limit:
            print(f"Reached limit of {limit} items.")
            break

        val = item.get(context_field)
        if val is None:
            continue

        if isinstance(val, str):
            if val.strip():
                unique_contexts.add(val)
        elif isinstance(val, dict):
            # Handle MS MARCO style passages
            if "passage_text" in val:
                for text in val["passage_text"]:
                    if text and isinstance(text, str) and text.strip():
                        unique_contexts.add(text)
            # Handle Natural Questions (often has 'html' or 'tokens')
            elif "html" in val:
                text = val["html"]
                if text and isinstance(text, str) and text.strip():
                    unique_contexts.add(text)
            # Handle TriviaQA
            elif "wiki_context" in val:
                for text in val["wiki_context"]:
                    if text and isinstance(text, str) and text.strip():
                        unique_contexts.add(text)
            # Handle KILT Wikipedia (text -> paragraph list)
            elif "paragraph" in val:
                for text in val["paragraph"]:
                    if text and isinstance(text, str) and text.strip():
                        unique_contexts.add(text)
        elif isinstance(val, list):
            for sub_val in val:
                if isinstance(sub_val, str) and sub_val.strip():
                    unique_contexts.add(sub_val)

    try:
        print(f"Total Rows: {len(dataset)}")
    except TypeError:
        # Use i+1 because i is 0-indexed, but if loop didn't run i might be stale or 0
        # If dataset empty, i might not be defined if using 'enumerate' without start?
        # Actually loop var leaks in python, but if loop doesn't enter, i is unbound.
        pass

    print(f"Unique Contexts: {len(unique_contexts)}")

    if not unique_contexts:
        print("No contexts found.")
        return

    # Chunk analysis
    print("\nCHUNK ANALYSIS:")
    print("-" * 80)

    print("Tokenizing unique contexts...")
    # Optimize: tokenize once, then calculate chunks for different sizes
    token_counts = []
    for context in unique_contexts:
        token_counts.append(len(tokenizer.encode(context)))

    for chunk_size in chunk_sizes:
        total_chunks = sum((tc + chunk_size - 1) // chunk_size for tc in token_counts)
        print(f"Chunk size {chunk_size:4d} tokens: {total_chunks:10d} chunks")
        print(f"\nTotal Tokens: {sum(token_counts):,}")


# def analyze_dataset(dataset_name, dataset, context_field, limit=None):
#     """Analyze a dataset and print chunk statistics."""
#     print("\n" + "=" * 80)
#     print(f"DATASET: {dataset_name}")
#     print("=" * 80)

#     # Get unique contexts
#     unique_contexts = set()
#     print("Extracting unique contexts...")

#     # Determine if we can get the dataset length for progress bar
#     try:
#         total_rows = len(dataset) if not limit else min(len(dataset), limit)
#         pbar = tqdm(total=total_rows, desc="Processing rows", unit="rows")
#     except (TypeError, AttributeError):
#         # Streaming dataset or no length available
#         if limit:
#             pbar = tqdm(total=limit, desc="Processing rows", unit="rows")
#         else:
#             pbar = tqdm(desc="Processing rows", unit="rows")

#     i = 0
#     total_tokens = 0
#     # Use the column directly if possible for speed, but iterate for dicts/lists
#     for i, item in enumerate(dataset):
#         if limit and i >= limit:
#             print(f"Reached limit of {limit} items.")
#             break

#         val = item.get(context_field)
#         if val is None:
#             pbar.update(1)
#             continue

#         if isinstance(val, str):
#             if val.strip():
#                 unique_contexts.add(val)
#         elif isinstance(val, dict):
#             # Handle MS MARCO style passages
#             if "passage_text" in val:
#                 for text in val["passage_text"]:
#                     if text and isinstance(text, str) and text.strip():
#                         unique_contexts.add(text)
#             # Handle Natural Questions (often has 'html' or 'tokens')
#             elif "html" in val:
#                 text = val["html"]
#                 if text and isinstance(text, str) and text.strip():
#                     unique_contexts.add(text)
#             # Handle TriviaQA
#             elif "wiki_context" in val:
#                 for text in val["wiki_context"]:
#                     if text and isinstance(text, str) and text.strip():
#                         # unique_contexts.add(text)
#                         total_tokens += len(tokenizer.encode(text))
#             # Handle KILT Wikipedia (text -> paragraph list)
#             elif "paragraph" in val:
#                 for text in val["paragraph"]:
#                     if text and isinstance(text, str) and text.strip():
#                         # unique_contexts.add(text)
#                         total_tokens += len(tokenizer.encode(text))
#         elif isinstance(val, list):
#             for sub_val in val:
#                 if isinstance(sub_val, str) and sub_val.strip():
#                     # unique_contexts.add(sub_val)
#                     total_tokens += len(tokenizer.encode(sub_val))

#         pbar.update(1)

#     pbar.close()

#     try:
#         print(f"Total Rows: {len(dataset)}")
#     except TypeError:
#         # Use i+1 because i is 0-indexed, but if loop didn't run i might be stale or 0
#         # If dataset empty, i might not be defined if using 'enumerate' without start?
#         # Actually loop var leaks in python, but if loop doesn't enter, i is unbound.
#         pass

#     print(f"total tokens: {total_tokens}")

#     # if not unique_contexts:
#     #     print("No contexts found.")
#     #     return

#     # Chunk analysis
#     print("\nCHUNK ANALYSIS:")
#     print("-" * 80)

#     print("Tokenizing unique contexts...")
#     # Optimize: tokenize once, then calculate chunks for different sizes
#     token_counts = []
#     for context in tqdm(unique_contexts, desc="Tokenizing unique contexts", unit="ctx"):
#         token_counts.append(len(tokenizer.encode(context)))

#     for chunk_size in chunk_sizes:
#         total_chunks = total_tokens / chunk_size
#         print(f"Chunk size {chunk_size:4d} tokens: {total_chunks:10,.0f} chunks")


def tokenize_text(text):
    """Worker function for parallel tokenization."""
    return len(tokenizer.encode(text))


def analyze_dataset_parallel(
    dataset_name, dataset, context_field, limit=None, num_workers=None
):
    """Analyze a dataset using parallel tokenization."""
    print("\n" + "=" * 80)
    print(f"DATASET: {dataset_name}")
    print("=" * 80)

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    print(f"Using {num_workers} parallel workers for tokenization.")

    # Phase 1: Extract all texts (sequential - often I/O bound)
    print("\nPhase 1: Extracting texts...")
    texts = set()

    # Determine total for progress bar
    try:
        total_rows = len(dataset) if not limit else min(len(dataset), limit)
        pbar = tqdm(total=total_rows, desc="Extracting", unit="rows")
    except (TypeError, AttributeError):
        if limit:
            pbar = tqdm(total=limit, desc="Extracting", unit="rows")
        else:
            pbar = tqdm(desc="Extracting", unit="rows")

    for i, item in enumerate(dataset):
        if limit and i >= limit:
            print(f"\nReached limit of {limit} items.")
            break

        val = item.get(context_field)
        if val is not None:
            if isinstance(val, str):
                if val.strip():
                    texts.add(val)
            elif isinstance(val, dict):
                if "wiki_context" in val:
                    texts.update(
                        [
                            t
                            for t in val["wiki_context"]
                            if t and isinstance(t, str) and t.strip()
                        ]
                    )
                elif "paragraph" in val:
                    texts.update(
                        [
                            t
                            for t in val["paragraph"]
                            if t and isinstance(t, str) and t.strip()
                        ]
                    )
                # ... other cases
            elif isinstance(val, list):
                texts.update([t for t in val if isinstance(t, str) and t.strip()])

        pbar.update(1)

    pbar.close()
    texts = list(texts)
    print(f"Extracted {len(texts)} text segments from {i + 1} rows.")

    # Phase 2: Parallel tokenization (CPU bound)
    print(f"\nPhase 2: Tokenizing {len(texts)} texts using {num_workers} workers...")

    with Pool(num_workers) as pool:
        token_counts = list(
            tqdm(
                pool.imap(tokenize_text, texts, chunksize=100),
                total=len(texts),
                desc="Tokenizing",
                unit="texts",
            )
        )

    total_tokens = sum(token_counts)

    # Summary
    print("\n" + "-" * 40)
    print("SUMMARY:")
    print(f"  Total texts processed: {len(texts)}")
    print(f"  Total tokens: {total_tokens:,}")

    # Chunk analysis
    print("\nCHUNK ANALYSIS:")
    print("-" * 80)
    for chunk_size in chunk_sizes:
        total_chunks = total_tokens / chunk_size
        print(f"Chunk size {chunk_size:4d} tokens: {total_chunks:10,.0f} chunks")


# # 1. SQuAD
# print("\nLoading SQuAD...")
# squad_dataset = load_dataset("squad", split="train")
# analyze_dataset("SQuAD", squad_dataset, "context")

# # 2. MS MARCO - Passage Ranking
# print("\nLoading MS MARCO Passage Ranking...")
# # Config 'v1.1' is the standard passage ranking config
# msmarco_passage_dataset = load_dataset("ms_marco", "v1.1", split="train")
# analyze_dataset("MS MARCO Passage Ranking", msmarco_passage_dataset, "passages")

# # 3. MS MARCO - Document Ranking
# print("\nLoading MS MARCO Passage Ranking v2.1...")
# # Config 'v2.1' is the updated passage ranking config in the ms_marco dataset
# msmarco_doc_dataset = load_dataset("ms_marco", "v2.1", split="train")
# analyze_dataset("MS MARCO Passage Ranking v2.1", msmarco_doc_dataset, "passages")

# # 4. Natural Questions
# print("\nLoading Natural Questions...")
# # Warning: NQ train split is very large (~40GB+)
# nq_dataset = load_dataset("natural_questions", split="train")
# analyze_dataset("Natural Questions", nq_dataset, "document")

# 5. TriviaQA
print("\nLoading TriviaQA...")
# Config 'rc' is the reading comprehension config
trivia_qa_dataset = load_dataset("trivia_qa", "rc", split="train")
# inspect_dataset_structure("TriviaQA", trivia_qa_dataset, num_samples=2)
# analyze_dataset("TriviaQA", trivia_qa_dataset, "entity_pages")
analyze_dataset_parallel("TriviaQA", trivia_qa_dataset, "entity_pages")


# Get the first row
# row = trivia_qa_dataset[0]

# print(f"Question: {row['question']}")
# print(f"Answer: {row['answer']['value']}")

# # Check Wikipedia Contexts
# if len(row["entity_pages"]["wiki_context"]) > 0:
#     print(f"\nNumber of Wiki Docs: {len(row['entity_pages']['wiki_context'])}")
#     print(
#         f"First Wiki Doc Preview: {row['entity_pages']['wiki_context'][0][:200]}..."
#     )  # First 200 chars
# else:
#     print("\nNo Wikipedia contexts found.")

# # Check Web Contexts
# if len(row["search_results"]["search_context"]) > 0:
#     print(f"\nNumber of Web Docs: {len(row['search_results']['search_context'])}")
#     print(
#         f"First Web Doc Preview: {row['search_results']['search_context'][0][:200]}..."
#     )

# 6. KILT Wikipedia
# print("\nLoading KILT Wikipedia...")
# kilt_dataset = load_dataset("facebook/kilt_wikipedia", split="full", streaming=True, trust_remote_code=True)
# inspect_dataset_structure("KILT Wikipedia", kilt_dataset, num_samples=2)
# # Limit analysis to 1000 items for demonstration/speed as it is streaming and huge
# analyze_dataset("KILT Wikipedia", kilt_dataset, "text", limit=1000)
