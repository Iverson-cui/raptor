import tiktoken
from datasets import load_dataset
from typing import Optional, List
import argparse
from tqdm import tqdm


def analyze_kilt_wikipedia(
    limit: Optional[int] = 1000, chunk_sizes: Optional[List[int]] = None
):
    """
    Loads KILT Wikipedia in streaming mode, inspects a row,
    and analyzes chunk counts for a subset of the data.
    If limit is None, processes the entire corpus.

    Args:
        limit: Number of rows to process. If None, processes all rows.
        chunk_sizes: List of chunk sizes (in tokens) to analyze. Defaults to [256, 512, 1024].
    """
    if chunk_sizes is None:
        chunk_sizes = [256, 512, 1024]

    print("=" * 80)
    print("KILT Wikipedia Analysis")
    print("=" * 80)

    # --- 1. Load Dataset ---
    print("\nAttempting to load 'facebook/kilt_wikipedia' (streaming)...")
    try:
        # Use the Parquet version from the conversion branch and 'train' split.
        # This is necessary to bypass issues with legacy loading scripts in newer
        # versions of the 'datasets' library.
        ds = load_dataset(
            "facebook/kilt_wikipedia",
            revision="refs/convert/parquet",
            split="train",
            streaming=True,
        )
        # ds = load_dataset(
        #     "facebook/kilt_wikipedia",
        #     split="full",
        #     streaming=True,
        #     trust_remote_code=True,
        # )
        print("--> Dataset object created successfully.")
    except Exception as e:
        print(f"--> FAILED to load dataset: {e}")
        print("\nThis might be due to a network issue, a problem with the Hugging Face Hub,")
        print("or the dataset structure having changed. The environment has been verified,")
        print("but this specific dataset is very large and can be slow to initialize.")
        return

    # --- 2. Inspect First Row ---
    print("\nFetching and inspecting the first row...")
    try:
        iterator = iter(ds)
        first_row = next(iterator)
        print("--> First row fetched. Contents:")
        for key, value in first_row.items():
            if isinstance(value, str):
                display_val = (
                    f"'{value[:150]}...'" if len(value) > 150 else f"'{value}'"
                )
            elif isinstance(value, dict):
                dict_str = str(value)
                display_val = (
                    f"[Dict: {dict_str[:200]}...]"
                    if len(dict_str) > 200
                    else f"[Dict: {dict_str}]"
                )
            elif isinstance(value, list):
                display_val = f"[List with {len(value)} items]"
            else:
                display_val = value
            print(f"  - {key}: {display_val}")
    except StopIteration:
        print("--> Dataset appears to be empty.")
        return
    except Exception as e:
        print(f"--> FAILED to fetch or inspect the first row: {e}")
        return

    # --- 3. Analyze Chunk Counts ---
    if limit is None:
        print(f"\nAnalyzing chunk counts for ALL rows (no limit)...")
    else:
        print(f"\nAnalyzing chunk counts for the first {limit} rows...")
    tokenizer = tiktoken.get_encoding("cl100k_base")

    unique_contexts = set()

    # Re-create iterator to start from the beginning for analysis
    iterator = iter(ds)

    # Add progress bar for row processing
    if limit is not None:
        pbar = tqdm(total=limit, desc="Processing rows", unit="rows")
    else:
        pbar = tqdm(desc="Processing rows", unit="rows")

    for i, row in enumerate(iterator):
        if limit is not None and i >= limit:
            break
        # The text content is in 'text' -> 'paragraph'
        if 'text' in row and isinstance(row['text'], dict) and 'paragraph' in row['text']:
            for para in row['text']['paragraph']:
                if para and isinstance(para, str) and para.strip():
                    unique_contexts.add(para)
        pbar.update(1)

    pbar.close()

    print(
        f"--> Processed {i+1} rows and found {len(unique_contexts)} unique paragraphs."
    )

    if not unique_contexts:
        print("--> No text contexts found to analyze.")
        return

    print("\nCalculating token counts...")
    token_counts = []
    for ctx in tqdm(unique_contexts, desc="Tokenizing paragraphs", unit="para"):
        token_counts.append(len(tokenizer.encode(ctx)))
    print("--> Tokenization complete.")

    print("\n" + "-" * 40)
    print("Chunk Analysis Results:")
    print("-" * 40)
    for chunk_size in chunk_sizes:
        total_chunks = sum((tc + chunk_size - 1) // chunk_size for tc in token_counts)
        print(f"Chunk size {chunk_size:4d} tokens: {total_chunks:10d} chunks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze KILT Wikipedia dataset and calculate chunk counts."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of rows to process. If not specified, processes all rows.",
    )
    parser.add_argument(
        "--chunk-sizes",
        type=int,
        nargs="+",
        default=[256, 512, 1024],
        help="List of chunk sizes (in tokens) to analyze. Example: --chunk-sizes 256 512 1024",
    )

    args = parser.parse_args()
    analyze_kilt_wikipedia(limit=args.limit, chunk_sizes=args.chunk_sizes)
