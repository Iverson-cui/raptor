from datasets import load_dataset
import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
chunk_sizes = [128, 256, 512, 1024]


def analyze_dataset(dataset_name, dataset, context_field):
    """Analyze a dataset and print chunk statistics."""
    print("\n" + "=" * 80)
    print(f"DATASET: {dataset_name}")
    print("=" * 80)

    # Get unique contexts
    unique_contexts = set()
    print("Extracting unique contexts...")
    
    # Use the column directly if possible for speed, but iterate for dicts/lists
    for item in dataset:
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
        elif isinstance(val, list):
            for sub_val in val:
                if isinstance(sub_val, str) and sub_val.strip():
                    unique_contexts.add(sub_val)

    print(f"Total Rows: {len(dataset)}")
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

    # Optional: Print first few contexts for verification (commented out)
    # print("\nSAMPLE CONTEXTS (First 3):")
    # for i, context in enumerate(list(unique_contexts)[:3]):
    #     print(f"\n--- Context {i+1} ---")
    #     print(context[:200] + "..." if len(context) > 200 else context)


# 1. SQuAD
print("\nLoading SQuAD...")
squad_dataset = load_dataset("squad", split="train")
analyze_dataset("SQuAD", squad_dataset, "context")

# 2. MS MARCO - Passage Ranking
print("\nLoading MS MARCO Passage Ranking...")
# Config 'v1.1' is the standard passage ranking config
msmarco_passage_dataset = load_dataset("ms_marco", "v1.1", split="train")
analyze_dataset("MS MARCO Passage Ranking", msmarco_passage_dataset, "passages")

# 3. MS MARCO - Document Ranking
print("\nLoading MS MARCO Passage Ranking v2.1...")
# Config 'v2.1' is the updated passage ranking config in the ms_marco dataset
msmarco_doc_dataset = load_dataset("ms_marco", "v2.1", split="train")
analyze_dataset("MS MARCO Passage Ranking v2.1", msmarco_doc_dataset, "passages")

# 4. Natural Questions
print("\nLoading Natural Questions...")
# Warning: NQ train split is very large (~40GB+)
nq_dataset = load_dataset("natural_questions", split="train")
analyze_dataset("Natural Questions", nq_dataset, "document")