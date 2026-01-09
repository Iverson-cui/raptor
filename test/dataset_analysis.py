from datasets import load_dataset
import tiktoken
import json

tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
chunk_sizes = [128, 256, 512, 1024]


def inspect_dataset_structure(dataset_name, dataset, num_samples=1):
    """Inspect the structure of a dataset by showing fields and sample rows."""
    print("\n" + "=" * 80)
    print(f"DATASET STRUCTURE: {dataset_name}")
    print("=" * 80)

    # Get column names
    print(f"\nTotal Rows: {len(dataset)}")
    print(f"\nFields ({len(dataset.column_names)}): {dataset.column_names}")

    # Show sample rows
    print(f"\n--- Sample Row(s) ---")
    for i in range(min(num_samples, len(dataset))):
        print(f"\nRow {i}:")
        row = dataset[i]
        for field_name in dataset.column_names:
            value = row[field_name]
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
    print()


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
            # Handle TriviaQA
            elif "wiki_context" in val:
                for text in val["wiki_context"]:
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
trivia_qa_dataset = load_dataset("trivia_qa", "rc", split="validation")
inspect_dataset_structure("TriviaQA", trivia_qa_dataset, num_samples=2)
# analyze_dataset("TriviaQA", trivia_qa_dataset, "entity_pages")


# Get the first row
row = trivia_qa_dataset[0]

print(f"Question: {row['question']}")
print(f"Answer: {row['answer']['value']}")

# Check Wikipedia Contexts
if len(row["entity_pages"]["wiki_context"]) > 0:
    print(f"\nNumber of Wiki Docs: {len(row['entity_pages']['wiki_context'])}")
    print(
        f"First Wiki Doc Preview: {row['entity_pages']['wiki_context'][0][:200]}..."
    )  # First 200 chars
else:
    print("\nNo Wikipedia contexts found.")

# Check Web Contexts
if len(row["search_results"]["search_context"]) > 0:
    print(f"\nNumber of Web Docs: {len(row['search_results']['search_context'])}")
    print(
        f"First Web Doc Preview: {row['search_results']['search_context'][0][:200]}..."
    )
