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
    unique_contexts = set(dataset[context_field])
    print(f"Total Rows: {len(dataset)}")
    print(f"Unique Contexts: {len(unique_contexts)}")

    # Chunk analysis
    print("\nCHUNK ANALYSIS:")
    print("-" * 80)
    for chunk_size in chunk_sizes:
        total_chunks = 0
        for context in unique_contexts:
            tokens = tokenizer.encode(context)
            num_chunks = (len(tokens) + chunk_size - 1) // chunk_size
            total_chunks += num_chunks
        print(f"Chunk size {chunk_size:4d} tokens: {total_chunks:6d} chunks")


# 1. SQuAD
squad_dataset = load_dataset("squad", split="train")
analyze_dataset("SQuAD", squad_dataset, "context")

# 2. MS MARCO - Passage Ranking
print("\nLoading MS MARCO Passage Ranking...")
msmarco_passage_dataset = load_dataset("ms_marco", "v1.1", split="train")
analyze_dataset("MS MARCO Passage Ranking", msmarco_passage_dataset, "passages")

# 3. MS MARCO - Document Ranking
print("\nLoading MS MARCO Document Ranking...")
msmarco_doc_dataset = load_dataset("ms_marco", "v2.1", split="train")
analyze_dataset("MS MARCO Document Ranking", msmarco_doc_dataset, "passages")

# 4. Natural Questions
print("\nLoading Natural Questions...")
nq_dataset = load_dataset("natural_questions", split="train")
analyze_dataset("Natural Questions", nq_dataset, "document")

# # 4. Print first 10 contexts
# print("\n" + "=" * 80)
# print("FIRST 10 CONTEXTS:")
# print("=" * 80)
# for i, context in enumerate(list(unique_contexts)[:10]):
#     print(f"\n--- Context {i+1} ---")
#     print(context[:200] + "..." if len(context) > 200 else context)

# # 5. Print first 10 questions
# print("\n" + "=" * 80)
# print("FIRST 10 QUESTIONS:")
# print("=" * 80)
# for i in range(10):
#     print(f"\n--- Question {i+1} ---")
#     print(f"Q: {dataset[i]['question']}")
#     print(f"A: {dataset[i]['answers']['text'][0]}")
#     print(f"Context (first 100 chars): {dataset[i]['context'][:]}...")
