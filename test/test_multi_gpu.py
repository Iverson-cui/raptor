import time
from sentence_transformers import SentenceTransformer


def main():
    # --- Configuration ---
    MODEL_NAME = "nomic-ai/modernbert-embed-base"

    # Define exactly which GPUs you want to use for the "Heavy Lifting"
    # Example: Using GPU 3, 4, 5, and 6
    TARGET_GPUS = ["cuda:3", "cuda:4", "cuda:5", "cuda:6"]

    # Create a dummy dataset (10,000 documents)
    print(f"1. Generating dummy data...")
    documents = [
        f"search_document: This is document number {i} with some content."
        for i in range(10000)
    ]

    # --- Step 1: Load the "Manager" Model ---
    # We load it on the first GPU in our list (cuda:3) to act as the controller.
    print(f"2. Loading Master Model on {TARGET_GPUS[0]}...")
    model = SentenceTransformer(
        MODEL_NAME, trust_remote_code=True, device=TARGET_GPUS[0]
    )

    # --- Step 2: Start the Worker Pool ---
    print(f"3. Spinning up workers on: {TARGET_GPUS}...")
    try:
        pool = model.start_multi_process_pool(target_devices=TARGET_GPUS)
    except Exception as e:
        print(f"❌ Error starting pool: {e}")
        return

    # --- Step 3: Embed (The Magic Part) ---
    print(f"4. Encoding {len(documents)} documents across {len(TARGET_GPUS)} GPUs...")
    start_time = time.time()

    embeddings = model.encode(
        documents,
        pool=pool,  # <--- PASS THE POOL HERE
        batch_size=128,  # 128 per GPU * 4 GPUs = 512 total batch
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    end_time = time.time()

    # --- Step 4: Cleanup ---
    print("5. Stopping worker pool...")
    model.stop_multi_process_pool(pool)

    # --- Results ---
    elapsed = end_time - start_time
    print(f"\n✅ Done!")
    print(f"Output Shape: {embeddings.shape}")
    print(f"Total Time:   {elapsed:.2f} seconds")
    print(f"Throughput:   {len(documents) / elapsed:.0f} docs/sec")


if __name__ == "__main__":
    # IMPORTANT: You must wrap multi-process code in 'if __name__ == "__main__":'
    main()
