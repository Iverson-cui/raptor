#!/usr/bin/env bash
set -e

# Generate a unique log filename (YYYYMMDD_HHMMSS)
LOG_FILE="run_$(date +'%Y%m%d_%H%M%S').log"

# Duplicate stdout+stderr to terminal and log file
exec > >(tee -a "$LOG_FILE") 2> >(tee -a "$LOG_FILE" >&2)

echo "=== Start: $(date) ==="
echo "Log file: $LOG_FILE"

# python test/test_k_mean_datasets.py --freetest --chunk_size 256 --n_clusters 400 --k_clusters_list 3 5 10 --k_chunks_list 10 10 10

# echo "Running 128 tokens test"
# python test/test_k_mean_datasets.py --dataset trivia_qa --freetest --node_info --chunk_size 128 --n_clusters 3500 --k_clusters_list 90 120 150 175 --k_chunks_list 32 32 32 32

# echo "Running 256 tokens test"
# python test/test_k_mean_datasets.py --dataset trivia_qa --freetest --node_info --chunk_size 256 --n_clusters 2900 --k_clusters_list 80 105 130 150 --k_chunks_list 16 16 16 16

# echo "Running 512 tokens test"
# python test/test_k_mean_datasets.py --dataset trivia_qa --freetest --node_info --chunk_size 512 --n_clusters 1859 --k_clusters_list 40 80 150 180 --k_chunks_list 8 8 8 8

# echo "Running 1024 tokens test"
# python test/test_k_mean_datasets.py --dataset trivia_qa --freetest --node_info --chunk_size 1024 --n_clusters 779 --k_clusters_list 20 40 60 80 --k_chunks_list 4 4 4 4

# # half dataset to reduce time
# echo "Running 1024 tokens test(half dataset)"
# python test/test_k_mean_datasets.py --dataset trivia_qa --freetest --context_ratio 0.5 --chunk_size 1024 --n_clusters 390 --k_clusters_list 4 7 12 15 18 20 25 30 35 40 45 --k_chunks_list 2 2 2 2 2 2 2 2 2 2 2

# echo "Running 512 tokens test"
# python test/test_k_mean_datasets.py --dataset trivia_qa --freetest --node_info --chunk_size 512 --n_clusters 1859 --k_clusters_list 40 80 150 180 --k_chunks_list 4 4 4 4

# echo "Running 2048 tokens test"
# python test/test_k_mean_datasets.py --dataset trivia_qa --freetest --node_info --chunk_size 2048 --n_clusters 492 --k_clusters_list 10 20 30 40 --k_chunks_list 2 2 2 2
# python test/test_k_mean_datasets.py --dataset trivia_qa --freetest --node_info --chunk_size 2048 --n_clusters 5 --k_clusters_list 3 4 --k_chunks_list 2 2

echo "Running 2048 tokens test"
python test/chunk_merge_distance.py --server --chunk_size=2048 --n_clusters 492 --top_k_clusters 25 --top_k_chunks 50 --save_tree distance_2048tokens.pkl

echo "Running 1024 tokens test"
python test/chunk_merge_distance.py --server --chunk_size=1024 --n_clusters 780 --top_k_clusters 35 --top_k_chunks 50 --save_tree distance_1024tokens.pkl

echo "Running 512 tokens test"
python test/chunk_merge_distance.py --server --chunk_size=512 --n_clusters 1860 --top_k_clusters 90 --top_k_chunks 50 --save_tree distance_512tokens.pkl

echo "=== End: $(date) ==="


