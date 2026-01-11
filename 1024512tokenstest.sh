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
# python test/test_k_mean_datasets.py --dataset trivia_qa --freetest --chunk_size 128 --n_clusters 3500 --k_clusters_list 35 60 90 120 150 175 --k_chunks_list 16 16 16 16 16 16

# echo "Running 256 tokens test"
# python test/test_k_mean_datasets.py --dataset trivia_qa --freetest --chunk_size 256 --n_clusters 2900 --k_clusters_list 30 60 80 105 130 150 --k_chunks_list 8 8 8 8 8 8

# echo "Running 512 tokens test"
# python test/test_k_mean_datasets.py --dataset trivia_qa --freetest --chunk_size 512 --n_clusters 1859 --k_clusters_list 20 40 60 80 90 110 --k_chunks_list 4 4 4 4 4 4

echo "Running 1024 tokens test"
python test/test_k_mean_datasets.py --dataset trivia_qa --freetest --chunk_size 1024 --n_clusters 779 --k_clusters_list 8 15 23 30 35 40 50 60 70 80 90 --k_chunks_list 2 2 2 2 2 2 2 2 2 2 2

echo "Running 512 tokens test"
python test/test_k_mean_datasets.py --dataset trivia_qa --freetest --chunk_size 512 --n_clusters 1859 --k_clusters_list 20 40 60 80 90 110 130 150 170 200 --k_chunks_list 4 4 4 4 4 4 4 4 4 4

# echo "Running 2048 tokens test"
# python test/test_k_mean_datasets.py --dataset trivia_qa --freetest --chunk_size 2048 --n_clusters 492 --k_clusters_list 5 10 15 20 25 30 --k_chunks_list 1 1 1 1 1 1

echo "=== End: $(date) ==="
