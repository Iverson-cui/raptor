#!/usr/bin/env bash
set -e

# Generate a unique log filename (YYYYMMDD_HHMMSS)
LOG_FILE="run_$(date +'%Y%m%d_%H%M%S').log"

# Redirect all stdout and stderr to the log file
exec > "$LOG_FILE" 2>&1

echo "=== Start: $(date) ==="
echo "Log file: $LOG_FILE"

echo "Running 128 tokens test"
python test/test_k_mean_datasets.py --dataset trivia_qa --freetest --chunk_size 128 --n_clusters 3500 --top_k_clusters 35 60 90 120 150 175 --top_k 16 16 16 16 16 16

echo "Running 256 tokens test"
python test/test_k_mean_datasets.py --dataset trivia_qa --freetest --chunk_size 256 --n_clusters 2900 --top_k_clusters 30 60 80 105 130 150 --top_k 8 8 8 8 8 8

echo "Running 512 tokens test"
python test/test_k_mean_datasets.py --dataset trivia_qa --freetest --chunk_size 512 --n_clusters 1859 --top_k_clusters 20 40 60 80 90 110 --top_k 4 4 4 4 4 4

echo "Running 1024 tokens test"
python test/test_k_mean_datasets.py --dataset trivia_qa --freetest --chunk_size 1024 --n_clusters 779 --top_k_clusters 8 15 23 30 35 40 --top_k 2 2 2 2 2 2

echo "Running 2048 tokens test"
python test/test_k_mean_datasets.py --dataset trivia_qa --freetest --chunk_size 2048 --n_clusters 492 --top_k_clusters 5 10 15 20 25 30 --top_k 1 1 1 1 1 1

echo "=== End: $(date) ==="
