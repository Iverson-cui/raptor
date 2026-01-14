
#!/usr/bin/env bash
set -e

# Generate a unique log filename (YYYYMMDD_HHMMSS)
LOG_FILE="run_$(date +'%Y%m%d_%H%M%S').log"

# Duplicate stdout+stderr to terminal and log file
exec > >(tee -a "$LOG_FILE") 2> >(tee -a "$LOG_FILE" >&2)

echo "=== Start: $(date) ==="
echo "Log file: $LOG_FILE"

echo "=== token size: 256 ==="
python test/test_perfect_recall.py --dataset trivia_qa --chunk_size 256 --top_k 16 --num_questions 200
python test/test_perfect_recall.py --dataset trivia_qa --chunk_size 256 --top_k 32 --num_questions 200

echo "=== token size: 512 ==="
python test/test_perfect_recall.py --dataset trivia_qa --chunk_size 512 --top_k 8 --num_questions 200
python test/test_perfect_recall.py --dataset trivia_qa --chunk_size 512 --top_k 16 --num_questions 200

echo "=== token size: 1024 ==="
python test/test_perfect_recall.py --dataset trivia_qa --chunk_size 1024 --top_k 4 --num_questions 200
python test/test_perfect_recall.py --dataset trivia_qa --chunk_size 1024 --top_k 8 --num_questions 200


echo "=== End: $(date) ==="
