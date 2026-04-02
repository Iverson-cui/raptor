#!/usr/bin/env bash
set -e

# Generate a unique log filename (YYYYMMDD_HHMMSS)
LOG_FILE="run_$(date +'%Y%m%d_%H%M%S').log"

# Duplicate stdout+stderr to terminal and log file
exec > >(tee -a "$LOG_FILE") 2> >(tee -a "$LOG_FILE" >&2)

echo "=== Start: $(date) ==="
echo "Log file: $LOG_FILE"

# echo "=== token size: 256 ==="
# python test/test_perfect_recall.py --dataset trivia_qa --chunk_size 256 --top_k 16 --num_questions 200
# python test/test_perfect_recall.py --dataset trivia_qa --chunk_size 256 --top_k 32 --num_questions 200

# echo "=== token size: 512 ==="
# python test/test_perfect_recall.py --dataset trivia_qa --chunk_size 512 --top_k 8 --num_questions 200
# python test/test_perfect_recall.py --dataset trivia_qa --chunk_size 512 --top_k 16 --num_questions 200

# echo "=== token size: 1024 ==="
# python test/test_perfect_recall.py --dataset trivia_qa --chunk_size 1024 --top_k 4 --num_questions 200
# python test/test_perfect_recall.py --dataset trivia_qa --chunk_size 1024 --top_k 8 --num_questions 200

# python test/merge_distance\&tree_exam.py --examine_tree_structure --load_tree pkl_files/squad_128.pkl
python test/merge_distance\&tree_exam.py --migrate_tree pkl_files/squad_64.pkl
# python test/merge_distance\&tree_exam.py --examine_tree_structure --load_tree pkl_files/squad_128.pkl
python test/merge_distance\&tree_exam.py --update_index_tree --source_tree pkl_files/squad_64m128.pkl --target_tree pkl_files/squad_64.pkl
python test/merge_distance\&tree_exam.py --examine_tree_structure --load_tree pkl_files/squad_64.pkl

python test/merge_distance\&tree_exam.py --migrate_tree pkl_files/triviaQA_256tokens.pkl
# python test/merge_distance\&tree_exam.py --examine_tree_structure --load_tree pkl_files/squad_128.pkl
python test/merge_distance\&tree_exam.py --update_index_tree --source_tree pkl_files/triviaQA_256m512.pkl --target_tree pkl_files/triviaQA_256tokens.pkl
python test/merge_distance\&tree_exam.py --examine_tree_structure --load_tree pkl_files/triviaQA_256tokens.pkl

python test/merge_distance\&tree_exam.py --migrate_tree pkl_files/triviaQA_512tokens.pkl
# python test/merge_distance\&tree_exam.py --examine_tree_structure --load_tree pkl_files/squad_128.pkl
python test/merge_distance\&tree_exam.py --update_index_tree --source_tree pkl_files/triviaQA_512m1024.pkl --target_tree pkl_files/triviaQA_512tokens.pkl
python test/merge_distance\&tree_exam.py --examine_tree_structure --load_tree pkl_files/triviaQA_512tokens.pkl

echo "=== End: $(date) ==="
