
#!/usr/bin/env bash
set -e

# Generate a unique log filename (YYYYMMDD_HHMMSS)
LOG_FILE="run_$(date +'%Y%m%d_%H%M%S').log"

# Duplicate stdout+stderr to terminal and log file
exec > >(tee -a "$LOG_FILE") 2> >(tee -a "$LOG_FILE" >&2)

echo "=== Start: $(date) ==="
echo "Log file: $LOG_FILE"

python analyse_kilt.py --limit 1000 --chunk-sizes 128 256 512

echo "=== End: $(date) ==="
