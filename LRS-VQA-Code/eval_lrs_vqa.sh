#!/bin/bash

#================================================================
# Script for evaluating on the full LRS-VQA benchmark.
# This script is designed to work with the merged dataset.
#================================================================

# --- User-configurable parameters ---

# Set the number of GPUs to use for parallel processing.
NUM_GPUS=4

# Path to the downloaded model weights.
# For Qwen2 model:
MODEL_PATH="/path/to/your/llava_largeimg_Qwen2_max8192"
# For Vicuna model (example):
# MODEL_PATH="/path/to/your/llava_largeimg_Vicuna_max2048"

# Path to the directory CONTAINING the 'LRS_VQA' folder.
# For example, if your data is at '/data/LRS_VQA', set this to '/data'.
DATA_DIR="/path/to/dataset_parent_directory"

# --- End of user-configurable parameters ---

# Activate Conda environment
source activate lrsvqa
# Note: Ensure 'nltk' is installed. (pip install nltk)

# --- Script logic ---
MODEL_NAME=$(basename "$MODEL_PATH")

# Based on your jsonl structure, the image path is "LRS_VQA/image/...",
# so the image_folder should be the parent directory of "LRS_VQA".
QUESTION_FILE="$DATA_DIR/LRS_VQA/LRS_VQA_merged.jsonl"
IMAGE_FOLDER="$DATA_DIR"

# Check if the question file exists
if [ ! -f "$QUESTION_FILE" ]; then
    echo "Error: Question file not found at $QUESTION_FILE"
    echo "Please ensure DATA_DIR is set to the parent directory of your 'LRS_VQA' folder."
    exit 1
fi

# Define output directories
BASE_ANSWER_DIR="./outputs/lrs_vqa/answers_temp/${MODEL_NAME}"
MERGED_ANSWER_FILE="./outputs/lrs_vqa/answers/${MODEL_NAME}.jsonl"
LOG_DIR="./outputs/lrs_vqa/logs/${MODEL_NAME}"

mkdir -p "$BASE_ANSWER_DIR"
mkdir -p "$(dirname "$MERGED_ANSWER_FILE")"
mkdir -p "$LOG_DIR"


echo "==================================================="
echo "Starting evaluation on the LRS-VQA benchmark..."
echo "Model: $MODEL_NAME"
echo "GPUs used: $NUM_GPUS"
echo "==================================================="

# Run inference in parallel across multiple GPUs
for (( CHUNK_ID=0; CHUNK_ID<$NUM_GPUS; CHUNK_ID++ )); do
    ANSWER_PATH="${BASE_ANSWER_DIR}/result_chunk_${CHUNK_ID}.jsonl"
    
    # Remove old temp file if it exists
    if [ -f "$ANSWER_PATH" ]; then
        rm "$ANSWER_PATH"
    fi

    CUDA_VISIBLE_DEVICES=$CHUNK_ID python -u llava/eval/model_rsvqa.py \
        --model-path "$MODEL_PATH" \
        --question-file "$QUESTION_FILE" \
        --image-folder "$IMAGE_FOLDER" \
        --answers-file "$ANSWER_PATH" \
        --temperature 0 \
        --max_block_num 1 \
        --conv-mode vicuna_v1 \
        --pyramid_crop True \
        --num-chunks "$NUM_GPUS" \
        --chunk-idx "$CHUNK_ID" > "$LOG_DIR/log_chunk_${CHUNK_ID}.txt" 2>&1 &

done

# Wait for all background processes to finish
wait

echo "Inference finished. Merging answer files..."

# Merge results from all chunks
# Ensure the merged file is empty before appending
> "$MERGED_ANSWER_FILE"
for (( CHUNK_ID=0; CHUNK_ID<$NUM_GPUS; CHUNK_ID++ )); do
    cat "${BASE_ANSWER_DIR}/result_chunk_${CHUNK_ID}.jsonl" >> "$MERGED_ANSWER_FILE"
done

# (Optional) Clean up temporary chunk files
# rm -rf "$BASE_ANSWER_DIR"

echo "Merging complete. Running final evaluation script..."

# Run the final evaluation
python llava/eval/model_rsvqa_eval.py \
    --results_file "$MERGED_ANSWER_FILE" | tee "$LOG_DIR/evaluation_summary.txt"

echo "==================================================="
echo "Evaluation complete. Results are saved in:"
echo "Answers: $MERGED_ANSWER_FILE"
echo "Logs: $LOG_DIR"
echo "==================================================="
