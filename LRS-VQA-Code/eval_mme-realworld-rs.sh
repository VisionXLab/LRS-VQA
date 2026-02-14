#!/bin/bash

#================================================================
# Script for evaluating on the MME-RealWorld benchmark.
#================================================================

# Set the number of GPUs to use for parallel processing.
NUM_GPUS=4

# Path to the downloaded model weights.
# For Qwen2 model:
MODEL_PATH="/path/to/your/llava_largeimg_Qwen2_max8192"
# For Vicuna model (example):
# MODEL_PATH="/path/to/your/llava_largeimg_Vicuna_max2048"

# Path to the MME-RealWorld dataset directory.
# This directory should contain 'MME_RealWorld_RS.json' and the 'remote_sensing/' image folder.
DATA_DIR="/path/to/your/MME_RealWorld"


# Activate Conda environment
source activate lrsvqa

# --- Script logic ---
MODEL_NAME=$(basename "$MODEL_PATH")
QUESTION_FILE="$DATA_DIR/MME_RealWorld_RS.json"
IMAGE_FOLDER="$DATA_DIR"
EVAL_SCRIPT="$DATA_DIR/evaluation/eval_your_results.py"

# Define output directories
BASE_ANSWER_DIR="./outputs/mme_realworld/answers_temp/${MODEL_NAME}"
MERGED_ANSWER_FILE="./outputs/mme_realworld/answers/${MODEL_NAME}.json"
LOG_DIR="./outputs/mme_realworld/logs/${MODEL_NAME}"

mkdir -p "$BASE_ANSWER_DIR"
mkdir -p "$(dirname "$MERGED_ANSWER_FILE")"
mkdir -p "$LOG_DIR"


echo "==================================================="
echo "Starting evaluation on MME-RealWorld..."
echo "Model: $MODEL_NAME"
echo "GPUs used: $NUM_GPUS"
echo "==================================================="

# Run inference in parallel across multiple GPUs
for (( CHUNK_ID=0; CHUNK_ID<$NUM_GPUS; CHUNK_ID++ )); do
    ANSWER_PATH="${BASE_ANSWER_DIR}/result_chunk_${CHUNK_ID}.json"
    
    # Remove old temp file if it exists
    if [ -f "$ANSWER_PATH" ]; then
        rm "$ANSWER_PATH"
    fi

    CUDA_VISIBLE_DEVICES=$CHUNK_ID python -u llava/eval/model_vqa_mme_realworld.py \
        --model-path "$MODEL_PATH" \
        --question-file "$QUESTION_FILE" \
        --image-folder "$IMAGE_FOLDER" \
        --answers-file "$ANSWER_PATH" \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --num-chunks "$NUM_GPUS" \
        --pyramid_crop True \
        --chunk-idx "$CHUNK_ID" > "$LOG_DIR/log_chunk_${CHUNK_ID}.txt" 2>&1 &

done

# Wait for all background processes to finish
wait

echo "Inference finished. Merging answer files..."

# Merge results from all chunks
# Ensure the merged file is empty before appending
> "$MERGED_ANSWER_FILE"
for (( CHUNK_ID=0; CHUNK_ID<$NUM_GPUS; CHUNK_ID++ )); do
    cat "${BASE_ANSWER_DIR}/result_chunk_${CHUNK_ID}.json" >> "$MERGED_ANSWER_FILE"
done

# (Optional) Clean up temporary chunk files
# rm -rf "$BASE_ANSWER_DIR"

echo "Merging complete. Running final evaluation script..."

# Run the final evaluation
python "$EVAL_SCRIPT" \
    --results_file "$MERGED_ANSWER_FILE" | tee "$LOG_DIR/evaluation_summary.txt"

echo "==================================================="
echo "Evaluation complete. Results are saved in:"
echo "Answers: $MERGED_ANSWER_FILE"
echo "Logs: $LOG_DIR"
echo "==================================================="
