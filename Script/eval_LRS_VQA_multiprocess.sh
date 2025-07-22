

MODEL_PATH='your_model_path'
MODEL_NAME='your_model_name'

LOG_DIR='your_log_dir'

mkdir -p "$LOG_DIR/${MODEL_NAME}"
# 设置并行进程数（根据可用的GPU数量进行调整）
NUM_PROCESSES=4

# 定义评估数据集
DATASETS=(
    "LRS_VQA"
)

for DATASET in "${DATASETS[@]}"; do
    # 设置答案文件的基础路径
    BASE_ANSWER_DIR="your_answer_output_path/${MODEL_NAME}_${DATASET}"
    mkdir -p "$BASE_ANSWER_DIR"

    # 启动多个进程
    for (( CHUNK_ID=0; CHUNK_ID<$NUM_PROCESSES; CHUNK_ID++ )); do
        # 设置每个进程的答案文件路径
        ANSWER_PATH="${BASE_ANSWER_DIR}/result_${CHUNK_ID}.jsonl"

        # 如果存在旧的答案文件，删除它
        if [ -f "$ANSWER_PATH" ]; then
            rm "$ANSWER_PATH"
        fi

        # 启动 Python 推理脚本，指定对应的 GPU 和数据块
        CUDA_VISIBLE_DEVICES="$CHUNK_ID" python llava_eval_LRSVQA.py \
            --model-path "$MODEL_PATH" \
            --question-file "./data/LRS_VQA_merged.jsonl" \
            --image-folder "path_of_LRS-VQA_image" \
            --answers-file "$ANSWER_PATH" \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --num-chunks "$NUM_PROCESSES" \
            --chunk-idx "$CHUNK_ID" >> "$LOG_DIR/${MODEL_NAME}/${DATASET}_output_chunk${CHUNK_ID}.txt" 2>&1 &
    done

    # 等待所有后台进程完成
    wait

    # 合并所有答案文件
    MERGED_ANSWER_FILE="./merge_answer_output_path/${MODEL_NAME}_${DATASET}_merged.jsonl"

    # 如果已存在合并的答案文件，删除它
    if [ -f "$MERGED_ANSWER_FILE" ]; then
        rm "$MERGED_ANSWER_FILE"
    fi

    # 合并所有分块的答案文件
    for (( CHUNK_ID=0; CHUNK_ID<$NUM_PROCESSES; CHUNK_ID++ )); do
        ANSWER_PATH="${BASE_ANSWER_DIR}/result_${CHUNK_ID}.jsonl"
        cat "$ANSWER_PATH" >> "$MERGED_ANSWER_FILE"
    done

    # （可选）删除分块的答案文件
    # rm -r "$BASE_ANSWER_DIR"

    # 运行评估脚本，并将输出追加到日志文件
    python evaluation_LRSVQA.py \
        --results_file "$MERGED_ANSWER_FILE" >> "$LOG_DIR/${MODEL_NAME}/${DATASET}_evaluation_merged.txt" 2>&1
done
