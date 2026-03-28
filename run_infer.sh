#!/bin/bash

# ==============================================================================
# 进行推理，支持并行推理
# ==============================================================================

# 配置 Python 解释器路径 (根据你的 conda 环境修改)
PYTHON_EXEC="python"
SCRIPT_PATH="/data/ZS/v11_input/inference.py"

# 配置基础路径
WEIGHTS_BASE="/data/ZS/v11_input/weights"
DATA_BASE="/data/ZS/flywheel_dataset/0_multi_input/sp123"
OUT_BASE="/data/ZS/flywheel_dataset/2_yolo_preds/iter0_0p1"     # 后续跑数据迭代修改这里
OUT_JSON1="sp012_col3.json"
OUT_JSON2="sp012_row3.json"

# 创建输出和日志目录
mkdir -p ${OUT_BASE}
mkdir -p logs

echo "🚀 开始启动无标签数据推理任务..."

# ==============================================================================
# 任务 1: COL3 (使用 0 号显卡)
# ==============================================================================
echo "启动 COL3 模型推理 (GPU 0) -> 后台运行中..."
CUDA_VISIBLE_DEVICES=0 ${PYTHON_EXEC} ${SCRIPT_PATH} \
    --model_path "${WEIGHTS_BASE}/col3_part_max_cbam_iter0.pt" \
    --input_dir "${DATA_BASE}/col3" \
    --output_json "${OUT_BASE}/${OUT_JSON1}" \
    --conf_thres 0.1 \
    > logs/infer_col3.log 2>&1 &  # 放入后台运行，日志重定向

# ==============================================================================
# 任务 2: ROW3 (使用 1 号显卡)
# ==============================================================================
echo "启动 ROW3 模型推理 (GPU 1) -> 后台运行中..."
CUDA_VISIBLE_DEVICES=1 ${PYTHON_EXEC} ${SCRIPT_PATH} \
    --model_path "${WEIGHTS_BASE}/row3_part_max_cbam_iter0.pt" \
    --input_dir "${DATA_BASE}/row3" \
    --output_json "${OUT_BASE}/${OUT_JSON2}" \
    --conf_thres 0.1 \
    > logs/infer_row3.log 2>&1 &  # 放入后台运行，日志重定向

# 等待所有后台任务完成
echo "⏳ 正在等待两张显卡推理完成，请稍候... (可以使用 'tail -f logs/infer_col3.log' 查看进度)"
wait

echo "🎉 恭喜！COL3 和 ROW3 的无标签数据推理全部完成！"