#!/bin/bash
# 注意：CUDA_VISIBLE_DEVICES 的 export 已经被移到了参数解析之后！
export WANDB_DISABLED=true       # <--- 强制禁用 WandB
export WANDB_MODE=offline        # <--- 强制 WandB 进入离线模式
export YOLOV8_NO_ULTRALYTICS_TELEMETRY=1  # <--- 关闭 YOLO 官方的数据收集（防卡死）

# 训练脚本路径
TRAIN_SCRIPT="train_group.py"
# 日志目录
LOG_DIR="training_logs"
# 开始时间
START_TIME=$(date +%s)

#################### 参数解析区 ####################
# 默认参数
BATCH_SIZE=128
PROJECT_DIR='runs/train_fly'
DATASET_BASE_DIR=""
EXP_PREFIX=""
CUDA_DEVICE="0"  # <--- 新增：默认使用 0 号显卡

# 解析命令行传入的参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset_dir) DATASET_BASE_DIR="$2"; shift ;;
        --batch) BATCH_SIZE="$2"; shift ;;
        --prefix) EXP_PREFIX="$2"; shift ;;
        --project) PROJECT_DIR="$2"; shift ;;
        --device) CUDA_DEVICE="$2"; shift ;;  # <--- 新增：解析 device 参数
        *) echo "❌ 未知参数: $1"; exit 1 ;;
    esac
    shift
done

# 校验必填参数
if [ -z "$DATASET_BASE_DIR" ] || [ -z "$EXP_PREFIX" ]; then
    echo "❌ 错误: 必须提供 --dataset_dir 和 --prefix 参数！"
    echo "💡 用法示例: bash $0 --dataset_dir /data/ZS/.../paint_flywheel2 --prefix iter3_4 --batch 128 --device 3"
    exit 1
fi

# ==========================================
# 核心修改：在这里导出 CUDA 环境变量！
# ==========================================
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
echo "🔥 指定显卡: CUDA_VISIBLE_DEVICES=$CUDA_DEVICE"
################################################

# 创建日志目录
mkdir -p $LOG_DIR
echo "创建日志目录: $LOG_DIR"

# 初始化任务计数器
TOTAL_TASKS=2
COMPLETED_TASKS=0
FAILED_TASKS=0
TASK_COUNTER=0

# 打印状态摘要函数
print_summary() {
    local success=$1
    local task_name=$2
    local log_file=$3
    
    TASK_COUNTER=$((TASK_COUNTER+1))
    if [ $success -eq 1 ]; then
        COMPLETED_TASKS=$((COMPLETED_TASKS+1))
        echo -e "\n✅ 任务成功完成: ${TASK_COUNTER}/${TOTAL_TASKS} $task_name"
    else
        FAILED_TASKS=$((FAILED_TASKS+1))
        echo -e "\n❌ 任务失败: ${TASK_COUNTER}/${TOTAL_TASKS} $task_name"
        echo "查看日志: $log_file"
    fi
}

# 执行单个训练任务函数
run_training_task() {
    local config=$1
    local data=$2
    local batch_size=${3:-$BATCH_SIZE}
    local name=$4 
    local model=${config%.*}
    local dataset=${data##*/}; dataset=${dataset%.*}
    local task_name="${model}_${dataset}"
    local log_file="${LOG_DIR}/${task_name}_$(date +%Y%m%d-%H%M%S).log"
    
    echo -e "\n================================================================="
    echo "🚀 开始训练任务 [$((TASK_COUNTER+1))/${TOTAL_TASKS}]：模型 $model 在数据集 $dataset"
    echo "📁 配置文件: $config"
    echo "📊 数据集: $data"
    echo "📦 保存路径: $PROJECT_DIR"
    echo "🏷️  实验命名: $name"
    echo "📝 日志文件: $log_file"
    echo "⏱️ 开始时间: $(date)"
    echo "================================================================="
    
    # 开始任务并记录日志
    start_time=$(date +%s)
    python3 $TRAIN_SCRIPT --batch $batch_size --data "$data" --config "$config" --project "$PROJECT_DIR" --name "$name" 2>&1 | tee "$log_file"
    task_status=${PIPESTATUS[0]}  # 获取实际命令的退出状态
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    # 格式化时间
    hours=$((duration / 3600))
    minutes=$(( (duration % 3600) / 60 ))
    seconds=$((duration % 60))
    
    # 打印状态
    if [ $task_status -eq 0 ]; then
        print_summary 1 "$task_name" "$log_file"
        echo "⏱️ 训练耗时: ${hours}时${minutes}分${seconds}秒"
    else
        print_summary 0 "$task_name" "$log_file"
        echo "⏱️ 失败时间: ${hours}时${minutes}分${seconds}秒"
    fi
    
    # 任务间等待GPU释放
    sleep 30
}

# 打印总体信息
echo "=============================================================="
echo "✨ 开始执行所有训练任务"
echo "📅 开始时间: $(date -d @$START_TIME)"
echo "📂 训练脚本: $TRAIN_SCRIPT"
echo "📊 任务总数: $TOTAL_TASKS"
echo "📁 日志目录: $LOG_DIR"
echo "=============================================================="

# yolo11s 模型系列训练
echo -e "\n================================================================="
echo "🌟🌟🌟 开始 yolo11s 模型系列训练 🌟🌟🌟"
echo "================================================================="

# 启动训练任务 (动态拼接路径和名称)
run_training_task "yolo11s.yaml" "${DATASET_BASE_DIR}/col3.yaml" $BATCH_SIZE "${EXP_PREFIX}_col3"
run_training_task "yolo11s.yaml" "${DATASET_BASE_DIR}/row3.yaml" $BATCH_SIZE "${EXP_PREFIX}_row3"

# 计算总耗时
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(( (TOTAL_DURATION % 3600) / 60 ))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

# 打印最终总结
echo -e "\n=============================================================="
echo "🎉 所有训练任务已完成!"
echo "📅 开始时间: $(date -d @$START_TIME)"
echo "📅 结束时间: $(date -d @$END_TIME)"
echo "⏱️ 总耗时: ${TOTAL_HOURS}时${TOTAL_MINUTES}分${TOTAL_SECONDS}秒"
echo "✅ 成功任务: $COMPLETED_TASKS"
echo "❌ 失败任务: $FAILED_TASKS"
echo "📁 日志目录: $LOG_DIR"
echo "=============================================================="

# 如果有失败任务，建议检查日志
if [ $FAILED_TASKS -gt 0 ]; then
    echo -e "\n⚠️ 注意: 共有 $FAILED_TASKS 个任务失败，请检查相应日志文件!"
    echo "使用以下命令查看日志:"
    echo "  grep '错误原因' $LOG_DIR/*.log"
    exit 1
else
    exit 0
fi