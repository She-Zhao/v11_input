#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
export WANDB_DISABLED=true       # <--- 强制禁用 WandB
export WANDB_MODE=offline        # <--- 强制 WandB 进入离线模式
export YOLOV8_NO_ULTRALYTICS_TELEMETRY=1  # <--- 关闭 YOLO 官方的数据收集（防卡死）
# 训练脚本路径
TRAIN_SCRIPT="train_group.py"
# 日志目录
LOG_DIR="training_logs"
# 开始时间
START_TIME=$(date +%s)

# 创建日志目录
mkdir -p $LOG_DIR
echo "创建日志目录: $LOG_DIR"

# 初始化任务计数器
TOTAL_TASKS=12
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
    local model=${config%.*}
    local dataset=${data##*/}; dataset=${dataset%.*}
    local task_name="${model}_${dataset}"
    local log_file="${LOG_DIR}/${task_name}_$(date +%Y%m%d-%H%M%S).log"
    
    echo -e "\n================================================================="
    echo "🚀 开始训练任务 [$((TASK_COUNTER+1))/${TOTAL_TASKS}]：模型 $model 在数据集 $dataset"
    echo "📁 配置文件: $config"
    echo "📊 数据集: $data"
    echo "📝 日志文件: $log_file"
    echo "⏱️ 开始时间: $(date)"
    echo "================================================================="
    
    # 开始任务并记录日志
    start_time=$(date +%s)
    python3 $TRAIN_SCRIPT --data "$data" --config "$config" 2>&1 | tee "$log_file"
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

run_training_task "yolo11s.yaml" "/data/ZS/v11_input/ultralytics/cfg/datasets/paint/col3.yaml"
run_training_task "yolo11s.yaml" "/data/ZS/v11_input/ultralytics/cfg/datasets/paint/row3.yaml"


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