# -*- coding: utf-8 -*-
import argparse
from ultralytics import YOLO

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLOv8 Training Script')
    parser.add_argument('--config', required=True, help='Model config file')
    parser.add_argument('--data', required=True, help='Data config file')
    parser.add_argument('--name', default='exp', help='Experiment name')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs')
    parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'Adam', 'AdamW'], help='Optimizer type')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # 加载模型
    model = YOLO(args.config)
    
    base_lr = 0.01 if args.optimizer == 'SGD' else 0.001
    adjusted_lr = base_lr * (args.batch / 64)  # 线性缩放
    
    # 执行训练, sh脚本
    model.train(
        lr0=adjusted_lr,  # 显式设置适配后的学习率
        warmup_epochs=3,  # 学习率预热
        warmup_momentum=0.8,  # 动量预热        
        data=args.data,
        cache=False,
        imgsz=300,
        patience=30,  # 30个epoch无改进则停止
        epochs=args.epochs,
        single_cls=False,
        batch=args.batch,
        close_mosaic=10,
        workers=8,
        optimizer=args.optimizer,
        amp=True,
        project='runs/train_semi',
        name=args.name,
        device=''
    )

    
    # model = YOLO('yolo11s.yaml')
    
    # base_lr = 0.01
    # adjusted_lr = base_lr * (4 / 64)  # 线性缩放
    
    # model.train(
    #     lr0=adjusted_lr,  # 显式设置适配后的学习率
    #     warmup_epochs=3,  # 学习率预热
    #     warmup_momentum=0.8,  # 动量预热        
    #     data="D:/Github/v11_input/my_datasets/example.yaml",
    #     cache=False,
    #     imgsz=300,
    #     patience=30,  # 30个epoch无改进则停止
    #     epochs=100,
    #     single_cls=False,
    #     batch=4,
    #     close_mosaic=10,
    #     workers=0,
    #     optimizer='SGD',
    #     amp=True,
    #     project='runs/train',
    #     name='exp',
    #     device=[0, ]
    # )

