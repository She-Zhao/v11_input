import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../../'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from ultralytics import YOLO

def test_custom_yolo():
    # 替换为你实际的魔改 yaml 路径
    yaml_path = '/data/ZS/v11_input/ultralytics/cfg/models/11/yolo11s.yaml'
    
    print(f"正在构建模型: {yaml_path}")
    model = YOLO(yaml_path)
    
    # 强制指定 imgsz=300，计算出的 FLOPs 才是你真实训练和推理的准确数据！
    # detailed=False 即可，我们只需要看最后一行总计
    model.model.info(detailed=False, imgsz=300)

if __name__ == '__main__':
    test_custom_yolo()