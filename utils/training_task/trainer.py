'''
Author: Will Cheng (will.cheng@efctw.com)
Date: 2024-08-07 13:33:01
LastEditors: Will Cheng (will.cheng@efctw.com)
LastEditTime: 2024-08-09 16:22:22
FilePath: /PoseidonAI-Server/utils/training_task/trainer.py
'''
from .trainer_yolov8_detection import start_training_yolo_detection_task, read_yolo_loss_values # 引入 YOLOv8 訓練任務的函數

# 定義訓練器選擇函數
def get_trainer(algo_name: str, framework_name: str):
    """
    根據算法名稱和框架名稱返回對應的訓練器函數。
    如果未實現則拋出 NotImplementedError。
    """
    if algo_name == 'ObjectDetection' and framework_name == 'YOLOv8':
        return start_training_yolo_detection_task
    else:
        raise NotImplementedError  # 如果未實現對應的訓練器，則拋出錯誤
