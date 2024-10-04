'''
Author: Will Cheng chengyong@pku.edu.cn
Date: 2024-08-25 14:44:25
LastEditors: Will Cheng (will.cheng@efctw.com)
LastEditTime: 2024-10-04 11:13:09
FilePath: /PoseidonAI-Server/utils/visualize_val/__init__.py
Description: 

Copyright (c) 2024 by chengyong@pku.edu.cn, All Rights Reserved. 
'''
import os
from utils.visualize_val.visualize_yolov8_detection import start_visualize_yolov8_detection_task
from utils.visualize_val.visualize_detectron2_segmentation import start_visualize_detectron2_segmentation_task


# Define constants for algorithm and framework names
OBJECT_DETECTION = 'ObjectDetection'
INSTANCE_SEGMENTATION = 'InstanceSegmentation'
YOLOV8 = 'YOLOv8'
DETECTRON2_INSTANCE_SEGMENTATION = 'Detectron2-InstanceSegmentation'


def get_visualized_file(algo_name: str, framework_name: str, training_project_root: str, user_id: str, save_key: str) -> str:
    project_root = os.path.join(training_project_root, user_id, save_key)
    if algo_name == OBJECT_DETECTION and framework_name == YOLOV8:
        training_dir = os.path.join(project_root, 'project', 'exp')
        vis_file = os.path.join(training_dir, 'visualized.json')
        return vis_file
    elif algo_name == INSTANCE_SEGMENTATION and framework_name == DETECTRON2_INSTANCE_SEGMENTATION:
        training_dir = os.path.join(project_root, 'project')
        vis_file = os.path.join(training_dir, 'visualized.json')
        return vis_file
    else:
        raise NotImplementedError(f"Visualized Files not implemented for algorithm: {algo_name}, framework: {framework_name}")

def get_visualizer(algo_name: str, framework_name: str):
    if algo_name == OBJECT_DETECTION and framework_name == YOLOV8:
        return start_visualize_yolov8_detection_task
    elif algo_name == INSTANCE_SEGMENTATION and framework_name == DETECTRON2_INSTANCE_SEGMENTATION:
        return start_visualize_detectron2_segmentation_task
    else:
        raise NotImplementedError(f"Visualizer not implemented for algorithm: {algo_name}, framework: {framework_name}")