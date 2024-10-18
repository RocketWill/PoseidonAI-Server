'''
Author: Will Cheng (will.cheng@efctw.com)
Date: 2024-09-11 15:51:10
LastEditors: Will Cheng (will.cheng@efctw.com)
LastEditTime: 2024-10-16 09:03:43
FilePath: /PoseidonAI-Server/utils/export_model/__init__.py
'''
import os
import os
from os.path import abspath, dirname

from utils.export_model.export_yolov8 import start_export_model as start_export_yolov8_model
from utils.export_model.export_detectron2 import start_export_model as start_export_detectron2_model

current_file_path = abspath(__file__)
current_dir = dirname(current_file_path)
current_data_dir = os.path.join(current_dir, 'data')

# Define constants for algorithm and framework names
CLASSIFICATION = 'Classification'
OBJECT_DETECTION = 'ObjectDetection'
INSTANCE_SEGMENTATION = 'InstanceSegmentation'
YOLOV8 = 'YOLOv8'
DETECTRON2_INSTANCE_SEGMENTATION = 'Detectron2-InstanceSegmentation'

# Define Model format
TORCHSCRIPT = 'Torchscript'
ONNX = 'Onnx'
OPENVINO = 'OpenVINO'
NCNN = 'ncnn'


def get_model_exporter(algo_name: str, framework_name: str):
    if (algo_name == OBJECT_DETECTION or algo_name == CLASSIFICATION) and framework_name == YOLOV8:
        return start_export_yolov8_model
    elif algo_name == INSTANCE_SEGMENTATION and framework_name == DETECTRON2_INSTANCE_SEGMENTATION:
        return start_export_detectron2_model
    else:
        raise NotImplementedError(f"Metrics Files not implemented for algorithm: {algo_name}, framework: {framework_name}")