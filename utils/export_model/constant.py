'''
Author: Will Cheng (will.cheng@efctw.com)
Date: 2024-09-18 17:11:30
LastEditors: Will Cheng (will.cheng@efctw.com)
LastEditTime: 2024-10-09 16:46:36
FilePath: /PoseidonAI-Server/utils/export_model/constant.py
'''
import os
from os.path import abspath, dirname


current_file_path = abspath(__file__)
current_dir = dirname(current_file_path)
current_data_dir = os.path.join(current_dir, 'data')

TORCHSCRIPT_DEPS_DIR = os.path.join(current_data_dir, 'torchscript_deps')
OPENVINO_DEPS_DIR = os.path.join(current_data_dir, 'openvino_deps')
YOLOV8_DETNET_TS_DLL = os.path.join(current_data_dir, 'yolov8_torchscript', 'EFC_DETNET_TS_runtime.dll')
DETECTRON2_SEGNET_TS_DLL = os.path.join(current_data_dir, 'detectron2_seg_torchscript', 'EFC_SEGNET_TS_runtime.dll')
YOLOV8_DETNET_OV_DLL = os.path.join(current_data_dir, 'yolov8_openvino', 'EFC_DETNET_OV_runtime.dll')
YOLOV8_CLSNET_OV_DLL = os.path.join(current_data_dir, 'yolov8_openvino', 'EFC_CLSNET_OV_runtime.dll')