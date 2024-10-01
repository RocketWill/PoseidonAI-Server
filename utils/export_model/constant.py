'''
Author: Will Cheng (will.cheng@efctw.com)
Date: 2024-09-18 17:11:30
LastEditors: Will Cheng (will.cheng@efctw.com)
LastEditTime: 2024-09-18 18:14:46
FilePath: /PoseidonAI-Server/utils/export_model/constant.py
'''
import os
from os.path import abspath, dirname


current_file_path = abspath(__file__)
current_dir = dirname(current_file_path)
current_data_dir = os.path.join(current_dir, 'data')

TORCHSCRIPT_DEPS_DIR = os.path.join(current_data_dir, 'torchscript_deps')
YOLOV8_DETNET_TS_DLL = os.path.join(current_data_dir, 'yolov8_torchscript', 'EFC_DETNET_TS_runtime.dll')