'''
Author: Will Cheng chengyong@pku.edu.cn
Date: 2024-08-25 15:12:59
LastEditors: Will Cheng (will.cheng@efctw.com)
LastEditTime: 2024-10-18 14:09:55
FilePath: /PoseidonAI-Server/utils/visualize_val/common.py
Description: 

Copyright (c) 2024 by chengyong@pku.edu.cn, All Rights Reserved. 
'''
import os
import shutil
import glob
import ntpath
from enum import Enum, auto


class VisualizeError(Enum):
    ASSERTION_FAILED = auto()
    FILE_NOT_FOUND = auto()
    MEMORY_ERROR = auto()
    RUNTIME_ERROR = auto()
    CUDA_OUT_OF_MEMORY = auto()
    CUDNN_RNN_BACKWARD_ERROR = auto()
    UNEXPECTED_ERROR = auto()

class VisualizeStatus(Enum):
    IDLE = auto
    PENDING = auto()
    PROCESSING = auto()
    SUCCESS = auto()
    FAILURE = auto()
    ERROR = auto()
    REVOKED = auto()

def move_images(src_dir, dst_dir, detect_type='classify'):
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)

    if detect_type == 'classify':
        image_files = [d for d in glob.glob(os.path.join(src_dir, '*', '*')) if os.path.isfile(d)]
    else:
        image_files = [d for d in glob.glob(os.path.join(src_dir, '*')) if os.path.isfile(d)]
    for image_file in image_files:
        file_name = ntpath.basename(image_file)
        dst_file = os.path.join(dst_dir, file_name)
        os.symlink(image_file, dst_file)