'''
Author: Will Cheng (will.cheng@efctw.com)
Date: 2024-09-11 16:11:09
LastEditors: Will Cheng (will.cheng@efctw.com)
LastEditTime: 2024-09-18 18:12:53
FilePath: /PoseidonAI-Server/utils/export_model/common.py
'''
from enum import Enum, auto

from utils.common import zip_files

class ExporterError(Enum):
    ASSERTION_FAILED = auto()
    FILE_NOT_FOUND = auto()
    MEMORY_ERROR = auto()
    RUNTIME_ERROR = auto()
    CUDA_OUT_OF_MEMORY = auto()
    CUDNN_RNN_BACKWARD_ERROR = auto()
    UNEXPECTED_ERROR = auto()

class ExporterStatus(Enum):
    IDLE = auto
    PENDING = auto()
    PROCESSING = auto()
    SUCCESS = auto()
    FAILURE = auto()
    ERROR = auto()
    REVOKED = auto()

def zip_model_and_deps(output_file, content, converted_model_file, deps_dir, runtime_file):
    if content == 'model':
        result = zip_files(output_file, [converted_model_file])
    elif content == 'model_runtime':
        result = zip_files(output_file, [converted_model_file, runtime_file])
    elif content == 'model_runtime_deps':
        result = zip_files(output_file, [converted_model_file, deps_dir, runtime_file])
    else:
        raise NotImplementedError('Unsupported content.')
    
    if not result:
        raise ValueError('Compress failed.')
    return output_file