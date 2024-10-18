import os
import numpy as np
import shutil
import yaml
import pandas as pd
import psutil
import gc
from enum import Enum, auto
from celery import task, Task
from celery.exceptions import SoftTimeLimitExceeded, Ignore
from billiard.exceptions import Terminated
from ultralytics import YOLO
from app.models import TrainingTask

# 定義訓練狀態的枚舉類型
class TrainingStatus(Enum):
    IDLE = 'IDLE'
    PENDING = 'PENDING'
    PROCESSING = 'PROCESSING'
    ERROR = 'ERROR'
    SUCCESS = 'SUCCESS'
    REVOKED = 'REVOKED'

# 定義訓練錯誤的枚舉類型
class TrainingError(Enum):
    PARAMETER_TYPE_ERROR = auto()
    BATCH_SIZE_TOO_LARGE = auto()
    CUDA_OUT_OF_MEMORY = auto()
    FILE_NOT_FOUND = auto()
    DIMENSION_MISMATCH = auto()
    MODEL_NOT_CONVERGING = auto()
    NUMERICAL_UNDERFLOW_OVERFLOW = auto()
    OTHER_ERROR = auto()
    WORKER_TERMINATED = auto()

# 讀取 YAML 文件的輔助函數
def read_yaml(file_path: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as exc:
            print(f"Error reading YAML file: {exc}")
            return None

# 列出目錄中的所有子目錄
def list_directories(path: str) -> list:
    directories = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            directories.append(os.path.abspath(item_path))
    return directories

# 讀取 YOLO 訓練過程中的損失值
def read_yolo_loss_values(csv_file_path: str, max_value: float = 1e3) -> dict:
    if not os.path.exists(csv_file_path):
        return {'epoch': [], 'train_loss': [], 'val_loss': []}
    
    df = pd.read_csv(csv_file_path)
    df.columns = df.columns.str.strip()

    # Replace 'inf' and '-inf' with a large positive value
    df.replace([np.inf, -np.inf], max_value, inplace=True)

    # Convert columns to lists
    epoch = df['epoch'].tolist()
    train_loss = df['train/loss'].tolist()
    val_loss = df['val/loss'].tolist()

    # Handle inf values in the loss columns
    train_loss = [min(float(loss), max_value) for loss in train_loss]
    val_loss = [min(float(loss), max_value) for loss in val_loss]

    return {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss}


# YOLO 訓練器類
class YoloTrainer:
    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.cfg_file = os.path.join(project_dir, 'cfg.yaml')
        self.cfg = read_yaml(self.cfg_file)
        self.model = self.cfg['model']
        self.loss_file = os.path.join(project_dir, 'project', 'exp', 'results.csv')
        self.status = TrainingStatus.IDLE
        self.error_detail = None
        self.__remove_prev_project()  # 清除之前的項目資料

    def __remove_prev_project(self):
        if not os.path.exists(os.path.join(self.project_dir, 'project')):
            return
        dirs = list_directories(os.path.join(self.project_dir, 'project'))
        for d in dirs:
            shutil.rmtree(d)

    def parse_loss(self) -> dict:
        return read_yolo_loss_values(self.loss_file)

    def start(self):
        self.status = TrainingStatus.PROCESSING
        try:
            model = YOLO(self.model)
            results = model.train(**self.cfg)
            self.status = TrainingStatus.SUCCESS
            return results
        except TypeError as e:
            self.status = TrainingStatus.ERROR
            self.error_detail = (TrainingError.PARAMETER_TYPE_ERROR, str(e))
        except ValueError as e:
            self.status = TrainingStatus.ERROR
            self.error_detail = (TrainingError.BATCH_SIZE_TOO_LARGE, str(e))
        except RuntimeError as e:
            self.status = TrainingStatus.ERROR
            if 'CUDA' in str(e):
                self.error_detail = (TrainingError.CUDA_OUT_OF_MEMORY, str(e))
            elif 'Dimension mismatch' in str(e):
                self.error_detail = (TrainingError.DIMENSION_MISMATCH, str(e))
            elif 'Model not converging' in str(e):
                self.error_detail = (TrainingError.MODEL_NOT_CONVERGING, str(e))
            else:
                self.error_detail = (TrainingError.OTHER_ERROR, str(e))
        except FileNotFoundError as e:
            self.status = TrainingStatus.ERROR
            self.error_detail = (TrainingError.FILE_NOT_FOUND, str(e))
        except FloatingPointError as e:
            self.status = TrainingStatus.ERROR
            self.error_detail = (TrainingError.NUMERICAL_UNDERFLOW_OVERFLOW, str(e))
        except Exception as e:
            self.status = TrainingStatus.ERROR
            self.error_detail = (TrainingError.OTHER_ERROR, str(e))
        return None

# 檢查系統資源是否足夠
def check_system_resources():
    memory_info = psutil.virtual_memory()
    if memory_info.available < 1 * 1024 * 1024 * 1024:  # 少於 1GB 的可用內存
        raise MemoryError("Not enough memory to start training")

# 釋放資源以避免內存洩漏
def release_resources():
    gc.collect()

# Celery 任務類，用於處理 YOLOv8 檢測任務
class YoloClassifyTask(Task):
    name = 'tasks.train.yolov8.classify'

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        if isinstance(exc, Terminated):
            TrainingTask.update_status(task_id, TrainingStatus.REVOKED.value)
        else:
            TrainingTask.update_status(task_id, TrainingStatus.ERROR.value)
        super().on_failure(exc, task_id, args, kwargs, einfo)

# Celery 任務函數，用於啟動 YOLOv8 訓練
@task(bind=True, base=YoloClassifyTask)
def start_training_yolo_classify_task(self, project_dir: str, task_id: str):
    try:
        TrainingTask.update_status(task_id, TrainingStatus.PENDING.value)
        check_system_resources()

        trainer = YoloTrainer(project_dir)
        TrainingTask.update_status(task_id, TrainingStatus.PROCESSING.value)
        self.update_state(state='PROCESSING', meta={'exc_type': '', 'exc_message': '', 'status': trainer.status.value})

        results = trainer.start()
    except SoftTimeLimitExceeded:
        TrainingTask.update_status(task_id, TrainingStatus.REVOKED.value)
        self.update_state(state=TrainingStatus.REVOKED.value, meta={'exc_type': '', 'exc_message': '', 'status': TrainingStatus.REVOKED.value})
        raise Ignore()
    except Exception as e:
        trainer.status = TrainingStatus.ERROR
        trainer.error_detail = (TrainingError.OTHER_ERROR, str(e))
        self.update_state(state='FAILURE', meta={'exc_type':  type(e).__name__, 'exc_message': str(trainer.error_detail), 'status': trainer.status.value, 'error_detail': str(trainer.error_detail)})
        results = None
    finally:
        release_resources()
        TrainingTask.update_status(task_id, trainer.status.value)
        self.update_state(state='PROCESSING', meta={'exc_type': 'error', 'exc_message': str(trainer.error_detail), 'status': trainer.status.value, 'error_detail': str(trainer.error_detail)})

    if trainer.status == TrainingStatus.SUCCESS:
        loss_values = trainer.parse_loss()
        TrainingTask.update_status(task_id, TrainingStatus.SUCCESS.value)
        self.update_state(state='SUCCESS', meta={'exc_type': '', 'exc_message': '', 'status': trainer.status.value})
        return {'status': trainer.status.value.upper(), 'results': loss_values, 'error_detail': None}
    else:
        error_detail_str = f"{trainer.error_detail[0].name}: {trainer.error_detail[1]}"
        TrainingTask.update_status(task_id, TrainingStatus.ERROR.value)
        self.update_state(state='FAILURE', meta={'exc_type': '', 'exc_message': '', 'status': trainer.status.value, 'error_detail': error_detail_str})
        return {'status': trainer.status.value.upper(), 'results': None, 'error_detail': error_detail_str}
