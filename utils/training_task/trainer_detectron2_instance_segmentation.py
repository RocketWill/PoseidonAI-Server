import os
import json
import shutil
import logging
from enum import Enum, auto
import torch
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
import detectron2.utils.comm as comm
from detectron2.data.datasets import register_coco_instances

from celery import Task, task
from celery.exceptions import SoftTimeLimitExceeded, Ignore
from billiard.exceptions import Terminated
from app.models import TrainingTask
from utils.common import read_json

# Setting up logging
logger = logging.getLogger(__name__)

# Define enums for training statuses and errors
class TrainingStatus(Enum):
    IDLE = 'IDLE'
    PENDING = 'PENDING'
    PROCESSING = 'PROCESSING'
    ERROR = 'ERROR'
    SUCCESS = 'SUCCESS'
    REVOKED = 'REVOKED'

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

def read_detectron2_metrics(json_file_path: str) -> dict:
    if not os.path.exists(json_file_path):
        logger.warning(f"Metrics file not found at {json_file_path}")
        return {
            'iteration': [],
            'train_loss': [],
            'val_loss': []
        }

    iterations = []
    train_losses = []
    val_losses = []

    with open(json_file_path, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            iterations.append(data.get('iteration', None))
            train_losses.append(data.get('total_loss', None))
            val_losses.append(data.get('total_val_loss', None))

    return {
        'iteration': iterations,
        'train_loss': train_losses,
        'val_loss': val_losses
    }

class Detectron2Trainer:
    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.cfg_file = os.path.join(project_dir, 'cfg.yaml')
        self.loss_file = os.path.join(project_dir, 'project', 'metrics.json')
        self.status = TrainingStatus.IDLE
        self.error_detail = None
        self.project = os.path.join(project_dir, 'project')
        self.__remove_prev_project()
        os.makedirs(self.project, exist_ok=True)
        self.cfg = self.__init_cfg()
        self.trainer = self.__init_trainer()

    def __remove_prev_project(self):
        if os.path.exists(self.project):
            logger.info(f"Removing previous project directory at {self.project}")
            shutil.rmtree(self.project)
        
    def __init_cfg(self):
        logger.info("Initializing configuration")
        register_coco_instances("train_dataset", {}, os.path.join(self.project_dir, "data/train.json"), os.path.join(self.project_dir, "data/train"))
        register_coco_instances("val_dataset", {}, os.path.join(self.project_dir, "data/val.json"), os.path.join(self.project_dir, "data/val"))
        cfg = get_cfg()
        cfg.merge_from_file(self.cfg_file)
        cfg.DATALOADER.NUM_WORKERS = 0
        cfg.DATASETS.TRAIN = ("train_dataset",)
        cfg.DATASETS.TEST = ("val_dataset",)
        return cfg.clone()
    
    def __init_trainer(self):
        logger.info("Initializing trainer")
        trainer = DefaultTrainer(self.cfg) 
        val_loss = ValidationLoss(self.cfg)  
        trainer.register_hooks([val_loss])
        # Swap the order of PeriodicWriter and ValidationLoss
        trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
        trainer.resume_or_load(resume=False)
        return trainer
    
    def parse_loss(self) -> dict:
        return read_detectron2_metrics(self.loss_file)
    
    def start(self) -> bool:
        logger.info("Starting training")
        self.status = TrainingStatus.PROCESSING
        try:
            self.trainer.train()
            self.status = TrainingStatus.SUCCESS
            return True
        except TypeError as e:
            self.status = TrainingStatus.ERROR
            self.error_detail = (TrainingError.PARAMETER_TYPE_ERROR, str(e))
            logger.error(f"Parameter type error: {str(e)}")
        except ValueError as e:
            self.status = TrainingStatus.ERROR
            self.error_detail = (TrainingError.BATCH_SIZE_TOO_LARGE, str(e))
            logger.error(f"Batch size too large: {str(e)}")
        except RuntimeError as e:
            self.status = TrainingStatus.ERROR
            if 'CUDA' in str(e):
                self.error_detail = (TrainingError.CUDA_OUT_OF_MEMORY, str(e))
                logger.error(f"CUDA out of memory: {str(e)}")
            elif 'Dimension mismatch' in str(e):
                self.error_detail = (TrainingError.DIMENSION_MISMATCH, str(e))
                logger.error(f"Dimension mismatch: {str(e)}")
            elif 'Model not converging' in str(e):
                self.error_detail = (TrainingError.MODEL_NOT_CONVERGING, str(e))
                logger.error(f"Model not converging: {str(e)}")
            else:
                self.error_detail = (TrainingError.OTHER_ERROR, str(e))
                logger.error(f"Runtime error: {str(e)}")
        except FileNotFoundError as e:
            self.status = TrainingStatus.ERROR
            self.error_detail = (TrainingError.FILE_NOT_FOUND, str(e))
            logger.error(f"File not found: {str(e)}")
        except FloatingPointError as e:
            self.status = TrainingStatus.ERROR
            self.error_detail = (TrainingError.NUMERICAL_UNDERFLOW_OVERFLOW, str(e))
            logger.error(f"Numerical underflow/overflow: {str(e)}")
        except Exception as e:
            self.status = TrainingStatus.ERROR
            self.error_detail = (TrainingError.OTHER_ERROR, str(e))
            logger.error(f"Unexpected error: {str(e)}")
        return False

# Custom hook for validation loss
class ValidationLoss(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST
        self._loader = iter(build_detection_train_loader(self.cfg))
        
    def after_step(self):
        data = next(self._loader)
        with torch.no_grad():
            loss_dict = self.trainer.model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {"val_" + k: v.item() for k, v in 
                                 comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                self.trainer.storage.put_scalars(total_val_loss=losses_reduced, 
                                                 **loss_dict_reduced)

class Detectron2TrainingTask(Task):
    name = 'tasks.train.detectron2'

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"Task {task_id} failed due to {exc}")
        if isinstance(exc, Terminated):
            TrainingTask.update_status(task_id, TrainingStatus.REVOKED.value)
        else:
            TrainingTask.update_status(task_id, TrainingStatus.ERROR.value)
        super().on_failure(exc, task_id, args, kwargs, einfo)

@task(bind=True, base=Detectron2TrainingTask)
def start_training_detectron2_task(self, project_dir: str, task_id: str):
    try:
        logger.info(f"Starting Detectron2 training task with task_id: {task_id}")
        TrainingTask.update_status(task_id, TrainingStatus.PENDING.value)
        trainer = Detectron2Trainer(project_dir)
        TrainingTask.update_status(task_id, TrainingStatus.PROCESSING.value)
        self.update_state(state='PROCESSING', meta={'exc_type': '', 'exc_message': '', 'status': trainer.status.value})

        success = trainer.start()

        if success:
            TrainingTask.update_status(task_id, TrainingStatus.SUCCESS.value)
            self.update_state(state='SUCCESS', meta={'exc_type': '', 'exc_message': '', 'status': trainer.status.value})
            logger.info(f"Task {task_id} completed successfully")
            return {'status': trainer.status.value.upper(), 'error_detail': None}
        else:
            error_detail_str = f"{trainer.error_detail[0].name}: {trainer.error_detail[1]}"
            TrainingTask.update_status(task_id, TrainingStatus.ERROR.value)
            self.update_state(state='FAILURE', meta={'exc_type': '', 'exc_message': '', 'status': trainer.status.value, 'error_detail': error_detail_str})
            logger.error(f"Task {task_id} failed with error: {error_detail_str}")
            return {'status': trainer.status.value.upper(), 'error_detail': error_detail_str}

    except SoftTimeLimitExceeded:
        TrainingTask.update_status(task_id, TrainingStatus.REVOKED.value)
        self.update_state(state=TrainingStatus.REVOKED.value, meta={'exc_type': '', 'exc_message': '', 'status': TrainingStatus.REVOKED.value})
        logger.warning(f"Task {task_id} was revoked due to time limit exceeded")
        raise Ignore()
    except Exception as e:
        trainer.status = TrainingStatus.ERROR
        trainer.error_detail = (TrainingError.OTHER_ERROR, str(e))
        self.update_state(state='FAILURE', meta={'exc_type': '', 'exc_message': '', 'status': trainer.status.value, 'error_detail': str(trainer.error_detail)})
        TrainingTask.update_status(task_id, TrainingStatus.ERROR.value)
        logger.error(f"Unexpected error occurred in task {task_id}: {str(e)}")
        return {'status': trainer.status.value.upper(), 'error_detail': str(trainer.error_detail)}
