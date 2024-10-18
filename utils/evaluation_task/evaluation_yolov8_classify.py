import os
import shutil
import logging

import torch
from celery import Task, task
import numpy as np
from ultralytics import YOLO

from utils.common import read_yaml, write_json
from .common import EvaluationError, EvaluationStatus

logger = logging.getLogger(__name__)

class EvalYolov8Classify:
    def __init__(self, project_root, iou_thres, batch_size, gpu_id=None) -> None:
        self.project_root = project_root
        self.training_dir = os.path.join(project_root, 'project', 'exp')
        self.weights_dir = os.path.join(self.training_dir, 'weights')
        self.args_file = os.path.join(self.training_dir, 'args.yaml')
        self.weights_file = os.path.join(self.weights_dir, 'best.pt')
        self.metrics_file = os.path.join(self.training_dir, 'evaluation.json')
        self.eval_dir = os.path.join(self.project_root, 'eval')
        self.iou_thres = iou_thres
        self.batch_size = batch_size
        self.gpu_id = gpu_id
        self.status = EvaluationStatus.IDLE
        self.error_detail = None
        self.__remove_prev_eval_dir()
        
    def __remove_prev_eval_dir(self):
        try:
            if os.path.exists(self.eval_dir):
                shutil.rmtree(self.eval_dir)
        except Exception as e:
            self.status = EvaluationStatus.ERROR
            self.error_detail = (EvaluationError.UNEXPECTED_ERROR, str(e))
            logger.error(f"Unexpected error while removing previous eval dir: {str(e)}")
            
    def __init_cfg(self):
        try:
            cfg = read_yaml(self.args_file)
            cfg = {
                'batch': self.batch_size, 
                'data': cfg.get('data'),
                'project': self.eval_dir,
                'name': 'eval',
                'plots': True,
                'save': False,
                'device': 'cuda:{}'.format(self.gpu_id) if (self.gpu_id and torch.cuda.is_available()) else 'cpu',
            }
            return cfg
        except FileNotFoundError as e:
            self.status = EvaluationStatus.ERROR
            self.error_detail = (EvaluationError.FILE_NOT_FOUND, str(e))
            logger.error(f"File not found: {str(e)}")
        except Exception as e:
            self.status = EvaluationStatus.ERROR
            self.error_detail = (EvaluationError.UNEXPECTED_ERROR, str(e))
            logger.error(f"Unexpected error while initializing config: {str(e)}")
    
    def run_eval(self):
        try:
            cfg = self.__init_cfg()
            if self.status == EvaluationStatus.ERROR:
                return False

            model = YOLO(self.weights_file)
            result = model.val(**cfg)
            names = model.names
            names = [{'id': i, 'name': val} for i, (key, val) in enumerate(model.names.items())]
            cm = result.confusion_matrix.matrix
            TP = np.diag(cm)
            FP = np.sum(cm, axis=0) - TP
            FN = np.sum(cm, axis=1) - TP
            TN = np.sum(cm) - (FP + FN + TP)

            # 计算Precision和Recall
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)

            # 计算每个类的F1分数
            f1_scores = 2 * (precision * recall) / (precision + recall)

            # 处理除以零的情况，将NaN替换为0
            f1_scores = np.nan_to_num(f1_scores)

            # 计算整体平均值
            precision_overall = np.mean(precision)
            recall_overall = np.mean(recall)
            f1_overall = np.mean(f1_scores)


            data_to_save = {
                "names": names,
                "f1": f1_scores.tolist(),
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                'result_dict': {'precision': float(precision_overall), 'recall': float(recall_overall), 'f1': float(f1_overall)},
                'parameters': {
                    'iou_thres': '-',
                    'batch_size': self.batch_size,
                    'gpu_id': self.gpu_id
                }
            }

            print(data_to_save)

            write_json(data_to_save, self.metrics_file)
            if not os.path.exists(self.metrics_file):
                self.status = EvaluationStatus.ERROR
                self.error_detail = (EvaluationError.FILE_NOT_FOUND, 'Save metrics file failed')
                logger.error('Save metrics file failed')
                return False
            
            return True
        except FileNotFoundError as e:
            self.status = EvaluationStatus.ERROR
            self.error_detail = (EvaluationError.FILE_NOT_FOUND, str(e))
            logger.error(f"File not found: {str(e)}")
        except RuntimeError as e:
            self.status = EvaluationStatus.ERROR
            if 'CUDA out of memory' in str(e):
                self.error_detail = (EvaluationError.CUDA_OUT_OF_MEMORY, str(e))
                logger.error(f"CUDA out of memory: {str(e)}")
            else:
                self.error_detail = (EvaluationError.RUNTIME_ERROR, str(e))
                logger.error(f"Runtime error: {str(e)}")
        except Exception as e:
            self.status = EvaluationStatus.ERROR
            self.error_detail = (EvaluationError.UNEXPECTED_ERROR, str(e))
            logger.error(f"Unexpected error during evaluation: {str(e)}")
        return False

    def eval(self):
        success = self.run_eval()
        if not success:
            logger.error(f"Evaluation failed: {self.error_detail}")
            return False
        return True

@task(bind=True, name='tasks.eval.yolov8.classify')
def start_eval_yolov8_classify_task(self, project_root, iou_thres, batch_size, gpu_id=None):
    evaluator = EvalYolov8Classify(project_root, iou_thres, batch_size, gpu_id)
    status = EvaluationStatus.PENDING
    
    self.update_state(state=EvaluationStatus.PENDING.name, meta={'exc_type': '', 'exc_message': '', 'status': status.name})
    
    try:
        status = EvaluationStatus.PROCESSING
        
        self.update_state(state=EvaluationStatus.PROCESSING.name, meta={'exc_type': '', 'exc_message': '', 'status': status.name})
        
        logger.info(f"Task status: {status.name}")
        success = evaluator.eval()
        if success:
            status = EvaluationStatus.SUCCESS
            self.update_state(state=EvaluationStatus.SUCCESS.name, meta={'exc_type': '', 'exc_message': '', 'status': status.name, 'metrics_file': evaluator.metrics_file})
            logger.info(f"Task status: {status.name}. Metrics saved at {evaluator.metrics_file}")
            return {'status': status.name, 'metrics_file': evaluator.metrics_file}
        else:
            status = EvaluationStatus.FAILURE
            error_detail_str = f"{evaluator.error_detail[0].name}: {evaluator.error_detail[1]}"
            self.update_state(state='FAILURE', meta={'exc_type': evaluator.error_detail[0].name, 'exc_message': evaluator.error_detail[1], 'status': status.name})
            logger.error(f"Task status: {status.name}. Evaluation failed with error: {error_detail_str}")
            return {'status': status.name, 'error_detail': error_detail_str}
    except Exception as e:
        status = EvaluationStatus.FAILURE
        self.update_state(state='FAILURE', meta={'exc_type': 'UnexpectedError', 'exc_message': str(e), 'status': status.name})
        logger.error(f"Task status: {status.name}. Unexpected error: {str(e)}")
        return {'status': status.name, 'error_detail': str(e)}
    finally:
        if status == EvaluationStatus.PROCESSING:
            status = EvaluationStatus.REVOKED
            self.update_state(state=EvaluationStatus.REVOKED.name, meta={'exc_type': '', 'exc_message': '', 'status': status.name})
            logger.warning(f"Task status: {status.name}. Task was revoked.")
