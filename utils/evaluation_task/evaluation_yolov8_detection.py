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

class EvalYolov8ObjectDetection:
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
                'conf': 0.01, 
                'iou': self.iou_thres,
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
            metrics = model.val(**cfg)
            
            confidence_1 = metrics.curves_results[1][0]
            f1_scores = metrics.curves_results[1][1]
            f1_mean = np.mean(f1_scores, axis=0)

            # 第二个图表：Confidence 和 Precision
            confidence_2 = metrics.curves_results[2][0]
            precision = metrics.curves_results[2][1]
            precision_mean = np.mean(precision, axis=0)

            # 第三个图表：Confidence 和 Recall
            confidence_3 = metrics.curves_results[3][0]
            recall = metrics.curves_results[3][1]
            recall_mean = np.mean(recall, axis=0)

            PR_recall = metrics.curves_results[0][0]  # Recall 数据
            PR_precision = metrics.curves_results[0][1]  # Precision 数据
            PR_precision_mean = np.mean(PR_precision, axis=0)

            data_to_save = {
                "names": [{"id": key, "name": value} for key, value in metrics.names.items()],
                "confidence": confidence_1.tolist(),
                "f1": {
                    "f1_scores": [f1.tolist() for f1 in f1_scores],  # F1-score 数据
                    "f1_mean": f1_mean.tolist()  # F1-score 平均值
                },
                "precision": {
                    "precision": [prec.tolist() for prec in precision],  # Precision 数据
                    "precision_mean": precision_mean.tolist()  # Precision 平均值
                },
                "recall": {
                    "recall": [rec.tolist() for rec in recall],  # Recall 数据
                    "recall_mean": recall_mean.tolist()  # Recall 平均值
                },
                "pr": {
                    'recall': [rec.tolist() for rec in PR_recall],
                    'precision': [prec.tolist() for prec in PR_precision],
                    'precision_mean': PR_precision_mean.tolist()
                },
                'result_dict': {'precision': metrics.results_dict['metrics/precision(B)'], 'recall': metrics.results_dict['metrics/recall(B)']}
            }

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

@task(bind=True, name='tasks.eval.yolov8.object.detection')
def start_eval_yolov8_detection_task(self, project_root, iou_thres, batch_size, gpu_id=None):
    evaluator = EvalYolov8ObjectDetection(project_root, iou_thres, batch_size, gpu_id)
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
