'''
Author: Will Cheng chengyong@pku.edu.cn
Date: 2024-08-25 14:44:40
LastEditors: Will Cheng (will.cheng@efctw.com)
LastEditTime: 2024-10-16 11:04:40
FilePath: /PoseidonAI-Server/utils/visualize_val/visualize_yolov8_classify.py
Description: 

Copyright (c) 2024 by chengyong@pku.edu.cn, All Rights Reserved. 
'''
import os
import glob
import ntpath
import logging

import torch
from celery import Task, task
import numpy as np
from ultralytics import YOLO

from utils.common import read_yaml, write_json
from .common import VisualizeError, VisualizeStatus

logger = logging.getLogger(__name__)

class VisualizeYolov8Classify:
    def __init__(self, project_root, *args):
        
        self.project_root = project_root
        self.training_dir = os.path.join(project_root, 'project', 'exp')
        self.weights_dir = os.path.join(self.training_dir, 'weights')
        self.weights_file = os.path.join(self.weights_dir, 'best.pt')
        self.visualized_file = os.path.join(self.training_dir, 'visualized.json')
        assert os.path.exists(self.weights_file)
        self.dataset_config_file = os.path.join(self.project_root, 'dataset.yaml')
        self.dataset_cfg = read_yaml(self.dataset_config_file)
        self.cfg = read_yaml( os.path.join(project_root, 'cfg.yaml'))
        self.device = '' if (torch.cuda.is_available()) else 'cpu'
        
        self.val_image_dir = os.path.join(self.project_root, 'data', 'val')
        self.model = YOLO(self.weights_file)
        self.class_names = self.model.names.values()
        self.val_class_images_pairs = self.__get_val_class_image_pairs()
        self.name_id_map = {v: k for k, v in self.model.names.items()}
        self.status = VisualizeStatus.IDLE
        self.error_detail = None

    def __get_val_class_image_pairs(self):
        class_image_pairs = []
        for class_name in self.class_names:
            image_files = glob.glob(os.path.join(self.val_image_dir, class_name, '*'))
            [class_image_pairs.append((class_name, image_file)) for image_file in image_files]
        return class_image_pairs
        
    def run_predict(self):
        preds = []
        val_image_files = [d[1] for d in self.val_class_images_pairs]
        results = self.model(val_image_files, save=False, device=self.device, imgsz=self.cfg['imgsz'])
        for i, result in enumerate(results):
            dt_cls = result.probs.top1
            gt_cls = self.name_id_map[self.val_class_images_pairs[i][0]]
            pred = dict(
                filename=ntpath.basename(val_image_files[i]),
                dt=dict(
                    points=-1,
                    conf=-1,
                    cls=dt_cls
                ),
                gt=dict(
                    points=-1,
                    cls=gt_cls
                )
            )
            preds.append(pred)
        return preds
    
    def predict(self):
        preds = self.run_predict()
        results = dict(
            class_names=[{'id': i, 'name': d} for (i, d) in enumerate(self.class_names)],
            preds=preds
        )
        write_json(results, self.visualized_file)
        return self.visualized_file
    
    def run_visualization(self):
        try:
            self.status = VisualizeStatus.PROCESSING
            preds = self.predict()
            self.status = VisualizeStatus.SUCCESS
            return preds
        except AssertionError as e:
            self.status = VisualizeStatus.ERROR
            self.error_detail = (VisualizeError.ASSERTION_FAILED, str(e))
            logger.error(f"Assertion failed: {str(e)}")
        except FileNotFoundError as e:
            self.status = VisualizeStatus.ERROR
            self.error_detail = (VisualizeError.FILE_NOT_FOUND, str(e))
            logger.error(f"File not found: {str(e)}")
        except RuntimeError as e:
            self.status = VisualizeStatus.ERROR
            if 'CUDA out of memory' in str(e):
                self.error_detail = (VisualizeError.CUDA_OUT_OF_MEMORY, str(e))
                logger.error(f"CUDA out of memory: {str(e)}")
            else:
                self.error_detail = (VisualizeError.RUNTIME_ERROR, str(e))
                logger.error(f"Runtime error: {str(e)}")
        except Exception as e:
            self.status = VisualizeStatus.ERROR
            self.error_detail = (VisualizeError.UNEXPECTED_ERROR, str(e))
            logger.error(f"Unexpected error during visualization: {str(e)}")
        return None

class VisualizeYolov8ClassifyTask(Task):
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f'Task {task_id} failed: {exc}')
        return super().on_failure(exc, task_id, args, kwargs, einfo)


@task(bind=True, base=VisualizeYolov8ClassifyTask, name='tasks.visualize.yolov8.classify')
def start_visualize_yolov8_classify_task(self, project_root, iou_thres, conf=0.01):
    self.status = VisualizeStatus.PENDING
    self.update_state(state=VisualizeStatus.PENDING.name, meta={'status': self.status.name, 'exc_type': '', 'exc_message': ''})
    visualizer = VisualizeYolov8Classify(project_root, iou_thres, conf)
    try:
        preds_file = visualizer.run_visualization()
        if preds_file is not None:
            self.status = VisualizeStatus.SUCCESS
            self.update_state(state=VisualizeStatus.SUCCESS.name, meta={'status': self.status.name, 'preds_file': preds_file, 'exc_type': '', 'exc_message': ''})
            logger.info(f"Task status: {self.status.name}")
            return {'status': self.status.name, 'preds': preds_file}
        else:
            self.status = VisualizeStatus.FAILURE
            error_detail_str = f"{self.error_detail[0].name}: {self.error_detail[1]}"
            self.update_state(state=VisualizeStatus.FAILURE.name, meta={'status': self.status.name, 'error_detail': error_detail_str, 'exc_type': self.error_detail[0].name, 'exc_message': self.error_detail[1]})
            logger.error(f"Task status: {self.status.name}. Visualization failed with error: {error_detail_str}")
            return {'status': self.status.name, 'error_detail': error_detail_str}
    except Exception as e:
        self.status = VisualizeStatus.FAILURE
        self.update_state(state=VisualizeStatus.FAILURE.name, meta={'status': self.status.name, 'error_detail': str(e), 'exc_type': 'UnexpectedError', 'exc_message': str(e)})
        logger.error(f"Task status: {self.status.name}. Unexpected error: {str(e)}")
        return {'status': self.status.name, 'error_detail': str(e)}
    finally:
        if self.status == VisualizeStatus.PROCESSING:
            self.status = VisualizeStatus.REVOKED
            self.update_state(state=VisualizeStatus.REVOKED.name, meta={'status': self.status.name, 'exc_type': '', 'exc_message': ''})
            logger.warning(f"Task status: {self.status.name}. Task was revoked.")


