'''
Author: Will Cheng chengyong@pku.edu.cn
Date: 2024-08-25 14:44:40
LastEditors: Will Cheng chengyong@pku.edu.cn
LastEditTime: 2024-08-25 19:50:36
FilePath: /PoseidonAI-Server/utils/visualize_val/visualize_yolov8_detection.py
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

def convert_yolo_to_bbox(label_file, img_width, img_height):
    bboxes = []
    classes = []
    with open(label_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # 解析一行，提取出class, x_center, y_center, width, height
            parts = line.strip().split()
            cls = int(parts[0])
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height
            
            # 计算矩形框的左上角 (x0, y0) 和右下角 (x1, y1) 坐标
            x0 = int(x_center - width / 2)
            y0 = int(y_center - height / 2)
            x1 = int(x_center + width / 2)
            y1 = int(y_center + height / 2)
            
            # 添加到结果列表中
            bboxes.append([x0, y0, x1, y1])
            classes.append(cls)
    
    return bboxes, classes


class VisualizeYolov8Detection:
    def __init__(self, project_root, iou_thres, conf=0.01):
        
        self.project_root = project_root
        self.iou_thres = iou_thres
        self.conf = conf
        self.training_dir = os.path.join(project_root, 'project', 'exp')
        self.weights_dir = os.path.join(self.training_dir, 'weights')
        self.weights_file = os.path.join(self.weights_dir, 'best.pt')
        self.visualized_file = os.path.join(self.training_dir, 'visualized.json')
        assert os.path.exists(self.weights_file)
        self.dataset_config_file = os.path.join(self.project_root, 'dataset.yaml')
        self.dataset_cfg = read_yaml(self.dataset_config_file)
        self.device = '' if (torch.cuda.is_available()) else 'cpu'
        
        self.val_image_dir = os.path.join(self.project_root, 'data', 'images', 'val')
        self.val_label_dir = os.path.join(self.project_root, 'data', 'labels', 'val')
        self.val_image_files = sorted([d for d in glob.glob(os.path.join(self.val_image_dir, '*')) if os.path.isfile(d)])
        self.val_label_files = sorted([d for d in glob.glob(os.path.join(self.val_label_dir, '*.txt')) if os.path.isfile(d)])
        self.class_names = self.dataset_cfg['names']
        self.model = YOLO(self.weights_file)
        
        self.status = VisualizeStatus.IDLE
        self.error_detail = None
        
    def run_predict(self):
        preds = []
        results = self.model(self.val_image_files, iou=self.iou_thres, conf=self.conf, save=False, device=self.device)
        for i, result in enumerate(results):
            boxes = result.boxes  # Boxes object for bounding box outputs
            probs = result.probs  # Probs object for classification outputs
            bboxes = np.asarray(boxes.xyxy.cpu().tolist(), dtype=np.int32).tolist()
            classes = [int(d) for d in boxes.cls.cpu().tolist()]
            confs = [float(d) for d in boxes.conf.cpu().tolist()]
            orig_height, orig_width = boxes.orig_shape
            orig_boxes, orig_classes = convert_yolo_to_bbox(self.val_label_files[i], orig_width, orig_height)
            pred = dict(
                filename=ntpath.basename(self.val_image_files[i]),
                dt=dict(
                    points=bboxes,
                    conf=confs,
                    cls=classes
                ),
                gt=dict(
                    points=orig_boxes,
                    cls=orig_classes
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

class VisualizeYolov8DetectionTask(Task):
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f'Task {task_id} failed: {exc}')
        return super().on_failure(exc, task_id, args, kwargs, einfo)


@task(bind=True, base=VisualizeYolov8DetectionTask, name='tasks.visualize.yolov8.detection')
def start_visualize_yolov8_detection_task(self, project_root, iou_thres, conf=0.01):
    self.status = VisualizeStatus.PENDING
    self.update_state(state=VisualizeStatus.PENDING.name, meta={'status': self.status.name, 'exc_type': '', 'exc_message': ''})
    visualizer = VisualizeYolov8Detection(project_root, iou_thres, conf)
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


