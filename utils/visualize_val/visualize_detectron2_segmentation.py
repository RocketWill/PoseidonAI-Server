import os
import glob
import ntpath
import logging

import cv2
import numpy as np
import torch
from celery import Task, task
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data import DatasetCatalog

from utils.common import read_yaml, write_json
from .common import VisualizeError, VisualizeStatus

logger = logging.getLogger(__name__)

class VisualizeDetectron2Segmentation:
    def __init__(self, project_root, iou_thres, conf=0.01):
        self.project_root = project_root
        self.iou_thres = iou_thres
        self.conf = conf
        self.cfg_file = os.path.join(project_root, 'cfg.yaml')
        self.key = ntpath.basename(project_root)
        self.training_dir = os.path.join(project_root, 'project')
        self.visualized_file = os.path.join(self.training_dir, 'visualized.json')
        self.weights_files = sorted(glob.glob(os.path.join(self.training_dir, '*.pth')))
        self.coco_val_gt_file = os.path.join(self.project_root, "data/val.json")
        self.coco_val_dt_file = os.path.join(self.training_dir, 'coco_instances_results.json')
        assert len(self.weights_files) > 0, "No weight files found in training directory."
        self.weight_file = self.weights_files[-1]
        self.dataset_names = ('train_dataset_{}'.format(self.key), 'val_dataset_{}'.format(self.key))
        self.status = VisualizeStatus.IDLE
        self.error_detail = None

        # Initialize COCO ground truth
        self.coco_gt = COCO(self.coco_val_gt_file)
        # Get class names
        cats = self.coco_gt.loadCats(self.coco_gt.getCatIds())
        cats = sorted(cats, key=lambda x: x['id'])
        self.class_names = cats  # List of dicts with 'id' and 'name'

    def __remove_registered_dataset(self):
        datasets = DatasetCatalog.list()
        for dataset_name in self.dataset_names:
            if dataset_name in datasets:
                DatasetCatalog.remove(dataset_name)

    def __register_dataset(self):
        try:
            self.__remove_registered_dataset()
            register_coco_instances(
                self.dataset_names[0],
                {},
                os.path.join(self.project_root, "data/train.json"),
                os.path.join(self.project_root, "data/train")
            )
            register_coco_instances(
                self.dataset_names[1],
                {},
                os.path.join(self.project_root, "data/val.json"),
                os.path.join(self.project_root, "data/val")
            )
        except FileNotFoundError as e:
            self.status = VisualizeStatus.ERROR
            self.error_detail = (VisualizeError.FILE_NOT_FOUND, str(e))
            logger.error(f"File not found: {str(e)}")
        except RuntimeError as e:
            self.status = VisualizeStatus.ERROR
            self.error_detail = (VisualizeError.RUNTIME_ERROR, str(e))
            logger.error(f"Runtime error: {str(e)}")

    def __init_cfg(self):
        cfg = get_cfg()
        cfg.merge_from_file(self.cfg_file)
        cfg.MODEL.WEIGHTS = self.weight_file
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.conf  # Set confidence threshold
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = self.iou_thres  # Set NMS IoU threshold
        cfg.DATASETS.TEST = (self.dataset_names[1],)
        cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        cfg.SOLVER.IMS_PER_BATCH = 1
        cfg.DATALOADER.NUM_WORKERS = 0
        return cfg.clone()

    def mask_to_polygons(self, mask):
        """
        Convert a binary mask to polygon coordinates.
        """
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        polygons = []
        for contour in contours:
            contour = contour.flatten().tolist()
            if len(contour) >= 6:  # At least 3 points (x, y)
                polygons.append(contour)
        return polygons

    def run_d2_segmentation(self):
        try:
            cfg = self.__init_cfg()
            if self.status == VisualizeStatus.ERROR:
                return False

            predictor = DefaultPredictor(cfg)

            # Perform inference and save results
            evaluator = COCOEvaluator(self.dataset_names[1], cfg, False, output_dir=self.training_dir)
            val_loader = build_detection_test_loader(cfg, self.dataset_names[1])

            # inference_on_dataset will automatically save predictions as a COCO-format JSON file
            inference_on_dataset(predictor.model, val_loader, evaluator)

            if not os.path.exists(self.coco_val_dt_file):
                self.status = VisualizeStatus.ERROR
                self.error_detail = (VisualizeError.FILE_NOT_FOUND, 'COCO detection results not found')
                logger.error('COCO detection results not found')
                return False
            return True
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
            logger.error(f"Unexpected error: {str(e)}")
            import traceback
            traceback.print_exc()
        return False

    def run_predict(self):
        preds = []
        coco_gt = self.coco_gt
        coco_dt = coco_gt.loadRes(self.coco_val_dt_file)

        img_ids = coco_gt.getImgIds()
        for img_id in img_ids:
            img_info = coco_gt.loadImgs(img_id)[0]
            filename = img_info['file_name']
            height = img_info['height']
            width = img_info['width']

            # Get ground truth annotations for this image
            ann_ids = coco_gt.getAnnIds(imgIds=img_id)
            anns = coco_gt.loadAnns(ann_ids)

            gt_points = []
            gt_classes = []
            for ann in anns:
                segmentation = ann['segmentation']
                category_id = ann['category_id']
                if isinstance(segmentation, list):
                    # Polygon format
                    for seg in segmentation:
                        gt_points.append(seg)
                        gt_classes.append(category_id)
                elif isinstance(segmentation, dict) and 'counts' in segmentation and 'size' in segmentation:
                    # RLE format
                    rle = segmentation
                    # Convert 'counts' to bytes if it's a list
                    if isinstance(rle['counts'], list):
                        rle['counts'] = maskUtils.frPyObjects([rle], rle['size'][0], rle['size'][1])[0]['counts']
                    
                    mask = maskUtils.decode(rle)
                    polygons = self.mask_to_polygons(mask)
                    for polygon in polygons:
                        gt_points.append(polygon)
                        gt_classes.append(category_id)

            # Get detection annotations for this image
            dt_ann_ids = coco_dt.getAnnIds(imgIds=img_id)
            dt_anns = coco_dt.loadAnns(dt_ann_ids)

            dt_points = []
            dt_classes = []
            dt_confs = []
            for ann in dt_anns:
                segmentation = ann['segmentation']
                category_id = ann['category_id']
                score = ann['score']

                if isinstance(segmentation, list):
                    # Polygon format
                    for seg in segmentation:
                        dt_points.append(seg)
                        dt_classes.append(category_id)
                        dt_confs.append(score)
                elif isinstance(segmentation, dict) or isinstance(segmentation, str):
                    # RLE format
                    rle = segmentation
                    if isinstance(rle['counts'], list):
                        rle['counts'] = maskUtils.frPyObjects([rle], rle['size'][0], rle['size'][1])[0]['counts']
                    
                    mask = maskUtils.decode(rle)
                    polygons = self.mask_to_polygons(mask)
                    for polygon in polygons:
                        dt_points.append(polygon)
                        dt_classes.append(category_id)
                        dt_confs.append(score)

            pred = dict(
                filename=filename,
                dt=dict(
                    points=dt_points,
                    conf=dt_confs,
                    cls=dt_classes
                ),
                gt=dict(
                    points=gt_points,
                    cls=gt_classes
                )
            )
            preds.append(pred)

        return preds


    def predict(self):
        preds = self.run_predict()
        results = dict(
            class_names=[{'id': cat['id'], 'name': cat['name']} for cat in self.class_names],
            preds=preds
        )
        write_json(results, self.visualized_file)
        return self.visualized_file

    def run_visualization(self):
        try:
            self.status = VisualizeStatus.PROCESSING
            # First, register the dataset
            self.__register_dataset()
            if self.status == VisualizeStatus.ERROR:
                # An error occurred during dataset registration
                return None
            # Run the detectron2 segmentation to generate detection results
            success = self.run_d2_segmentation()
            if not success:
                # An error occurred, status and error_detail are already set
                return None

            preds_file = self.predict()
            self.status = VisualizeStatus.SUCCESS
            return preds_file
        except Exception as e:
            self.status = VisualizeStatus.ERROR
            self.error_detail = (VisualizeError.UNEXPECTED_ERROR, str(e))
            logger.error(f"Unexpected error during visualization: {str(e)}")
            return None

class VisualizeDetectron2SegmentationTask(Task):
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f'Task {task_id} failed: {exc}')
        return super().on_failure(exc, task_id, args, kwargs, einfo)

@task(bind=True, base=VisualizeDetectron2SegmentationTask, name='tasks.visualize.detectron2.segmentation')
def start_visualize_detectron2_segmentation_task(self, project_root, iou_thres, conf=0.01):
    self.status = VisualizeStatus.PENDING
    self.update_state(state=VisualizeStatus.PENDING.name, meta={'status': self.status.name, 'exc_type': '', 'exc_message': ''})
    visualizer = VisualizeDetectron2Segmentation(project_root, iou_thres, conf)
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


if __name__ == '__main__':
    project_root = '/root/workspace/PoseidonAI-Server/data/projects/66a6eb9d4c0e8525ee44c787/5dccb913-c5f6-4aed-8ae1-8cb76b73ba81'
    iou_thres = 0.1
    viser = VisualizeDetectron2Segmentation(project_root, iou_thres)
    viser.run_visualization()