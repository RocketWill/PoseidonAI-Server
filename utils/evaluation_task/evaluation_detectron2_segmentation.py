import os
import glob
import ntpath
import logging
import traceback

import torch
from celery import Task, task
from pycocotools.coco import COCO
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data import DatasetCatalog

from utils.evaluation_task.improved_coco_eval_tool import SelfEval
from utils.common import write_json
from .common import EvaluationError, EvaluationStatus

logger = logging.getLogger(__name__)

class EvalDetectron2Segmentation:
    def __init__(self, project_root, iou_thres, batch_size, gpu_id=0):
        self.project_root = project_root
        self.iou_thres = iou_thres
        self.batch_size = batch_size
        self.gpu_id = int(gpu_id)
        self.cfg_file = os.path.join(project_root, 'cfg.yaml')
        self.key = ntpath.basename(project_root)
        self.training_dir = os.path.join(project_root, 'project')
        self.metrics_file = os.path.join(self.training_dir, 'evaluation.json')
        self.weights_files = sorted(glob.glob(os.path.join(self.training_dir, '*.pth')))
        self.cooc_val_gt_file = os.path.join(self.project_root, "data/val.json")
        assert len(self.weights_files) > 0
        self.weight_file = self.weights_files[-1]
        self.dataset_names = ('train_dataset_'.format(self.key), 'val_dataset_'.format(self.key))
        self.status = EvaluationStatus.IDLE
        self.error_detail = None

    def __remove_registed_dataset(self):
        datasets = DatasetCatalog.list()
        for dataset_name in self.dataset_names:
            if dataset_name in datasets:
                DatasetCatalog.remove(dataset_name)

    def __register_dataset(self):
        try:
            self.__remove_registed_dataset()
            register_coco_instances(self.dataset_names[0], {}, os.path.join(self.project_root, "data/train.json"), os.path.join(self.project_root, "data/train"))
            register_coco_instances(self.dataset_names[1], {}, os.path.join(self.project_root, "data/val.json"), os.path.join(self.project_root, "data/val"))
        except FileNotFoundError as e:
            self.status = EvaluationStatus.ERROR
            self.error_detail = (EvaluationError.FILE_NOT_FOUND, str(e))
            logger.error(f"File not found: {str(e)}")
        except RuntimeError as e:
            self.status = EvaluationStatus.ERROR
            self.error_detail = (EvaluationError.RUNTIME_ERROR, str(e))
            logger.error(f"Runtime error: {str(e)}")

    def __init_cfg(self):
        try:
            self.__register_dataset()
            cfg = get_cfg()
            cfg.merge_from_file(self.cfg_file)
            cfg.DATALOADER.NUM_WORKERS = 0
            cfg.DATASETS.TRAIN = (self.dataset_names[0],)
            cfg.DATASETS.TEST = (self.dataset_names[1],)
            cfg.MODEL.WEIGHTS = self.weight_file
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
            cfg.MODEL.DEVICE = "cuda:{}".format(self.gpu_id) if torch.cuda.is_available() else "cpu"
            cfg.SOLVER.IMS_PER_BATCH = self.batch_size
            return cfg.clone()
        except FileNotFoundError as e:
            self.status = EvaluationStatus.ERROR
            self.error_detail = (EvaluationError.FILE_NOT_FOUND, str(e))
            logger.error(f"File not found: {str(e)}")
        except RuntimeError as e:
            self.status = EvaluationStatus.ERROR
            self.error_detail = (EvaluationError.RUNTIME_ERROR, str(e))
            logger.error(f"Runtime error: {str(e)}")

    def run_coco_eval(self, cooc_val_gt_file, coco_val_dt_file, iou_thres):
        try:
            coco_gt = COCO(cooc_val_gt_file)
            coco_dt = coco_gt.loadRes(coco_val_dt_file)
            segm_eval = SelfEval(coco_gt, coco_dt, all_points=True, iou_type='segmentation', iou_thres=[iou_thres])
            segm_eval.evaluate()
            segm_eval.accumulate()
            segm_eval.summarize()
            metrics = segm_eval.get_curves_data_for_iou(iou_thres)
            return metrics
        except FileNotFoundError as e:
            self.status = EvaluationStatus.ERROR
            self.error_detail = (EvaluationError.FILE_NOT_FOUND, str(e))
            logger.error(f"File not found: {str(e)}")
        except ValueError as e:
            self.status = EvaluationStatus.ERROR
            self.error_detail = (EvaluationError.RUNTIME_ERROR, str(e))
            logger.error(f"Runtime error: {str(e)}")
        except Exception as e:
            self.status = EvaluationStatus.ERROR
            self.error_detail = (EvaluationError.UNEXPECTED_ERROR, str(e))
            logger.error(f"Unexpected error: {str(e)}")

    def eval_d2_segmentaion(self):
        try:
            cfg = self.__init_cfg()
            if self.status == EvaluationStatus.ERROR:
                return False

            predictor = DefaultPredictor(cfg)

            # 执行推理并保存结果
            evaluator = COCOEvaluator(self.dataset_names[1], cfg, False, output_dir=self.training_dir)
            val_loader = build_detection_test_loader(cfg, self.dataset_names[1])

            # inference_on_dataset将预测结果自动保存为COCO格式的JSON文件
            inference_on_dataset(predictor.model, val_loader, evaluator)

            coco_val_dt_file = os.path.join(self.training_dir, 'coco_instances_results.json')
            if not os.path.exists(coco_val_dt_file):
                self.status = EvaluationStatus.ERROR
                self.error_detail = (EvaluationError.FILE_NOT_FOUND, 'Coco dt results not found')
                logger.error('Coco dt results not found')
                return False

            metrics = self.run_coco_eval(self.cooc_val_gt_file, coco_val_dt_file, self.iou_thres)
            metrics['parameters'] = {
                'iou_thres': self.iou_thres,
                'batch_size': self.batch_size,
                'gpu_id': self.gpu_id
            }
            write_json(metrics, self.metrics_file)

            if not os.path.exists(self.metrics_file):
                self.status = EvaluationStatus.ERROR
                self.error_detail = (EvaluationError.FILE_NOT_FOUND, 'Save metrics file failed')
                logger.error('Save metrics file failed')
                return False

            return True
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
            logger.error(f"Unexpected error: {str(e)}")
            import traceback
            traceback.print_exc()
        return False

    def eval(self):
        success = self.eval_d2_segmentaion()
        if not success:
            logger.error(f"Evaluation failed: {self.error_detail}")
            return False
        return True

@task(bind=True, name='tasks.eval.detectron2.insseg')
def start_eval_detectron2_instance_segmentation_task(self, project_root, iou_thres, batch_size, gpu_id=0):
    evaluator = EvalDetectron2Segmentation(project_root, iou_thres, batch_size, gpu_id)
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
            return {'status': status.name, 'error_detail': None, 'metrics_file': evaluator.metrics_file}
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
        traceback.print_exc()
        return {'status': status.name, 'error_detail': str(e)}
    finally:
        if status == EvaluationStatus.PROCESSING:
            status = EvaluationStatus.REVOKED
            self.update_state(state=EvaluationStatus.REVOKED.name, meta={'exc_type': '', 'exc_message': '', 'status': status.name})
            logger.warning(f"Task status: {status.name}. Task was revoked.")
        
