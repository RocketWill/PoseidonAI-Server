import os
import glob

import torch
from pycocotools.coco import COCO
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from utils.evaluation_task.improved_coco_eval_tool import SelfEval
from utils.common import write_json

def register_dataset(training_dir):
    register_coco_instances("train_dataset", {}, os.path.join(training_dir, "data/train.json"), os.path.join(training_dir, "data/train"))
    register_coco_instances("val_dataset", {}, os.path.join(training_dir, "data/val.json"), os.path.join(training_dir, "data/val"))

def eval(cooc_val_gt_file, coco_val_dt_file, iou_thres):
    coco_gt = COCO(cooc_val_gt_file)
    coco_dt = coco_gt.loadRes(coco_val_dt_file)
    segm_eval = SelfEval(coco_gt, coco_dt, all_points=True, iou_type='segmentation', iou_thres=[iou_thres])
    segm_eval.evaluate()
    segm_eval.accumulate()
    segm_eval.summarize()
    metrics = segm_eval.get_curves_data_for_iou(iou_thres)
    return metrics

def eval_d2_segmentaion(project_root, iou_thres, batch_size, gpu_id=None):
    training_dir = os.path.join(project_root, 'project')
    metrics_file = os.path.join(training_dir, 'evaluation.json')
    weights_files = sorted(glob.glob(os.path.join(training_dir, '*.pth')))
    cooc_val_gt_file = os.path.join(training_dir, "data/val.json")
    assert len(weights_files) > 0
    weight_file = weights_files[-1]
    register_dataset(training_dir)
    cfg_file = os.path.join(project_root, 'cfg.yaml')
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.DATASETS.TRAIN = ("train_dataset",)
    cfg.DATASETS.TEST = ("val_dataset",)
    cfg.MODEL.WEIGHTS = weight_file
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0  # 设置阈值
    cfg.MODEL.DEVICE = "cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu"
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    predictor = DefaultPredictor(cfg)

    # 4. 执行推理并保存结果
    evaluator = COCOEvaluator("train", cfg, False, output_dir=training_dir)
    val_loader = build_detection_test_loader(cfg, "val_dataset")

    # inference_on_dataset将预测结果自动保存为COCO格式的JSON文件
    inference_on_dataset(predictor.model, val_loader, evaluator)
    
    coco_val_dt_file = os.path.join(training_dir, 'coco_instances_results.json')
    assert os.path.exists(coco_val_dt_file), 'Coco dt results not found'
    
    metrics = eval(cooc_val_gt_file, coco_val_dt_file, iou_thres)
    write_json(metrics, metrics_file)
    
    if not os.path.exists(metrics_file):
        raise FileNotFoundError('Save metrics file failed')