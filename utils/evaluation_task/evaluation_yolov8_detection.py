'''
Author: Will Cheng chengyong@pku.edu.cn
Date: 2024-08-16 21:05:19
LastEditors: Will Cheng chengyong@pku.edu.cn
LastEditTime: 2024-08-17 17:04:33
FilePath: /PoseidonAI-Server/utils/evaluation_task/evaluation_yolov8_detection.py
Description: 

Copyright (c) 2024 by chengyong@pku.edu.cn, All Rights Reserved. 
'''
import os

import numpy as np
from ultralytics import YOLO

from utils.common import read_yaml, write_json


def eval_yolov8_detection(project_root, iou_thres, batch_size, gpu_id=None):
    training_dir = os.path.join(project_root, 'project', 'exp')
    weights_dir = os.path.join(training_dir, 'weights')
    args_file = os.path.join(training_dir, 'args.yaml')
    weights_file = os.path.join(weights_dir, 'best.pt')
    metrics_file = os.path.join(training_dir, 'evaluation.json')
    print(weights_file, args_file)
    assert os.path.exists(weights_file) and os.path.exists(args_file)
    cfg = read_yaml(args_file)
    cfg = {
        'batch': batch_size, 
        'conf': 0.01, 
        'iou': iou_thres,
        'data': cfg.get('data'),
        'project': None,
        'name': 'eval',
        'plots': False,
        'save': False,
        'device': 'cuda:{}'.format(gpu_id) if gpu_id else 'cpu',
    }


    model = YOLO(weights_file)
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

    write_json(data_to_save, metrics_file)
    if not os.path.exists(metrics_file):
        raise FileNotFoundError('Save metrics file failed')


if __name__ == '__main__':
    project_root = '/root/workspace/PoseidonAI-Server/data/projects/66a6eb9d4c0e8525ee44c787/53d02812-12f3-4a1a-8afd-ff18ca95a1fe'
    iou_thres = 0.2
    batch_size = 1
    gpu_id=None
    eval_yolov8_detection(project_root, iou_thres, batch_size, gpu_id)