'''
Author: Will Cheng chengyong@pku.edu.cn
Date: 2024-07-26 11:43:42
LastEditors: Will Cheng (will.cheng@efctw.com)
LastEditTime: 2024-10-18 11:26:04
FilePath: /PoseidonAI-Server/services/training_task_service.py
Description: 

Copyright (c) 2024 by chengyong@pku.edu.cn, All Rights Reserved. 
'''

import os
import uuid
import shutil
import glob
import time
import signal
import random

import cv2
from flask import jsonify

from app import redis_client
from app.models import TrainingTask
from app.config import Config
from services.algorithm_service import AlgorithmService
from services.dataset_service import DatasetService
from services.training_configuration_service import TrainingConfigurationService
from services.detect_type_service import DetectTypeService
from utils.training_task.create_task import create_task
from utils.training_task.trainer import get_trainer, get_loss_parser, get_loss_file
from utils.evaluation_task import get_evaluator, get_metrics_file
from utils.visualize_val import get_visualizer, get_visualized_file
from utils.visualize_val.common import move_images
from utils.common import read_json
from utils.export_model import get_model_exporter

# 定義全局變量，用於存儲不同的項目路徑
training_project_root = Config.TRAINING_PROJECT_FOLDER
training_configurations_root = Config.TRAINING_CONFIGS_FOLDER
val_visualization_root = Config.VAL_VISUALIZATION_IMAGE_FOLDER
dataset_root = Config.DATASET_RAW_FOLDER
project_preview_root = Config.PROJECT_PRVIEW_IMAGE_FOLDER
model_export_root = Config.MODEL_EXPORT_FOLDER
os.makedirs(training_project_root, exist_ok=True)
os.makedirs(project_preview_root, exist_ok=True)
os.makedirs(val_visualization_root, exist_ok=True)
os.makedirs(model_export_root, exist_ok=True)

# 取得任務的狀態標籤
def get_task_status_tag(_id):
    status = redis_client.get(f'{_id}__status')
    if isinstance(status, bytes):
        status = status.decode()
    if not status:
        status = 'IDLE'
    return status

def get_last_three_levels(path):
    # 规范化路径，移除多余的分隔符
    path = os.path.normpath(path)
    # 分割路径为各个部分
    parts = path.split(os.sep)
    # 检查路径深度是否至少包含两级
    if len(parts) < 3:
        return path
    # 获取最后两部分
    last_two = parts[-3:]
    # 重新组合为所需的路径格式
    return os.path.join(*last_two)

# 處理預覽圖像
def handle_preview_image(image_file, output_file):
    preview_image = cv2.imread(image_file)
    original_height, original_width = preview_image.shape[:2]
    target_width = 256
    target_height = int((target_width / original_width) * original_height)
    preview_image = cv2.resize(preview_image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_file, preview_image)
    return output_file

# 格式化數據，轉換為字符串並附加算法信息
def format_data(data):
    data['_id'] = str(data['_id'])
    data['user_id'] = str(data['user_id'])
    data['dataset_id'] = str(data['dataset_id'])
    data['algorithm_id'] = str(data['algorithm_id'])
    data['training_configuration_id'] = str(data['training_configuration_id'])
    algorithm_data = AlgorithmService.get_algorithm(data['algorithm_id'])
    data['algorithm'] = algorithm_data
    return data

# 訓練任務服務類
class TrainingTaskService:
    
    @staticmethod
    def create_training_task(name, user_id, algorithm_id, dataset_id, training_configuration_id, model_name, epochs, val_ratio, gpu_id, save_key, description):
        algorithm_data = AlgorithmService.get_algorithm(algorithm_id)
        dataset_data = DatasetService.get_dataset(dataset_id).to_dict()
        detect_type_id = str(dataset_data['detect_type_id'])
        detect_type_data = DetectTypeService.get_detect_type(detect_type_id)
        detect_type = detect_type_data['tag_name']

        training_configuration_data = TrainingConfigurationService.get_training_configuration(training_configuration_id)
        config_file = os.path.join(training_configurations_root, user_id, training_configuration_data['save_key'], 'args.json')
        training_framework_name = algorithm_data['training_framework']['name']
        dataset_format_name = algorithm_data['training_framework']['dataset_format']['name']
        dataset_dir = os.path.join(dataset_root, user_id, dataset_data['save_key'])
        project_dir = os.path.join(training_project_root, user_id, save_key)
        os.makedirs(project_dir, exist_ok=True)
        
        if not os.path.exists(config_file):
            raise FileExistsError(config_file)
        
        # 保存預覽圖像
        if detect_type == 'classify':
            image_file = random.choice(glob.glob(os.path.join(dataset_dir, 'dataset', '*', '*')))
        else:
            image_file = random.choice(glob.glob(os.path.join(dataset_dir, 'images', '*')))
        output_dir = os.path.join(project_preview_root, user_id, save_key)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'preview.jpg')
        handle_preview_image(image_file, output_file)

        # 創建異步任務
        task = create_task.apply_async(args=[
            training_framework_name, config_file, epochs, gpu_id, val_ratio, dataset_dir,
            model_name, project_dir,
            name, user_id, algorithm_id, dataset_id, training_configuration_id, save_key, description, detect_type
        ])
        return task.id
    
    @staticmethod
    def task_create_status(task_id):
        task = create_task.AsyncResult(task_id)
        if task.state == 'PENDING':
            response = {
                'state': task.state,
                'current': 0,
                'total': 1,
                'description': 'Pending',
                'steps': []
            }
        elif task.state in ['PROCESSING', 'PROGRESS']:
            response = {
                'state': task.state,
                'current': task.info.get('current', 0),
                'total': task.info.get('total', 1),
                'description': task.info.get('description', ''),
                'steps': task.info.get('steps', [])
            }
        elif task.state == 'SUCCESS':
            response = {
                'state': task.state,
                'result': task.result,
                'steps': task.info.get('steps', [])
            }
        else:
            # 處理任務失敗的情況
            response = {
                'state': task.state,
                'current': 1,
                'total': 1,
                'description': str(task.info),  # 失敗時的異常信息
                'steps': []
            }
        return response
    
    @staticmethod
    def get_tasks_by_user_id(user_id):
        task_list = TrainingTask.find_by_user(user_id)
        task_list = [format_data(d) for d in task_list]
        return task_list

    @staticmethod
    def get_task_by_id(task_id):
        task = TrainingTask.find_by_id(task_id)
        task = format_data(task)
        return task
    
    @staticmethod
    def delete_task_by_id(task_id, user_id):
        task = TrainingTask.find_by_id(task_id)
        task = format_data(task)
        if user_id != task['user_id']:
            raise ValueError('Current user cannot delete the project.')
        save_key = task['save_key']
        project_root = os.path.join(training_project_root, user_id, save_key)
        vis_root = os.path.join(val_visualization_root, user_id, save_key)
        project_preview_dir = os.path.join(project_preview_root, user_id, save_key)
        project_export_root = os.path.join(model_export_root, user_id, save_key)
        if os.path.exists(project_root):
            shutil.rmtree(project_root)
        if os.path.exists(vis_root):
            shutil.rmtree(vis_root)
        if os.path.exists(vis_root):
            shutil.rmtree(vis_root)
        if os.path.exists(project_preview_dir):
            shutil.rmtree(project_preview_dir)
        if os.path.exists(project_export_root):
            shutil.rmtree(project_export_root)
        return TrainingTask.delete(task_id)

    @staticmethod
    def train(task_id):
        task_data = TrainingTaskService.get_task_by_id(task_id)
        user_id = task_data['user_id']
        save_key = task_data['save_key']
        project_dir = os.path.join(training_project_root, user_id, save_key)
        algo_name = task_data['algorithm']['name'].replace(' ', '')
        framework_name = task_data['algorithm']['training_framework']['name']
        trainer = get_trainer(algo_name, framework_name)
        training_task = trainer.apply_async(args=[project_dir, task_id])
        return training_task.id
    
    @staticmethod
    def get_loss_result(task_data):
        user_id = task_data['user_id']
        save_key = task_data['save_key']
        algo_name = task_data['algorithm']['name'].replace(" ", "")
        framework_name = task_data['algorithm']['training_framework']['name']
        loss_file = get_loss_file(algo_name, framework_name, training_project_root, user_id, save_key)
        loss_parser = get_loss_parser(algo_name, framework_name)
        return {
                'state': 'SUCCESS',
                'data': {
                    'error_detail': None,
                    'results': loss_parser(loss_file),
                    'status': 'SUCCESS'
                }
            }


    @staticmethod
    def task_training_status(training_task_id, algo_name, framework_name, save_key, user_id):
        # loss_file = os.path.join(training_project_root, user_id, save_key, 'project', 'exp', 'results.csv')
        loss_file = get_loss_file(algo_name, framework_name, training_project_root, user_id, save_key)
        trainer = get_trainer(algo_name, framework_name)
        loss_parser = get_loss_parser(algo_name, framework_name)
        task = trainer.AsyncResult(training_task_id)

        if task.state == 'PENDING':
            response = {
                'state': task.state,
                'data': {
                    'error_detail': None,
                    'results': None,
                    'status': 'PENDING'
                }
            }
        elif task.state == 'PROCESSING':
            response = {
                'state': task.state,
                'data': {
                    'error_detail': None,
                    'results': loss_parser(loss_file),
                    'status': 'PROCESSING'
                }
            }
        elif task.state == 'SUCCESS':
            response = {
                'state': task.state,
                'data': {
                    'error_detail': None,
                    'results': loss_parser(loss_file),
                    'status': 'SUCCESS'
                }
            }
        else:
            # 處理任務失敗的情況
            try:
                response = {
                    'state': task.state,
                    'data': task.result
                }
                jsonify(response)
                print(response)
            except Exception:
                response = {
                    'state': task.state,
                    'data': {
                        'error_detail': str(task.result),
                        'results': None,
                        'status': 'ERROR'
                    }
                }
        return response
    
    @staticmethod
    def stop_training(training_task_id, algo_name, framework_name):
        trainer = get_trainer(algo_name, framework_name)
        task = trainer.AsyncResult(training_task_id)
        task.revoke(terminate=True, signal='SIGKILL')
        if task.state == 'PROCESSING' and hasattr(task, 'worker_pid'):
            os.kill(task.worker_pid, signal.SIGKILL)

    @staticmethod
    def task_training_status_by_object(training_task_id, task_data):
        algo_name = task_data['algorithm']['name'].replace(" ", "")
        framework_name = task_data['algorithm']['training_framework']['name']
        save_key = task_data['save_key']
        user_id = task_data['user_id']
        return TrainingTaskService.task_training_status(training_task_id, algo_name, framework_name, save_key, user_id)
    
    @staticmethod
    def update_training_task_status(task_id, status):
        return TrainingTask.update_status(task_id, status)
    
    @staticmethod
    def task_evaluation(user_id, task_id, batch_size, iou_thres, gpu_id):
        task_data = TrainingTaskService.get_task_by_id(task_id)
        algo_name = task_data['algorithm']['name'].replace(" ", "")
        framework_name = task_data['algorithm']['training_framework']['name']
        save_key = task_data['save_key']
        project_root = os.path.join(training_project_root, user_id, save_key)        
        evaluator = get_evaluator(algo_name, framework_name)
        # self, project_root, iou_thres, batch_size, gpu_id=0):
        eval_task = evaluator.apply_async(args=[project_root, iou_thres, batch_size, gpu_id])
        return eval_task.id
    
    @staticmethod
    def task_evaluation_status(eval_task_id, algo_name, framework_name):
        evaluator = get_evaluator(algo_name, framework_name)
        task = evaluator.AsyncResult(eval_task_id)

        if task.state == 'PENDING':
            response = {
                'state': task.state,
                'data': {
                    'error_detail': None,
                    'results': None,
                    'status': 'PENDING'
                }
            }
        elif task.state == 'PROCESSING':
            response = {
                'state': task.state,
                'data': {
                    'error_detail': None,
                    'results': None,
                    'status': 'PROCESSING'
                }
            }
        elif task.state == 'SUCCESS':
            try: 
                metrics_file = task.result['metrics_file']
                response = {
                    'state': task.state,
                    'data': {
                        'error_detail': None,
                        'results': read_json(metrics_file),
                        'status': 'SUCCESS'
                    }
                }
            except Exception as e:
                response = {
                    'state': task.result.status,
                    'data': {
                        'error_detail': task.result.error_detail,
                        'results': read_json(metrics_file),
                        'status': 'ERROR'
                    }
                }
        else:
            # 處理任務失敗的情況
            try:
                response = {
                    'state': task.state,
                    'data': task.result
                }
                jsonify(response)
            except Exception:
                response = {
                    'state': task.state,
                    'data': {
                        'error_detail': str(task.result),
                        'results': None,
                        'status': 'ERROR'
                    }
                }
        return response
    
    @staticmethod
    def task_evaluation_result(algo_name, framework_name, user_id, save_key):
        metrics_file = get_metrics_file(algo_name, framework_name, training_project_root, user_id, save_key)
        if os.path.exists(metrics_file):
            return read_json(metrics_file)
        return False
    
    @staticmethod
    def task_visualization(user_id, task_id, iou_thres, conf=0.01):
        task_data = TrainingTaskService.get_task_by_id(task_id)
        algo_name = task_data['algorithm']['name'].replace(" ", "")
        detect_type = task_data['algorithm']['detect_type']['tag_name']
        framework_name = task_data['algorithm']['training_framework']['name']
        save_key = task_data['save_key']
        project_root = os.path.join(training_project_root, user_id, save_key)
        val_image_dir = os.path.join(project_root, 'data', 'images', 'val')
        if detect_type == 'classify':
            val_image_dir = os.path.join(project_root, 'data', 'val')
        if 'yolo' not in framework_name.lower():
            val_image_dir = os.path.join(project_root, 'data', 'val')
        static_val_image_dir = os.path.join(val_visualization_root, user_id, save_key)
        visualizer = get_visualizer(algo_name, framework_name)
        vis_task = visualizer.apply_async(args=[project_root, iou_thres, conf])
        move_images(val_image_dir, static_val_image_dir, detect_type)
        return vis_task.id
    
    @staticmethod
    def task_visualization_status(vis_task_id, algo_name, framework_name):
        visualizer = get_visualizer(algo_name, framework_name)
        task = visualizer.AsyncResult(vis_task_id)

        if task.state == 'PENDING':
            response = {
                'state': task.state,
                'data': {
                    'error_detail': None,
                    'results': None,
                    'status': 'PENDING'
                }
            }
        elif task.state == 'PROCESSING':
            response = {
                'state': task.state,
                'data': {
                    'error_detail': None,
                    'results': None,
                    'status': 'PROCESSING'
                }
            }
        elif task.state == 'SUCCESS':
            try: 
                preds_file = task.result['preds_file']
                response = {
                    'state': task.state,
                    'data': {
                        'error_detail': None,
                        'results': read_json(preds_file),
                        'status': 'SUCCESS'
                    }
                }
            except Exception as e:
                response = {
                    'state': task.result.status,
                    'data': {
                        'error_detail': task.result.error_detail,
                        'results': None,
                        'status': 'ERROR'
                    }
                }
        else:
            # 處理任務失敗的情況
            try:
                response = {
                    'state': task.state,
                    'data': task.result
                }
                jsonify(response)
            except Exception:
                response = {
                    'state': task.state,
                    'data': {
                        'error_detail': str(task.result),
                        'results': None,
                        'status': 'ERROR'
                    }
                }
        return response
    
    @staticmethod
    def task_visualization_result(algo_name, framework_name, user_id, save_key):
        visualization_file = get_visualized_file(algo_name, framework_name, training_project_root, user_id, save_key)
        if os.path.exists(visualization_file):
            return read_json(visualization_file)
        return False
    
    @staticmethod
    def task_export_model(task_id, format, filename, content):
        # task id is db id
        filename = filename if filename else 'model'
        task_data = TrainingTaskService.get_task_by_id(task_id)
        user_id = task_data['user_id']
        save_key = task_data['save_key']
        project_dir = os.path.join(training_project_root, user_id, save_key)
        algo_name = task_data['algorithm']['name'].replace(' ', '')
        framework_name = task_data['algorithm']['training_framework']['name']
        model_exporter = get_model_exporter(algo_name, framework_name)
        output_dir = os.path.join(model_export_root, save_key)
        os.makedirs(output_dir, exist_ok=True)
        output_zip_file = os.path.join(output_dir, filename + '.zip')
        model_exporter_task = model_exporter.apply_async(args=[project_dir, output_zip_file, format, content])
        return model_exporter_task.id
    
    @staticmethod
    def task_export_status(export_task_id, algo_name, framework_name):
        model_exporter = get_model_exporter(algo_name, framework_name)
        task = model_exporter.AsyncResult(export_task_id)

        if task.state == 'PENDING':
            response = {
                'state': task.state,
                'data': {
                    'error_detail': None,
                    'results': None,
                    'status': 'PENDING'
                }
            }
        elif task.state == 'PROCESSING':
            response = {
                'state': task.state,
                'data': {
                    'error_detail': None,
                    'results': None,
                    'status': 'PROCESSING'
                }
            }
        elif task.state == 'SUCCESS':
            try: 
                output_file = task.result['output_file']
                response = {
                    'state': task.state,
                    'data': {
                        'error_detail': None,
                        'results': get_last_three_levels(output_file),
                        'status': 'SUCCESS'
                    }
                }
            except Exception as e:
                print(e)
                response = {
                    'state': task.result['status'],
                    'data': {
                        'error_detail': task.result['error_detail'],
                        'results': get_last_three_levels(output_file),
                        'status': 'ERROR'
                    }
                }
        else:
            # 處理任務失敗的情況
            try:
                response = {
                    'state': task.state,
                    'data': task.result
                }
                jsonify(response)
            except Exception:
                response = {
                    'state': task.state,
                    'data': {
                        'error_detail': str(task.result),
                        'results': None,
                        'status': 'ERROR'
                    }
                }
        return response

# 測試訓練任務服務的功能
if __name__ == '__main__':
    name = '123'
    user_id = '66a6eb9d4c0e8525ee44c787'
    task_id = '66b2cacde5e069ac6583640e'
    x = TrainingTaskService.train(task_id)
    import pprint
    pprint.pprint(x)
