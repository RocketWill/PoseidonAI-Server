'''
Author: Will Cheng chengyong@pku.edu.cn
Date: 2024-07-26 11:43:42
LastEditors: Will Cheng (will.cheng@efctw.com)
LastEditTime: 2024-08-09 16:31:46
FilePath: /PoseidonAI-Server/services/training_task_service.py
Description: 

Copyright (c) 2024 by chengyong@pku.edu.cn, All Rights Reserved. 
'''

import os
import uuid
import shutil
import glob
import time

import cv2
from flask import jsonify

from app import redis_client
from app.models import TrainingTask
from app.config import Config
from services.algorithm_service import AlgorithmService
from services.dataset_service import DatasetService
from services.training_configuration_service import TrainingConfigurationService
from utils.training_task.create_task import create_task
from utils.training_task.trainer import get_trainer, get_loss_parser, get_loss_file

# 定義全局變量，用於存儲不同的項目路徑
training_project_root = Config.TRAINING_PROJECT_FOLDER
training_configurations_root = Config.TRAINING_CONFIGS_FOLDER
dataset_root = Config.DATASET_RAW_FOLDER
project_preview_root = Config.PROJECT_PRVIEW_IMAGE_FOLDER
os.makedirs(training_project_root, exist_ok=True)
os.makedirs(project_preview_root, exist_ok=True)

# 取得任務的狀態標籤
def get_task_status_tag(_id):
    status = redis_client.get(f'{_id}__status')
    if isinstance(status, bytes):
        status = status.decode()
    if not status:
        status = 'IDLE'
    return status

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
        image_file = glob.glob(os.path.join(dataset_dir, 'images', '*'))[0]
        output_dir = os.path.join(project_preview_root, user_id, save_key)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'preview.jpg')
        handle_preview_image(image_file, output_file)

        # 創建異步任務
        task = create_task.apply_async(args=[
            training_framework_name, config_file, epochs, gpu_id, val_ratio, dataset_dir,
            model_name, project_dir,
            name, user_id, algorithm_id, dataset_id, training_configuration_id, save_key, description
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

# 測試訓練任務服務的功能
if __name__ == '__main__':
    name = '123'
    user_id = '66a6eb9d4c0e8525ee44c787'
    task_id = '66b2cacde5e069ac6583640e'
    x = TrainingTaskService.train(task_id)
    import pprint
    pprint.pprint(x)
