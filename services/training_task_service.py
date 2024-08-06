'''
Author: Will Cheng chengyong@pku.edu.cn
Date: 2024-07-26 11:43:42
LastEditors: Will Cheng (will.cheng@efctw.com)
LastEditTime: 2024-08-06 17:03:02
FilePath: /PoseidonAI-Server/services/training_task_service.py
Description: 

Copyright (c) 2024 by chengyong@pku.edu.cn, All Rights Reserved. 
'''
import os
import uuid
import shutil
import glob

import cv2

from app.models import TrainingTask
from app.config import Config
from services.algorithm_service import AlgorithmService
from services.dataset_service import DatasetService
from services.training_configuration_service import TrainingConfigurationService
from utils.training_task.create_task import create_task

training_project_root = Config.TRAINING_PROJECT_FOLDER
training_configurations_root = Config.TRAINING_CONFIGS_FOLDER
dataset_root = Config.DATASET_RAW_FOLDER
project_preview_root = Config.PROJECT_PRVIEW_IMAGE_FOLDER
os.makedirs(training_project_root, exist_ok=True)
os.makedirs(project_preview_root, exist_ok=True)


def handle_preview_image(image_file, output_file):
    preview_image = cv2.imread(image_file)
    original_height, original_width = preview_image.shape[:2]
    target_width = 256
    target_height = int((target_width / original_width) * original_height)
    preview_image = cv2.resize(preview_image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_file, preview_image)

    return output_file

def format_data(data):
    data['_id'] = str(data['_id'])
    data['user_id'] = str(data['user_id'])
    data['dataset_id'] = str(data['dataset_id'])
    data['algorithm_id'] = str(data['algorithm_id'])
    data['training_configuration_id'] = str(data['training_configuration_id'])
    algorithm_data = AlgorithmService.get_algorithm(data['algorithm_id'])
    data['algorithm'] = algorithm_data
    return data

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
        
        # save preview image
        image_file = glob.glob(os.path.join(dataset_dir, 'images', '*'))[0]
        output_dir = os.path.join(project_preview_root, user_id, save_key)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'preview.jpg')
        handle_preview_image(image_file, output_file)

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
        elif task.state == 'PROGRESS':
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
            # 处理任务失败的情况
            response = {
                'state': task.state,
                'current': 1,
                'total': 1,
                'description': str(task.info),  # 失败时的异常信息
                'steps': []
            }
        return response
    
    @staticmethod
    def get_tasks_by_user_id(user_id):
        task_list = TrainingTask.find_by_user(user_id)
        return [format_data(d) for d in task_list]

    @staticmethod
    def get_task_by_id(task_id):
        task = TrainingTask.find_by_id(task_id)
        return(format_data(task))


if __name__ == '__main__':
    name = '123'
    user_id = '66a6eb9d4c0e8525ee44c787'
    task_id = '66b1cae649e4f0ffffe3ab1a'
    # algorithm_id = '66a74cf1c96399a9ff514d6f'
    # tr_id = '66a75082087af8b1a7a0b761'
    # data_id = '66aaf5faaff932a6d4c715e6'
    # x = TrainingTaskService.create_training_task(name, user_id, algorithm_id, data_id, tr_id, 2,3,4,5,6)
    x = TrainingTaskService.get_task_by_id(task_id)
    import pprint

    pprint.pprint(x)