'''
Author: Will Cheng chengyong@pku.edu.cn
Date: 2024-07-26 11:43:42
LastEditors: Will Cheng (will.cheng@efctw.com)
LastEditTime: 2024-08-05 14:33:10
FilePath: /PoseidonAI-Server/services/training_task_service.py
Description: 

Copyright (c) 2024 by chengyong@pku.edu.cn, All Rights Reserved. 
'''
import os
import uuid
import shutil

from app.models import TrainingTask
from app.config import Config
from services.algorithm_service import AlgorithmService
from services.dataset_service import DatasetService
from services.training_configuration_service import TrainingConfigurationService
from utils.training_task.create_task import create_task

training_project_root = Config.TRAINING_PROJECT_FOLDER
training_configurations_root = Config.TRAINING_CONFIGS_FOLDER
dataset_root = Config.DATASET_RAW_FOLDER
os.makedirs(training_project_root, exist_ok=True)

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
        # [train_num, val_num] = create_task(training_framework_name, config_file, epochs, gpu_id, val_ratio, dataset_dir, model_name, project_dir)
        # task = TrainingTask(name, user_id, algorithm_id, dataset_id, training_configuration_id, model_name, epochs, val_ratio, gpu_id, save_key, [train_num, val_num], description)
        # return task.save()
        # print('='*50)
        # print(dataset_data)
        # print('='*50)
        # print(training_configuration_data)
        # print('='*50)
    # @staticmethod
    # def get_training_configuration(training_configuration_id):
    #     training_config = TrainingFramework.find_by_id(training_configuration_id)
    #     return format_data(training_config)

    # @staticmethod
    # def get_training_configurations():
    #     training_configs = TrainingConfiguration.list_all()
    #     return [format_data(d) for d in training_configs]
    
    # @staticmethod
    # def get_training_configurations_by_user(user_id):
    #     training_configs = TrainingConfiguration.find_by_user(user_id)
    #     return [format_data(d) for d in training_configs]

    # @staticmethod
    # def delete_training_configuration(training_configuration_id):
    #     ...

if __name__ == '__main__':
    name = '123'
    user_id = '66a6eb9d4c0e8525ee44c787'
    algorithm_id = '66a74cf1c96399a9ff514d6f'
    tr_id = '66a75082087af8b1a7a0b761'
    data_id = '66aaf5faaff932a6d4c715e6'
    x = TrainingTaskService.create_training_task(name, user_id, algorithm_id, data_id, tr_id, 2,3,4,5,6)