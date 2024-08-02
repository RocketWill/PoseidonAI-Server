'''
Author: Will Cheng chengyong@pku.edu.cn
Date: 2024-07-26 11:43:42
LastEditors: Will Cheng (will.cheng@efctw.com)
LastEditTime: 2024-08-02 14:07:30
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


training_project_root = Config.TRAINING_PROJECT_FOLDER
training_configurations_root = Config.TRAINING_CONFIGS_FOLDER
dataset_root = Config.DATASET_RAW_FOLDER
os.makedirs(training_project_root, exist_ok=True)


class TrainingTaskService:
    @staticmethod
    def create_training_task(name, user_id, algorithm_id, dataset_id, training_configuration_id, model_name, epoch, gpu_id, save_key, description):
        algorithm_data = AlgorithmService.get_algorithm(algorithm_id)
        dataset_data = DatasetService.get_dataset(dataset_id).to_dict()
        training_configuration_data = TrainingConfigurationService.get_training_configuration(training_configuration_id)
        config_file = os.path.join(training_configurations_root, user_id, training_configuration_data['save_key'], 'args.json')
        training_framework_name = algorithm_data['training_framework']['name']
        dataset_format_name = algorithm_data['training_framework']['dataset_format']['name']
        dataset_dir = os.path.join(dataset_root, user_id, dataset_data['save_key'])
        if not os.path.exists(config_file):
            raise FileExistsError(config_file)
        print(dataset_dir)
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