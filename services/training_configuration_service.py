'''
Author: Will Cheng chengyong@pku.edu.cn
Date: 2024-07-26 11:43:42
LastEditors: Will Cheng chengyong@pku.edu.cn
LastEditTime: 2024-07-30 20:35:03
FilePath: /PoseidonAI-Server/services/training_configuration_service.py
Description: 

Copyright (c) 2024 by chengyong@pku.edu.cn, All Rights Reserved. 
'''
import os
import uuid
import shutil

from app.models import TrainingFramework, TrainingConfiguration, DatasetFormat
from app.config import Config
from services.training_framwork_service import TrainingFrameworkService
from utils.training_configuration import save_config_file

training_configurations_root = Config.TRAINING_CONFIGS_FOLDER
os.makedirs(training_configurations_root, exist_ok=True)

def format_data(training_config):
    training_framework_id = str(training_config['training_framework_id'])
    training_framework = TrainingFramework.find_by_id(training_framework_id)
    training_config['_id'] = str(training_config['_id'])
    training_config['user_id'] = str(training_config['user_id'])
    training_config['training_framework_id'] = str(training_config['training_framework_id'])
    training_framework['_id'] = str(training_framework['_id'])
    training_framework['dataset_format_id'] = str(training_framework['dataset_format_id'])
    dataset_format = DatasetFormat.find_by_id(training_framework['dataset_format_id']).to_dict()
    training_framework['dataset_format'] = dataset_format
    training_config['training_framework'] = training_framework
    
    return training_config

class TrainingConfigurationService:
    @staticmethod
    def create_configuration(name, user_id, training_framework_id, description, data):
        save_key = str(uuid.uuid4())
        training_framework_data = TrainingFrameworkService.get_training_framework(training_framework_id)
        framework_name = training_framework_data['name']
        output_dir = os.path.join(training_configurations_root, user_id, save_key)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'args.json')
        args_data = save_config_file(framework_name, data, output_file)
        
        data = TrainingConfiguration(name, user_id, training_framework_id, description=description, args_data=args_data, save_key=save_key)
        result = data.save()
        return result

    @staticmethod
    def get_training_configuration(training_configuration_id):
        training_config = TrainingConfiguration.find_by_id(training_configuration_id)
        return format_data(training_config)

    @staticmethod
    def get_training_configurations():
        training_configs = TrainingConfiguration.list_all()
        return [format_data(d) for d in training_configs]
    
    @staticmethod
    def get_training_configurations_by_user(user_id):
        training_configs = TrainingConfiguration.find_by_user(user_id)
        return [format_data(d) for d in training_configs]

    @staticmethod
    def delete_training_configuration(training_configuration_id):
        training_config = TrainingConfiguration.find_by_id(training_configuration_id)
        save_key = training_config['save_key']
        user_id = str(training_config['user_id'])
        config_dir = os.path.join(training_configurations_root, user_id, save_key)
        shutil.rmtree(config_dir)
        return TrainingConfiguration.delete(training_configuration_id)

    @staticmethod
    def get_training_configurations_by_user_and_framework_id(user_id, training_framework_id):
        training_configs = TrainingConfiguration\
            .find_by_user_and_training_framework(
                user_id, training_framework_id)
        return [format_data(d) for d in training_configs]
