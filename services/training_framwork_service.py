'''
Author: Will Cheng chengyong@pku.edu.cn
Date: 2024-07-24 21:47:08
LastEditors: Will Cheng chengyong@pku.edu.cn
LastEditTime: 2024-07-24 22:33:07
FilePath: /PoseidonAI-Server/services/training_framwork_service.py
Description: 

Copyright (c) 2024 by chengyong@pku.edu.cn, All Rights Reserved. 
'''
from app.models import TrainingFramework, DatasetFormat

class TrainingFrameworkService:
    @staticmethod
    def create_dataset_format(name, dataset_format_id, description=''):
        training_framework = TrainingFramework(name, dataset_format_id, description)
        result = training_framework.save()
        return result

    @staticmethod
    def get_training_framework(training_framework_id):
        training_framework = TrainingFramework.find_by_id(training_framework_id)
        dataset_format_id = training_framework['dataset_format_id']
        training_framework['dataset_format_id'] = str(dataset_format_id)
        training_framework['_id'] = str(training_framework['_id'])
        dataset_format = DatasetFormat.find_by_id(str(dataset_format_id)).to_dict() # class
        training_framework['dataset_format'] = dataset_format
        return training_framework

    @staticmethod
    def get_training_frameworks():
        training_frameworks = TrainingFramework.list_all()
        results = []
        for training_framework in training_frameworks:
            dataset_format_id = training_framework['dataset_format_id']
            training_framework['dataset_format_id'] = str(dataset_format_id)
            training_framework['_id'] = str(training_framework['_id'])
            dataset_format = DatasetFormat.find_by_id(str(dataset_format_id)).to_dict() # class
            training_framework['dataset_format'] = dataset_format
            results.append(training_framework)
        return results

    @staticmethod
    def delete_training_framework(training_framework_id):
        return TrainingFramework.delete(training_framework_id)
