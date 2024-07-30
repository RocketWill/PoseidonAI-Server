'''
Author: Will Cheng (will.cheng@efctw.com)
Date: 2024-07-30 08:36:35
LastEditors: Will Cheng (will.cheng@efctw.com)
LastEditTime: 2024-07-30 09:39:55
FilePath: /PoseidonAI-Server/services/algorithm_service.py
'''
from app.models import Algorithm
from .detect_type_service import DetectTypeService
from .training_framwork_service import TrainingFrameworkService


def get_detect_type_data(detect_type_id):
    return DetectTypeService.get_detect_type(detect_type_id)

def get_training_framework_data(training_framework_id):
    return TrainingFrameworkService.get_training_framework(training_framework_id)

def get_all_data(detect_type_id, training_framework_id):
    return dict(
        detect_type=get_detect_type_data(detect_type_id),
        training_framework=get_training_framework_data(training_framework_id)
    )

def handle_object_ids(algorithm):
    algorithm['_id'] = str(algorithm['_id'])
    del algorithm['detect_type_id']
    del algorithm['training_framework_id']

class AlgorithmService:

    @staticmethod
    def get_algorithm(algorithm_id):
        algorithm = Algorithm.find_by_id(algorithm_id)
        detect_type_id = algorithm['detect_type_id']
        training_framework_id = algorithm['training_framework_id']
        algorithm.update(get_all_data(detect_type_id, training_framework_id))
        handle_object_ids(algorithm)
        return algorithm

    @staticmethod
    def get_algorithms():
        results = []
        for algorithm in [d.to_dict() for d in Algorithm.list_all()]:
            detect_type_id = algorithm['detect_type_id']
            training_framework_id = algorithm['training_framework_id']
            algorithm.update(get_all_data(detect_type_id, training_framework_id))
            handle_object_ids(algorithm)
            results.append(algorithm)
        return results
