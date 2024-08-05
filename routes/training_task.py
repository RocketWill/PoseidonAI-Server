'''
Author: Will Cheng chengyong@pku.edu.cn
Date: 2024-07-24 22:17:36
LastEditors: Will Cheng (will.cheng@efctw.com)
LastEditTime: 2024-08-05 14:25:52
FilePath: /PoseidonAI-Server/routes/training_task.py
Description: 

Copyright (c) 2024 by chengyong@pku.edu.cn, All Rights Reserved. 
'''
import os
from bson import ObjectId
import traceback
import uuid

from flask import Blueprint, request, jsonify

from .utils import parse_immutable_dict
from routes.auth import jwt_required
from services.training_task_service import TrainingTaskService

training_tasks_bp = Blueprint('training_tasks', __name__)

# {'name': 'Training-Task__strong-shark', 'description': '', 'algorithm': '66a74cf1c96399a9ff514d6f', 'dataset': '66b035f2687d0c81e9845eca', 'config': '66a75082087af8b1a7a0b761', 'gpu': 0, 'model': 'yolov8n.pt', 'epoch': 30, 'trainValRatio': 0.1}

@training_tasks_bp.route('/create', methods=['POST'])
@jwt_required
def create_training_task(user_id):
    response = {'code': 200, 'msg': 'ok', 'show_msg': 'ok', 'results': None}
    try:
        data = request.form.to_dict()
        data = parse_immutable_dict(data)
        name = data['name']
        description = data['description']
        algorithm_id = data['algorithm']
        dataset_id = data['dataset']
        training_configuration_id = data['config']
        gpu_id = data['gpu']
        model_name = data['model']
        epochs = data['epoch']
        val_ratio = data['trainValRatio']
        save_key = str(uuid.uuid4())
        task_id = TrainingTaskService.create_training_task(name, user_id, algorithm_id, dataset_id, training_configuration_id, model_name, epochs, val_ratio, gpu_id, save_key, description)
        response['results'] = task_id
        print(task_id)
    except Exception as e:
        response['code'] = 500
        response['msg'] = str(e)
        traceback.print_exc()
    return jsonify(response), 200

@training_tasks_bp.route('/task-create-status/<task_id>', methods=['GET'])
def get_task_create_status(task_id):
    response = {'code': 200, 'msg': 'ok', 'show_msg': 'ok', 'results': None}
    try:
        status = TrainingTaskService.task_create_status(task_id)
        response['results'] = status
    except Exception as e:
        response['code'] = 500
        response['msg'] = str(e)
    return jsonify(response), 200