'''
Author: Will Cheng chengyong@pku.edu.cn
Date: 2024-07-24 22:17:36
LastEditors: Will Cheng (will.cheng@efctw.com)
LastEditTime: 2024-08-09 16:33:19
FilePath: /PoseidonAI-Server/routes/training_task.py
Description: 

Copyright (c) 2024 by chengyong@pku.edu.cn, All Rights Reserved. 
'''

import os
import uuid
import traceback
from flask import Blueprint, request, jsonify
from app import redis_client
from routes.auth import jwt_required
from services.training_task_service import TrainingTaskService
from .utils import parse_immutable_dict

training_tasks_bp = Blueprint('training_tasks', __name__)

# 創建訓練任務的路由
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
    except Exception as e:
        response['code'] = 500
        response['msg'] = str(e)
        traceback.print_exc()
    return jsonify(response), 200

# 查詢訓練任務創建狀態的路由
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

# 列出用戶所有訓練任務的路由
@training_tasks_bp.route('/list', methods=['GET'])
@jwt_required
def list_tasks(user_id):
    response = {'code': 200, 'msg': 'ok', 'show_msg': 'ok', 'results': []}
    try:
        task_list = TrainingTaskService.get_tasks_by_user_id(user_id)
        response['results'] = task_list
    except Exception as e:
        response['code'] = 500
        response['msg'] = str(e)
    return jsonify(response), 200

# 查詢特定訓練任務詳情的路由
@training_tasks_bp.route('/list/<task_id>', methods=['GET'])
@jwt_required
def get_task_details(user_id, task_id):
    response = {'code': 200, 'msg': 'ok', 'show_msg': 'ok', 'results': []}
    try:
        training_task_id = redis_client.get(task_id)
        task_detail = TrainingTaskService.get_task_by_id(task_id)
        task_state = None
        if training_task_id:
            task_state = TrainingTaskService.task_training_status_by_object(training_task_id, task_detail)
        response['results'] = dict(
            task_detail=task_detail,
            task_state=task_state
        )
    except Exception as e:
        response['code'] = 500
        response['msg'] = str(e)
        traceback.print_exc()
    try:
        return jsonify(response), 200
    except:
        response['results']['task_state']['data'] = str(response['results']['task_state']['data'])
        return jsonify(response), 200

# 開始訓練任務的路由
@training_tasks_bp.route('/train/<task_id>', methods=['POST'])
@jwt_required
def train(user_id, task_id):
    response = {'code': 200, 'msg': 'ok', 'show_msg': 'ok', 'results': None}
    try:
        training_task_id = TrainingTaskService.train(task_id)
        redis_client.set(task_id, str(training_task_id))
        response['results'] = str(training_task_id)
    except Exception as e:
        response['code'] = 500
        response['msg'] = str(e)
        traceback.print_exc()
    return jsonify(response), 200

# 查詢訓練進度的路由
@training_tasks_bp.route('/train-status/<task_id>/<algo_name>/<framework_name>/<save_key>', methods=['GET'])
@jwt_required
def get_train_status(user_id, task_id, algo_name, framework_name, save_key):
    response = {'code': 200, 'msg': 'ok', 'show_msg': 'ok', 'results': None}
    try:
        training_task_id = redis_client.get(task_id)
        status = TrainingTaskService.task_training_status(training_task_id, algo_name, framework_name, save_key, user_id)
        response['results'] = status
        TrainingTaskService.update_training_task_status(task_id, status['state'])
        redis_client.set(f'{task_id}__status', str(status['state']))
    except Exception as e:
        response['code'] = 500
        response['msg'] = str(e)
        traceback.print_exc()
    return jsonify(response), 200

# 停止訓練任務的路由
@training_tasks_bp.route('/train/<task_id>/<algo_name>/<framework_name>', methods=['DELETE'])
@jwt_required
def stop_training(user_id, task_id, algo_name, framework_name):
    response = {'code': 200, 'msg': '已送出終止信號，任務可能不會立即停止', 'show_msg': 'ok', 'results': None}
    try:
        training_task_id = redis_client.get(task_id)
        TrainingTaskService.stop_training(training_task_id.decode(), algo_name, framework_name)
        TrainingTaskService.update_training_task_status(task_id, 'REVOKED')
    except Exception as e:
        response['code'] = 500
        response['msg'] = str(e)
        traceback.print_exc()
    return jsonify(response), 200
