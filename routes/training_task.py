'''
Author: Will Cheng chengyong@pku.edu.cn
Date: 2024-07-24 22:17:36
LastEditors: Will Cheng chengyong@pku.edu.cn
LastEditTime: 2024-08-25 16:52:27
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
        if task_detail['status'] == 'SUCCESS':
            task_state = TrainingTaskService.get_loss_result(task_detail)
            
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
        training_task_id = training_task_id.decode()
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

@training_tasks_bp.route('/evaluation/<task_id>', methods=['POST'])
@jwt_required
def task_evaluation(user_id, task_id):
    # task id is Task model id
    response = {'code': 200, 'msg': 'ok', 'show_msg': 'ok', 'results': None}
    try:
        data = request.form.to_dict()
        data = parse_immutable_dict(data)
        batch_size = data['batchSize']
        iou_thres = data['iou']
        gpu_id = data['gpu']
        eval_id = TrainingTaskService.task_evaluation(user_id, task_id, batch_size, iou_thres, gpu_id)
        redis_client.set('eval_'.format(task_id), str(eval_id))
        response['results'] = eval_id
    except Exception as e:
        response['code'] = 500
        response['msg'] = str(e)
        traceback.print_exc()
    return jsonify(response), 200

# 查詢eval進度的路由
@training_tasks_bp.route('/eval-task-status/<task_id>/<algo_name>/<framework_name>', methods=['GET'])
@jwt_required
def get_eval_status(user_id, task_id, algo_name, framework_name):
    # task_id is Task model id
    response = {'code': 200, 'msg': 'ok', 'show_msg': 'ok', 'results': None}
    try:
        eval_task_id = redis_client.get('eval_'.format(task_id))
        eval_task_id = eval_task_id.decode()
        status = TrainingTaskService.task_evaluation_status(eval_task_id, algo_name, framework_name)
        response['results'] = status
    except Exception as e:
        response['code'] = 500
        response['msg'] = str(e)
        traceback.print_exc()
    return jsonify(response), 200


@training_tasks_bp.route('/evaluation-results/<task_id>', methods=['GET'])
@jwt_required
def get_eval_results(user_id, task_id):
    response = {'code': 200, 'msg': 'ok', 'show_msg': 'ok', 'results': None}
    try:
        task_data = TrainingTaskService.get_task_by_id(task_id)
        algo_name = task_data['algorithm']['name'].replace(" ", "")
        framework_name = task_data['algorithm']['training_framework']['name']
        save_key = task_data['save_key']
        user_id = task_data['user_id']
        metrics_data = TrainingTaskService.task_evaluation_result(algo_name, framework_name, user_id, save_key)
        response['results'] = metrics_data
    except Exception as e:
        response['code'] = 500
        response['msg'] = str(e)
        traceback.print_exc()
    return jsonify(response), 200


@training_tasks_bp.route('/visualization/<task_id>', methods=['POST'])
@jwt_required
def task_visualization(user_id, task_id):
    # task id is Task model id
    response = {'code': 200, 'msg': 'ok', 'show_msg': 'ok', 'results': None}
    try:
        data = request.form.to_dict()
        data = parse_immutable_dict(data)
        iou_thres = data['iou']
        conf = data['conf']
        vis_id = TrainingTaskService.task_visualization(user_id, task_id, iou_thres, conf)
        redis_client.set('vis_'.format(task_id), str(vis_id))
        response['results'] = vis_id
    except Exception as e:
        response['code'] = 500
        response['msg'] = str(e)
        traceback.print_exc()
    return jsonify(response), 200

# 查詢vis進度的路由
@training_tasks_bp.route('/vis-task-status/<task_id>/<algo_name>/<framework_name>', methods=['GET'])
@jwt_required
def get_vis_status(user_id, task_id, algo_name, framework_name):
    # task_id is Task model id
    response = {'code': 200, 'msg': 'ok', 'show_msg': 'ok', 'results': None}
    try:
        eval_task_id = redis_client.get('vis_'.format(task_id))
        eval_task_id = eval_task_id.decode()
        status = TrainingTaskService.task_visualization_status(eval_task_id, algo_name, framework_name)
        response['results'] = status
    except Exception as e:
        response['code'] = 500
        response['msg'] = str(e)
        traceback.print_exc()
    return jsonify(response), 200

@training_tasks_bp.route('/visualization-results/<task_id>', methods=['GET'])
@jwt_required
def get_vis_results(user_id, task_id):
    response = {'code': 200, 'msg': 'ok', 'show_msg': 'ok', 'results': None}
    try:
        task_data = TrainingTaskService.get_task_by_id(task_id)
        algo_name = task_data['algorithm']['name'].replace(" ", "")
        framework_name = task_data['algorithm']['training_framework']['name']
        save_key = task_data['save_key']
        user_id = task_data['user_id']
        visualization_data = TrainingTaskService.task_visualization_result(algo_name, framework_name, user_id, save_key)
        response['results'] = visualization_data
    except Exception as e:
        response['code'] = 500
        response['msg'] = str(e)
        traceback.print_exc()
    return jsonify(response), 200