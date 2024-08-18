'''
Author: Will Cheng chengyong@pku.edu.cn
Date: 2024-07-24 22:17:36
LastEditors: Will Cheng (will.cheng@efctw.com)
LastEditTime: 2024-07-31 08:17:39
FilePath: /PoseidonAI-Server/routes/training_configurations.py
Description: 

Copyright (c) 2024 by chengyong@pku.edu.cn, All Rights Reserved. 
'''
import os
from bson import ObjectId
import traceback
import json
import uuid

from flask import Blueprint, request, jsonify

from routes.auth import jwt_required
from app.config import Config
from services.training_configuration_service import TrainingConfigurationService
from .utils import parse_immutable_dict

training_configurations_bp = Blueprint('training_configurations', __name__)
training_configurations_root = Config.TRAINING_CONFIGS_FOLDER
os.makedirs(training_configurations_root, exist_ok=True)

@training_configurations_bp.route('/user', methods=['GET'])
@jwt_required
def get_training_frameworks_by_user(user_id):
    response = {'code': 200, 'msg': 'ok', 'show_msg': 'ok', 'results': []}
    try:
        training_configs = TrainingConfigurationService.get_training_configurations_by_user(user_id)
        response['results'] = training_configs
    except Exception as e:
        response['code'] = 500
        response['msg'] = str(e)
        traceback.print_exc()
    return jsonify(response), 200

@training_configurations_bp.route('/findby/<training_framework_id>', methods=['GET'])
@jwt_required
def get_training_frameworks_by_user_and_framework_id(user_id, training_framework_id):
    response = {'code': 200, 'msg': 'ok', 'show_msg': 'ok', 'results': []}
    try:
        training_configs = TrainingConfigurationService\
            .get_training_configurations_by_user_and_framework_id(
                user_id, training_framework_id
            )
        response['results'] = training_configs
    except Exception as e:
        response['code'] = 500
        response['msg'] = str(e)
        traceback.print_exc()
    return jsonify(response), 200

@training_configurations_bp.route('/create', methods=['POST'])
@jwt_required
def create(user_id):
    try:
        response = {'code': 200, 'msg': 'ok', 'show_msg': 'ok', 'results': None}
        data = request.form.to_dict()
        data = parse_immutable_dict(data)
        if (data.get('INPUT_MIN_SIZE_TRAIN', None)):
            if isinstance(data.get('INPUT_MIN_SIZE_TRAIN', []), str):
                data['INPUT_MIN_SIZE_TRAIN'] = [int(data['INPUT_MIN_SIZE_TRAIN'])]
        name = data['config_name']
        description = data.get('description', '')
        training_framework_id = data['training_framework_id']
        del data['config_name']
        del data['description']
        del data['training_framework_id']
        result = TrainingConfigurationService.create_configuration(name, user_id, training_framework_id, description, data)
        if not result:
            raise ValueError('Could not create training configuration')
    except Exception as e:
        response['code'] = 500
        response['msg'] = str(e)
        response['show_msg'] = 'error'
        traceback.print_exc()
    return jsonify(response), 200

@training_configurations_bp.route('/delete/<training_configuration_id>', methods=['DELETE'])
@jwt_required
def delete(user_id, training_configuration_id):
    response = {'code': 200, 'msg': 'ok', 'show_msg': 'ok', 'results': None}
    try:
        result = TrainingConfigurationService.delete_training_configuration(training_configuration_id)
        if result:
            return jsonify(response), 200
        raise ValueError('Could not delete training configuration')
    except Exception as e:
        response['msg'] = e
        return jsonify(response), 200