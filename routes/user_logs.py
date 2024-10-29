'''
Author: Will Cheng (will.cheng@efctw.com)
Date: 2024-10-21 13:32:23
LastEditors: Will Cheng (will.cheng@efctw.com)
LastEditTime: 2024-10-29 11:42:04
FilePath: /PoseidonAI-Server/routes/user_logs.py
'''
import os
from bson import ObjectId
import traceback
from datetime import datetime

from flask import Blueprint, request, jsonify

from routes.auth import jwt_required
from services.user_log_service import UserLogService

user_logs_bp = Blueprint('user_logs', __name__)

@user_logs_bp.route('/logs', methods=['POST'])
@jwt_required
def receive_logs(user_id):
    response = {'code': 200, 'msg': 'ok', 'show_msg': 'ok', 'results': None}
    if not request.is_json:
        response['code'] = 400
        response['msg'] = 'Request must be JSON'
        response['show_msg'] = 'error'
        return jsonify(response), 200

    logs = request.get_json()  # 获取传入的日志列表
    if not isinstance(logs, list):
        response['code'] = 400
        response['msg'] = 'Logs must be a list'
        response['show_msg'] = 'error'
        return jsonify(response), 200

    try:
        for log in logs:
            log['created_at'] = datetime.utcnow()
            log['user_id'] = ObjectId(user_id)
            UserLogService.create(**log)  # 插入到 MongoDB
        
        response['msg'] = 'Logs successfully saved'
        return jsonify(response), 201

    except Exception as e:
        response['code'] = 500
        response['msg'] = str(e)
        response['show_msg'] = 'Error inserting logs'
        traceback.print_exc()
        return jsonify(response), 500
    
@user_logs_bp.route('/logs', methods=['GET'])
@jwt_required
def get_logs(user_id):
    response = {'code': 200, 'msg': 'ok', 'show_msg': 'ok', 'results': []}
    try:
        logs = UserLogService.get_user_logs(user_id)
        response['results'] = logs
        return jsonify(response), 200
    except Exception as e:
        response['code'] = 500
        response['msg'] = str(e)
        response['show_msg'] = 'Error fetching logs'
        traceback.print_exc()
        return jsonify(response), 500