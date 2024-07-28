'''
Author: Will Cheng chengyong@pku.edu.cn
Date: 2024-07-24 22:17:36
LastEditors: Will Cheng chengyong@pku.edu.cn
LastEditTime: 2024-07-24 22:18:47
FilePath: /PoseidonAI-Server/routes/training_frameworks.py
Description: 

Copyright (c) 2024 by chengyong@pku.edu.cn, All Rights Reserved. 
'''
import os
from bson import ObjectId
import traceback

from flask import Blueprint, request, jsonify
from services.training_framwork_service import TrainingFrameworkService

training_frameworks_bp = Blueprint('training_frameworks', __name__)

@training_frameworks_bp.route('/list', methods=['GET'])
def get_training_frameworks():
    response = {'code': 200, 'msg': 'ok', 'show_msg': 'ok', 'results': []}
    try:
        training_frameworks = TrainingFrameworkService.get_training_frameworks()
        response['results'] = training_frameworks
    except Exception as e:
        response['code'] = 500
        response['msg'] = str(e)
        traceback.print_exc()
    return jsonify(response), 200