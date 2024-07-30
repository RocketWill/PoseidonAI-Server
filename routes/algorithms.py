'''
Author: Will Cheng chengyong@pku.edu.cn
Date: 2024-07-24 22:17:36
LastEditors: Will Cheng (will.cheng@efctw.com)
LastEditTime: 2024-07-30 09:37:29
FilePath: /PoseidonAI-Server/routes/algorithms.py
Description: 

Copyright (c) 2024 by chengyong@pku.edu.cn, All Rights Reserved. 
'''
import traceback

from flask import Blueprint, request, jsonify
from services.algorithm_service import AlgorithmService

algorithms_bp = Blueprint('algorithms', __name__)

@algorithms_bp.route('/list', methods=['GET'])
def get_algorithms():
    response = {'code': 200, 'msg': 'ok', 'show_msg': 'ok', 'results': []}
    try:
        algorithms = AlgorithmService.get_algorithms()
        response['results'] = algorithms
    except Exception as e:
        response['code'] = 500
        response['msg'] = str(e)
        traceback.print_exc()
    return jsonify(response), 200