'''
Author: Will Cheng (will.cheng@efctw.com)
Date: 2024-07-29 08:28:38
LastEditors: Will Cheng (will.cheng@efctw.com)
LastEditTime: 2024-07-30 09:03:55
FilePath: /PoseidonAI-Server/routes/detect_types.py
'''
import os
from bson import ObjectId
import traceback

from flask import Blueprint, request, jsonify
from services.detect_type_service import DetectTypeService

detect_types_bp = Blueprint('detect_types', __name__)

@detect_types_bp.route('/list', methods=['GET'])
def get_detect_types():
    response = {'code': 200, 'msg': 'ok', 'show_msg': 'ok', 'results': []}
    try:
        detect_types = DetectTypeService.get_detect_types()
        response['results'] = detect_types
    except Exception as e:
        response['code'] = 500
        response['msg'] = str(e)
        traceback.print_exc()
    return jsonify(response), 200