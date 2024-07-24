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
        results = []
        for detect_type in detect_types:
            results.append({
                '_id': str(detect_type._id),
                'name': detect_type.name,
                'tag_name': detect_type.tag_name,
                'description': detect_type.description,
                'created_at': detect_type.created_at
            })
        response['results'] = results
    except Exception as e:
        response['code'] = 500
        response['msg'] = str(e)
        traceback.print_exc()
    return jsonify(response), 200