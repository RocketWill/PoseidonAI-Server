import os
from bson import ObjectId
import traceback

from flask import Blueprint, request, jsonify
from services.dataset_format_service import DatasetFormatService

dataset_formats_bp = Blueprint('dataset_formats', __name__)

@dataset_formats_bp.route('/list', methods=['GET'])
def get_dataset_formats():
    response = {'code': 200, 'msg': 'ok', 'show_msg': 'ok', 'results': []}
    try:
        dataset_formats = DatasetFormatService.get_dataset_formats()
        results = []
        for dataset_format in dataset_formats:
            results.append({
                '_id': str(dataset_format._id),
                'name': dataset_format.name,
                'description': dataset_format.description,
                'created_at': dataset_format.created_at
            })
        response['results'] = results
    except Exception as e:
        response['code'] = 500
        response['msg'] = str(e)
        traceback.print_exc()
    return jsonify(response), 200