import os
import glob
import shutil
import json
from bson import ObjectId
import uuid
import traceback

from flask import Blueprint, request, jsonify
from services.dataset_service import DatasetService
from routes.auth import jwt_required

from utils.dataset.create_datatset import create_dataset_helper
from utils.dataset.tools.visualize_coco_dataset import draw_annotations

datasets_bp = Blueprint('dataset', __name__)
dataset_raw_root = '/mnt/f/cy/workspace/EFC/PoseidonAI/data/dataset_raw'
static_folder = '/mnt/f/cy/workspace/EFC/PoseidonAI/static/vis_dataset'

@datasets_bp.route('/create', methods=['POST'])
@jwt_required
def create_dataset(user_id):
    try:
        label_file = request.files.get('jsonFile')
        image_files = request.files.getlist('imageFiles')

        # 将表单数据转换为 JSON 格式
        data = json.loads(request.form.to_dict()['data'])
        name = data['name']
        detect_types = list(data['detect_types'])
        dataset_format = list(data['dataset_format'])
        dataset_format = [d.lower() for d in dataset_format]
        description = data['description']
        r_label_file = [d['name'] for d in data['label_file']][-1]
        r_image_list = [d['name'] for d in data['image_list']]
        save_key = str(uuid.uuid4())
        valid_images = create_dataset_helper(dataset_raw_root, user_id, save_key, dataset_format,
                                      detect_types, r_image_list, label_file, image_files)
        
        result = DatasetService\
                .create_dataset(ObjectId(user_id), name, description, detect_types, 
                                               r_label_file, r_image_list, valid_images, 
                                               save_key, dataset_format)
        if not result:
            raise ValueError('保存資料集錯誤')
        return jsonify({ 'code': 200, 'show_msg': 'ok', 'msg': 'ok', 'results': None }), 200
    except Exception as e:
        print(traceback.print_exc())
        return jsonify({ 'code': 500, 'show_msg': 'error', 'msg': str(e), 'results': None }), 200

@datasets_bp.route('/datasets/<dataset_id>', methods=['GET'])
def get_dataset(dataset_id):
    dataset = DatasetService.get_dataset(dataset_id)
    if not dataset:
        return jsonify({'error': 'Dataset not found'}), 404
    return jsonify(dataset), 200

@datasets_bp.route('/user', methods=['GET'])
@jwt_required
def get_datasets_by_user(user_id):
    datasets = DatasetService.get_datasets_by_user(user_id)
    return jsonify(datasets), 200

@datasets_bp.route('/datasets/<dataset_id>', methods=['PUT'])
def update_dataset(dataset_id):
    data = request.get_json()
    name = data.get('name')
    description = data.get('description')
    detect_type = data.get('detect_type')
    label_file = data.get('label_file')
    image_files = data.get('image_files')
    format = data.get('format', 'MSCOCO')

    if not name or not description or not detect_type or not label_file or not image_files:
        return jsonify({'error': 'Missing required fields'}), 400

    try:
        DatasetService.update_dataset(dataset_id, name, description, detect_type, label_file, image_files, format)
        return jsonify({'message': 'Dataset updated successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@datasets_bp.route('/<dataset_id>', methods=['DELETE'])
@jwt_required
def delete_dataset(user_id, dataset_id):
    try:
        dataset = DatasetService.get_dataset(dataset_id)
        if not dataset:
            raise ValueError("刪除資料集失敗")
        save_key = dataset.save_key
        dataset_dir = os.path.join(dataset_raw_root, user_id, save_key)
        if not os.path.exists(dataset_dir):
            raise ValueError("找不到資料集位置")
        if DatasetService.delete_dataset(dataset_id):
            shutil.rmtree(dataset_dir)
        else:
            raise ValueError("刪除資料集失敗")
        return jsonify({'code': 200, 'msg': 'Dataset deleted successfully', 'show_msg': 'ok', 'results': None}), 200
    except Exception as e:
        return jsonify({'code': 500, 'msg': str(e), 'show_msg': 'errpr', 'results': None}), 500


@datasets_bp.route('/vis/<dataset_id>', methods=['GET'])
@jwt_required
def vis_dataset(user_id, dataset_id):
    try:
        dataset = DatasetService.get_dataset(dataset_id)
        if not dataset:
            raise ValueError("Dataset not found")
        label_file = glob.glob(os.path.join(dataset_raw_root, user_id, dataset.save_key, 'mscoco', '*.json'))[0]
        image_dir = os.path.join(dataset_raw_root, user_id, dataset.save_key, 'images')
        vis_dir = os.path.join(static_folder, user_id, dataset.save_key, 'preview')
        detect_types = dataset.detect_types
        draw_masks = True if 'seg' in detect_types else False
        draw_bboxes = True if 'det' in detect_types else False
        os.makedirs(vis_dir, exist_ok=True)
        draw_annotations(image_dir, label_file, vis_dir, draw_mask=draw_masks, draw_bbox=draw_bboxes)
        return jsonify({'code': 200, 'msg': 'Process preview images successfully', 'show_msg': 'ok', 'results': None}), 200
    except Exception as e:
        return jsonify({'code': 500, 'msg': str(e), 'show_msg': 'error', 'results': None}), 200
    
# {'code': 200, 'msg': 'Dataset deleted successfully', 'show_msg': 'ok', 'results': None}
# image_directory = 'image'
#     coco_annotation_file = '/mnt/d/workspace/general/plot_coco_dataset/instances_val.json'
#     output_directory = 'res'
#     draw_masks = True  # Set to False if you don't want to draw masks
#     draw_bboxes = False  # Set to True if you want to draw bounding boxes

#     draw_annotations(image_directory, coco_annotation_file, output_directory, draw_mask=draw_masks, draw_bbox=draw_bboxes)