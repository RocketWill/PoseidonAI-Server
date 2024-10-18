import os
import glob
import shutil
import json
import ntpath
from bson import ObjectId
import uuid
import traceback

from flask import Blueprint, request, jsonify
from services.dataset_service import DatasetService
from services.detect_type_service import DetectTypeService
from services.dataset_format_service import DatasetFormatService
from routes.auth import jwt_required
from celery.result import AsyncResult

from app.config import Config
from utils.dataset.create_datatset import create_dataset_helper
from tasks.dataset import draw_annotations_task, vis_classify_dataset

datasets_bp = Blueprint('dataset', __name__)
dataset_raw_root = Config.DATASET_RAW_FOLDER
dataset_visualization_root = os.path.join(Config.STATIC_FOLDER, 'dataset_visualization')
dataset_preview_root = Config.DATASET_PRVIEW_IMAGE_FOLDER
os.makedirs(dataset_raw_root, exist_ok=True)
os.makedirs(dataset_visualization_root, exist_ok=True)

@datasets_bp.route('/create', methods=['POST'])
@jwt_required
def create_dataset(user_id):
    try:
        # 将表单数据转换为 JSON 格式
        data = json.loads(request.form.to_dict()['data'])
        name = data['name']
        detect_type_id = data['detect_type_id']
        dataset_format = list(data['dataset_formats'])
        description = data['description']
        save_key = str(uuid.uuid4())
        dataset_format_data = [DatasetFormatService.get_dataset_format(d) for d in dataset_format]
        dataset_formats = [d.name.lower() for d in dataset_format_data]
        detect_type_data = DetectTypeService.get_detect_type(detect_type_id)
        detect_type = detect_type_data['tag_name'].lower()

        if detect_type == 'classify':
            zip_file = request.files.get('zipFile')
            valid_images, class_names, dataset_statistics, filenames = \
                DatasetService.process_classify_dataset(dataset_raw_root, user_id, save_key, zip_file)
            result = DatasetService\
                    .create_dataset(user_id, name, description, detect_type_id, 
                                    '-', filenames, valid_images, 
                                    save_key, dataset_format, class_names, dataset_statistics)
        else:
            label_file = request.files.get('jsonFile')
            image_files = request.files.getlist('imageFiles')
            r_label_file = [d['name'] for d in data['labelFile']][-1]
            r_image_list = [d['name'] for d in data['imageList']]
            
            valid_images, class_names, dataset_statistics, coco_image_filenames = create_dataset_helper(dataset_raw_root, user_id, save_key, dataset_formats,
                                        detect_type, r_image_list, label_file, image_files)
            result = DatasetService\
                    .create_dataset(user_id, name, description, detect_type_id, 
                                                r_label_file, coco_image_filenames, valid_images, 
                                                save_key, dataset_format, class_names, dataset_statistics)
        save_preview_result = DatasetService.create_preview_image(user_id, save_key, detect_type)
        
        if not (result or save_preview_result):
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
    try:
        response = {'code': 200, 'msg': 'ok', 'show_msg': 'ok', 'results': []}
        datasets = DatasetService.get_datasets_by_user_v2(user_id)
        response['results'] = datasets
    except Exception as e:
        traceback.print_exc()
        response['code'] = 500
        response['msg'] = str(e)
    return jsonify(response), 200

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
        vis_dir = os.path.join(dataset_visualization_root, user_id, dataset.save_key)
        preview_dir = os.path.join(dataset_preview_root, user_id, save_key)
        if not os.path.exists(dataset_dir):
            raise ValueError("找不到資料集位置")
        if DatasetService.delete_dataset(dataset_id):
            shutil.rmtree(dataset_dir)
            if os.path.exists(vis_dir):
                shutil.rmtree(vis_dir)
            if os.path.exists(preview_dir):
                shutil.rmtree(preview_dir)
        else:
            raise ValueError("刪除資料集失敗")
        return jsonify({'code': 200, 'msg': 'Dataset deleted successfully', 'show_msg': 'ok', 'results': None}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({'code': 500, 'msg': str(e), 'show_msg': 'errpr', 'results': None}), 200


@datasets_bp.route('/vis/<dataset_id>', methods=['POST'])
@jwt_required
def vis_dataset(user_id, dataset_id):
    try:
        dataset = DatasetService.get_dataset(dataset_id)
        if not dataset:
            raise ValueError("Dataset not found")
                    
        vis_dir = os.path.join(dataset_visualization_root, user_id, dataset.save_key)
        os.makedirs(vis_dir, exist_ok=True)
        detect_type_id = dataset.detect_type_id
        detect_type_data = DetectTypeService.get_detect_type(detect_type_id)
        detect_type = detect_type_data['tag_name'].lower()

        if detect_type == 'classify':
            task = vis_classify_dataset.delay(os.path.join(dataset_raw_root, user_id, dataset.save_key, 'dataset'), vis_dir)
        else:
            label_file = glob.glob(os.path.join(dataset_raw_root, user_id, dataset.save_key, 'mscoco', '*.json'))[0]
            image_dir = os.path.join(dataset_raw_root, user_id, dataset.save_key, 'images')
            draw_masks = True if 'seg' in detect_type else False
            draw_bboxes = True if 'det' in detect_type else False
            task = draw_annotations_task.delay(image_dir, label_file, vis_dir, draw_mask=draw_masks, draw_bbox=draw_bboxes)
        print(task.id)
        return jsonify({'code': 200, 'msg': 'Process preview images successfully', 'show_msg': 'ok', 'results': task.id}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({'code': 500, 'msg': str(e), 'show_msg': 'error', 'results': None}), 200
    
@datasets_bp.route('/vis/<task_id>', methods=['GET'])
def vis_dataset_status(task_id):
    try:
        task = draw_annotations_task.AsyncResult(task_id)
        return jsonify({'code': 200, 'msg': 'ok', 'show_msg': 'ok', 'results': task.status}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({'code': 500, 'msg': str(e), 'show_msg': 'error', 'results': 'FAILURE'}), 200

@datasets_bp.route('/checkVis/<dataset_id>', methods=['GET'])
@jwt_required
def check_is_vis_dataset_existed(user_id, dataset_id):
    exists = False
    files = [] 
    msg = 'Dataset visualization directory dose not exist.'
    try:
        dataset = DatasetService.get_dataset(dataset_id)
        if not dataset:
            raise ValueError("Dataset not found")
        vis_dir = os.path.join(dataset_visualization_root, user_id, dataset.save_key)
        if os.path.exists(vis_dir):
            exists = True
            files = glob.glob(os.path.join(vis_dir, "*"))
            files = [ntpath.basename(f) for f in files]
            msg = 'Dataset visualization directory exists.'
        return jsonify({'code': 200, 'msg': msg, 'show_msg': 'ok', 'results': { 'exists': exists, 'files': files }}), 200
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'code': 500, 'msg': str(e), 'show_msg': 'error', 'results': { 'exists': exists, 'files': files }}), 200
    
@datasets_bp.route('/findby/<dataset_format_id>/<detect_type_id>', methods=['GET'])
@jwt_required
def find_by_format_detect_type(user_id, dataset_format_id, detect_type_id):
    response = {'code': 200, 'msg': 'ok', 'show_msg': 'ok', 'results': {}}
    try:
        datasets = DatasetService.\
            find_by_user_format_detect_type(user_id, dataset_format_id, detect_type_id)
        response['results'] = datasets
        return jsonify(response), 200
    except Exception as e:
        traceback.print_exc()
        response['msg'] = str(e)
        response['code'] = 500
        return jsonify(response), 200
    
@datasets_bp.route('/statistics/<dataset_id>', methods=['GET'])
@jwt_required
def get_dataset_statistics(user_id, dataset_id):
    response = {'code': 200, 'msg': 'ok', 'show_msg': 'ok', 'results': {}}
    try:
        statistics = DatasetService.\
            get_dataset_statistics(dataset_id)
        response['results'] = statistics
        return jsonify(response), 200
    except Exception as e:
        traceback.print_exc()
        response['msg'] = str(e)
        response['code'] = 500
        return jsonify(response), 200