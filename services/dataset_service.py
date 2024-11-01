'''
Author: Will Cheng (will.cheng@efctw.com)
Date: 2024-07-29 08:28:38
LastEditors: Will Cheng (will.cheng@efctw.com)
LastEditTime: 2024-10-18 13:58:40
FilePath: /PoseidonAI-Server/services/dataset_service.py
'''
import os
import glob
import random
from bson import ObjectId
from datetime import datetime

from app.config import Config
from app.models import Dataset

from .detect_type_service import DetectTypeService
from .dataset_format_service import DatasetFormatService
from utils.common.dataset_statistics import analyze_coco_annotation
from utils.common import handle_preview_image
from utils.dataset.create_datatset import create_classify_dataset_helper

dataset_raw_root = Config.DATASET_RAW_FOLDER
dataset_preview_root = Config.DATASET_PRVIEW_IMAGE_FOLDER
os.makedirs(dataset_raw_root, exist_ok=True)
os.makedirs(dataset_preview_root, exist_ok=True)

def format_data(data):
    data['_id'] = str(data['_id'])
    data['user_id'] = str(data['user_id'])
    data['detect_type_id'] = str(data['detect_type_id'])
    data['dataset_format_ids'] = [str(d) for d in data['dataset_format_ids']]
    dataset_format_data = [DatasetFormatService.get_dataset_format(d).to_dict() for d in data['dataset_format_ids']]
    detect_type_data = DetectTypeService.get_detect_type(data['detect_type_id'])
    data['dataset_format'] = dataset_format_data
    data['detect_type'] = detect_type_data
    data['image_files'] = sorted(data['image_files'])
    return data

class DatasetService:
    @staticmethod
    def create_dataset(user_id, name, description, detect_type_id, label_file, image_files, valid_images_num, save_key, dataset_format_ids, class_names, statistics):
        dataset = Dataset(user_id, name, description, detect_type_id, label_file, image_files, valid_images_num, save_key, dataset_format_ids, class_names, statistics)
        result = dataset.save()
        return result
    
    @staticmethod
    def process_classify_dataset(dataset_raw_root, user_id, save_key, zip_file):
        valid_images, class_names, dataset_statistics, filenames = create_classify_dataset_helper(dataset_raw_root, user_id, save_key, zip_file)
        return valid_images, class_names, dataset_statistics, filenames

    @staticmethod
    def create_preview_image(user_id, save_key, detect_type='classify'):
        if detect_type == 'classify':
            dataset_dir = os.path.join(dataset_raw_root, user_id, save_key, 'dataset')
            image_file = random.choice(glob.glob(os.path.join(dataset_dir, '*', '*')))
        else:
            dataset_image_dir = os.path.join(dataset_raw_root, user_id, save_key, 'images')
            if not os.path.exists(dataset_image_dir):
                return False
            image_file = random.choice(glob.glob(os.path.join(dataset_image_dir, '*')))
        output_dir = os.path.join(dataset_preview_root, user_id, save_key)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'preview.jpg')
        if not os.path.exists(handle_preview_image(image_file, output_file)):
            return False
        return True

    @staticmethod
    def get_dataset(dataset_id):
        return Dataset.find_by_id(dataset_id)
    
    @staticmethod
    def get_dataset_statistics(dataset_id):
        dataset_data = Dataset.find_by_id(dataset_id)
        user_id = str(dataset_data.user_id)
        save_key = dataset_data.save_key
        coco_label_file = os.path.join(dataset_raw_root, user_id, save_key, 'mscoco', 'annotations.json')
        return analyze_coco_annotation(coco_label_file)

    @staticmethod
    def get_datasets_by_user(user_id):
        return [format_data(d) for d in Dataset.find_by_user(user_id)]

    @staticmethod
    def update_dataset(dataset_id, name, description, save_key):
        dataset = Dataset.find_by_id(dataset_id)
        dataset.name = name
        dataset.description = description
        dataset.save_key = save_key
        return dataset.update()

    @staticmethod
    def delete_dataset(dataset_id):
        return Dataset.delete(dataset_id)
    
    @staticmethod
    def get_datasets_by_user_v2(dataset_id):
        results = Dataset.find_by_user(dataset_id)
        return [format_data(d) for d in results]

    @staticmethod
    def find_by_user_format_detect_type(user_id, dataset_format_id, detect_type_id):
        # def handle_dataset_format():
        #     data = DatasetFormatService.get_dataset_format(dataset_format_id).to_dict()
        #     return dict(
        #         name=data['name']
        #     )
        # def handle_detect_type():
        #     data = DetectTypeService.get_detect_type(detect_type_id)
        #     return dict(
        #         name=data['name'],
        #         tag_name=data['tag_name']
        #     )
        # def handle_datasets(data):
        #     return dict(
        #         _id=str(data['_id']),
        #         name=data['name'],
        #         description=data['description'],
        #         valid_images_num=data['valid_images_num'],
        #         created_at=data['created_at']
        #     )
        datasets = Dataset.find_by_user_format_detect_type(user_id, dataset_format_id, detect_type_id)
        results = [format_data(d) for d in datasets]
        # results = dict(
        #     # datasets=[handle_datasets(d) for d in datasets],
        #     datasets=[format_data(d) for d in datasets],
        #     detect_type=handle_detect_type(),
        #     dataset_format=handle_dataset_format()
        # )
        # return results
        return results
