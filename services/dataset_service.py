from app import mongo
from bson import ObjectId
from datetime import datetime

from app.models import Dataset
from .detect_type_service import DetectTypeService
from .dataset_format_service import DatasetFormatService


class DatasetService:
    @staticmethod
    def create_dataset(user_id, name, description, detect_type_id, label_file, image_files, valid_images_num, save_key, dataset_format_ids):
        dataset = Dataset(user_id, name, description, detect_type_id, label_file, image_files, valid_images_num, save_key, dataset_format_ids)
        result = dataset.save()
        return result

    @staticmethod
    def get_dataset(dataset_id):
        return Dataset.find_by_id(dataset_id)

    @staticmethod
    def get_datasets_by_user(user_id):
        return Dataset.find_by_user(user_id)

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
    def find_by_user_format_detect_type(user_id, dataset_format_id, detect_type_id):
        def handle_dataset_format():
            data = DatasetFormatService.get_dataset_format(dataset_format_id).to_dict()
            return dict(
                name=data['name']
            )
        def handle_detect_type():
            data = DetectTypeService.get_detect_type(detect_type_id)
            return dict(
                name=data['name'],
                tag_name=data['tag_name']
            )
        def handle_datasets(data):
            return dict(
                _id=str(data['_id']),
                name=data['name'],
                description=data['description'],
                valid_images_num=data['valid_images_num'],
                created_at=data['created_at']
            )
        datasets = Dataset.find_by_user_format_detect_type(user_id, dataset_format_id, detect_type_id)
        results = dict(
            datasets=[handle_datasets(d) for d in datasets],
            detect_type=handle_detect_type(),
            dataset_format=handle_dataset_format()
        )
        return results