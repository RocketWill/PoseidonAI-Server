from app import mongo
from bson import ObjectId
from datetime import datetime

from app.models import Dataset

class DatasetService:
    @staticmethod
    def create_dataset(user_id, name, description, detect_type, label_file, image_files, valid_images_num, save_key, format=['mscoco']):
        dataset = Dataset(user_id, name, description, detect_type, label_file, image_files, valid_images_num, save_key, format)
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
