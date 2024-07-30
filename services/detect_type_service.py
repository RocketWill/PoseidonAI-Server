'''
Author: Will Cheng (will.cheng@efctw.com)
Date: 2024-07-29 08:28:38
LastEditors: Will Cheng (will.cheng@efctw.com)
LastEditTime: 2024-07-30 08:59:16
FilePath: /PoseidonAI-Server/services/detect_type_service.py
'''
from app.models import DetectType

class DetectTypeService:
    @staticmethod
    def create_detect_type(name, tag_name, description=''):
        detect_type = DetectType(name, tag_name, description)
        result = detect_type.save()
        return result

    @staticmethod
    def get_detect_type(detect_type_id):
        return DetectType.find_by_id(detect_type_id).to_dict()

    @staticmethod
    def get_detect_types():
        return [d.to_dict() for d in DetectType.list_all()]

    @staticmethod
    def delete_detect_type(detect_type_id):
        return DetectType.delete(detect_type_id)
