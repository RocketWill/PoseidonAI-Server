from app.models import DetectType

class DetectTypeService:
    @staticmethod
    def create_detect_type(name, tag_name, description=''):
        detect_type = DetectType(name, tag_name, description)
        result = detect_type.save()
        return result

    @staticmethod
    def get_detect_type(detect_type_id):
        return DetectType.find_by_id(detect_type_id)

    @staticmethod
    def get_detect_types():
        return DetectType.list_all()

    @staticmethod
    def delete_detect_type(detect_type_id):
        return DetectType.delete(detect_type_id)
