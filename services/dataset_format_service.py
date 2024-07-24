from app.models import DatasetFormat

class DatasetFormatService:
    @staticmethod
    def create_dataset_format(name, description=''):
        dataset_format = DatasetFormat(name, description)
        result = dataset_format.save()
        return result

    @staticmethod
    def get_dataset_format(dataset_format_id):
        return DatasetFormat.find_by_id(dataset_format_id)

    @staticmethod
    def get_dataset_formats():
        return DatasetFormat.list_all()

    @staticmethod
    def delete_dataset_format(dataset_format_id):
        return DatasetFormat.delete(dataset_format_id)
