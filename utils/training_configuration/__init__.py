from .tools.yolov8 import save_config_file as save_yolov8_config
from .tools.detectron2_instance_segmentation import save_config_file as save_d2_insseg_config

def save_config_file(config_name, config_data, output_file):
    if config_name.lower() == 'yolov8':
        return save_yolov8_config(config_data, output_file)
    elif config_name.lower() == 'Detectron2-InstanceSegmentation'.lower():
        return save_d2_insseg_config(config_data, output_file)
    else:
        raise NotImplementedError