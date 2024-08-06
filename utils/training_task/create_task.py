import os
import glob
import shutil
import random
import copy
import time

from celery import task

from app.models import TrainingTask
from utils.common import read_json, write_json, write_yaml, remove_unannotated_images


progress_steps = [
    'Starting task',
    'Preparing task',
    'Reading COCO label file',
    'Splitting dataset',
    'Writing dataset and config files',
    'Task preparation complete',
    'Finishing task'
]

@task(bind=True, name='tasks.create.task')
def create_task(self, training_framework_name, args_file, epochs, gpu_id, val_ratio, dataset_dir, model, project_dir,
                name, user_id, algorithm_id, dataset_id, training_configuration_id, save_key, description):
    total_steps = 7  # 进度总步数
    current_step = 0

    def update_progress(step, description):
        nonlocal current_step
        current_step += step
        self.update_state(state='PROGRESS', meta={'current': current_step, 'total': total_steps, 'description': description, 'steps': progress_steps})

    update_progress(1, 'Starting task')
    time.sleep(1)

    if training_framework_name == 'YOLOv8':
        update_progress(1, 'Preparing task')
        time.sleep(1)
        [train_num, val_num] = create_yolov8_task(args_file, epochs, gpu_id, val_ratio, dataset_dir, model, project_dir, update_progress)
    elif training_framework_name == 'Detectron2-InstanceSegmentation':
        update_progress(1, 'Preparing task')
        time.sleep(1)
        [train_num, val_num] = create_d2_insseg_dataset(args_file, epochs, gpu_id, val_ratio, dataset_dir, model, project_dir, update_progress)
    else:
        raise NotImplementedError

    task = TrainingTask(name, user_id, algorithm_id, dataset_id, training_configuration_id, model, epochs, val_ratio, gpu_id, save_key, [train_num, val_num], description)
    task.save()
    update_progress(1, 'Finishing task')
    time.sleep(2)

    return {'train_num': train_num, 'val_num': val_num}

def create_yolov8_task(args_file, epochs, gpu_id, val_ratio, dataset_dir, model, project_dir, update_progress):
    update_progress(1, 'Reading COCO label file')
    time.sleep(1)
    coco_label_file = glob.glob(os.path.join(dataset_dir, 'mscoco', '*.json'))[0]
    class_names = get_class_names(coco_label_file)

    update_progress(1, 'Splitting dataset')
    time.sleep(1)
    dataset_file_content, [train_num, val_num] = split_yolov8_dataset(dataset_dir, val_ratio, class_names, os.path.join(project_dir, 'data'))

    update_progress(1, 'Writing dataset and config files')
    time.sleep(1)
    dataset_file = os.path.join(project_dir, 'dataset.yaml')
    cfg_file = os.path.join(project_dir, 'cfg.yaml')
    tarining_args = read_json(args_file)
    task_dir = os.path.join(project_dir, 'project')
    tarining_args.update(
        {
            'epochs': int(epochs), 
            'device': 'cuda:{}'.format(int(gpu_id)), 
            'project': task_dir,
            'model': model,
            'data': dataset_file,
            'mask_ratio': int(tarining_args['mask_ratio'])
        }
    )
    write_yaml(dataset_file_content, dataset_file)
    write_yaml(tarining_args, cfg_file)

    update_progress(1, 'Task preparation complete.')
    time.sleep(1)
    return [train_num, val_num]

def create_d2_insseg_dataset(args_file, epochs, gpu_id, val_ratio, dataset_dir, model, project_dir, update_progress):
    update_progress(1, 'Reading COCO label file')
    time.sleep(1)
    coco_label_file = glob.glob(os.path.join(dataset_dir, 'mscoco', '*.json'))[0]

    update_progress(1, 'Splitting dataset')
    time.sleep(1)
    train_num, val_num = split_d2_dataset(dataset_dir, val_ratio, os.path.join(project_dir, 'data'))

    update_progress(1, 'Writing dataset and config files')
    time.sleep(1)
    cfg_file = os.path.join(project_dir, 'cfg.yaml')
    tarining_args = read_json(args_file)
    task_dir = os.path.join(project_dir, 'project')
    tarining_args.update({'SOLVER_MAX_ITER': epochs, 'DATASETS_TRAIN': ['train_dataset'], 'DATASETS_TEST': ['val_dataset'] if val_num else []})
    training_args_yaml = d2_dict_to_yaml(tarining_args)
    training_args_yaml.update({'OUTPUT_DIR': task_dir})
    training_args_yaml['MODEL']['DEVICE'] = 'cuda:{}'.format(gpu_id)
    write_yaml(training_args_yaml, cfg_file)

    update_progress(1, 'Task preparation complete.')
    time.sleep(1)
    return [train_num, val_num]

def get_class_names(coco_label_file):
    data = read_json(coco_label_file)
    # Extract class names from the 'categories' field
    class_names = [category['name'] for category in data['categories']]
    
    return class_names

def split_yolov8_dataset(dataset_dir, val_ratio, class_names, output_dir):
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    train_images_dir = os.path.join(images_dir, 'train')
    train_labels_dir = os.path.join(labels_dir, 'train')
    val_images_dir = os.path.join(images_dir, 'val')
    val_labels_dir = os.path.join(labels_dir, 'val')
    [os.makedirs(d, exist_ok=True) for d in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]]
    
    image_files = glob.glob(os.path.join(dataset_dir, 'images', '*'))
    label_files = glob.glob(os.path.join(dataset_dir, 'yolo', '*'))
    label_basenames = set(os.path.splitext(os.path.basename(f))[0] for f in label_files)
    sub_image_files = [f for f in image_files if os.path.splitext(os.path.basename(f))[0] in label_basenames]
    data_pairs = list(zip(sub_image_files, label_files))
    val_image_num = int(val_ratio * len(data_pairs))
    val_pairs = random.sample(data_pairs, val_image_num)
    train_pairs = list(set(data_pairs) - set(val_pairs))
    for image_file, label_file in train_pairs:
        shutil.copy(image_file, train_images_dir)
        shutil.copy(label_file, train_labels_dir)
    for image_file, label_file in val_pairs:
        shutil.copy(image_file, val_images_dir)
        shutil.copy(label_file, val_labels_dir)
    # handle_preview_image(train_pairs[0][0], os.path.join(output_dir, '..'))
        
    return dict(
        path=output_dir, # dataset root dir
        train='images/train', # train images (relative to 'path') 4 images
        val='images/val' if val_image_num else '', # val images (relative to 'path') 4 images
        test='',
        names=class_names
    ), [len(train_pairs), len(val_pairs)]

def split_coco_dataset(coco_data, val_size):
    # 确保val_size不超过图像总数
    total_images = len(coco_data['images'])
    val_size = min(val_size, total_images)
    
    # 随机选择验证集图像
    val_images = random.sample(coco_data['images'], val_size)
    val_image_ids = {img['id'] for img in val_images}
    
    # 训练集图像
    train_images = [img for img in coco_data['images'] if img['id'] not in val_image_ids]
    train_image_ids = {img['id'] for img in train_images}
    
    # 根据图像ID分配标注
    train_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in train_image_ids]
    val_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in val_image_ids]
    
    # 创建新的COCO数据结构
    train_coco_data = copy.deepcopy(coco_data)
    val_coco_data = copy.deepcopy(coco_data)
    
    train_coco_data['images'] = train_images
    train_coco_data['annotations'] = train_annotations
    
    val_coco_data['images'] = val_images
    val_coco_data['annotations'] = val_annotations
    return train_coco_data, val_coco_data

def copy_images_to_train_val_dirs(image_files, train_data, val_data, train_dir, val_dir):
    """
    根据训练和验证数据，将图像文件复制到训练和验证目录。

    参数:
    image_files (list): 图片的绝对路径列表。
    train_data (dict): 训练数据的COCO格式字典。
    val_data (dict): 验证数据的COCO格式字典。
    train_dir (str): 训练目录的路径。
    val_dir (str): 验证目录的路径。
    """
    
    # 创建训练和验证目录（如果不存在）
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # 获取训练和验证数据中的图像文件名
    train_image_ids = {img['file_name'] for img in train_data['images']}
    val_image_ids = {img['file_name'] for img in val_data['images']}
    
    # 遍历图像文件并复制到相应的目录
    for image_file in image_files:
        image_name = os.path.basename(image_file)
        
        if image_name in train_image_ids:
            shutil.copy(image_file, os.path.join(train_dir, image_name))
        elif image_name in val_image_ids:
            shutil.copy(image_file, os.path.join(val_dir, image_name))
    # handle_preview_image(image_files[0], os.path.join(train_dir, '..', '..'))

def split_d2_dataset(dataset_dir, val_ratio, output_dir):
    coco_dataset_file = glob.glob(os.path.join(dataset_dir, 'mscoco', '*.json'))[0]
    image_files = glob.glob(os.path.join(dataset_dir, 'images', '*'))
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    [os.makedirs(d, exist_ok=True) for d in [train_dir, val_dir]]
    coco_annos = remove_unannotated_images(coco_dataset_file)
    train_data_file = os.path.join(output_dir, 'train.json')
    val_data_file = os.path.join(output_dir, 'val.json')
    val_image_num = int(val_ratio * len(coco_annos['images']))
    train_coco_data, val_coco_data = split_coco_dataset(coco_annos, val_image_num)
    write_json(train_coco_data, train_data_file)
    write_json(val_coco_data, val_data_file)
    copy_images_to_train_val_dirs(image_files, train_coco_data, val_coco_data, train_dir, val_dir)
    return len(train_coco_data['images']), len(val_coco_data['images'])

def d2_dict_to_yaml(data):
    yaml_structure = {
    "MODEL": {
        "META_ARCHITECTURE": data["META_ARCHITECTURE"],
        "BACKBONE": {
            "NAME": data["BACKBONE_NAME"]
        },
        "RESNETS": {
            "OUT_FEATURES": data["RESNETS_OUT_FEATURES"],
            "DEPTH": data["RESNETS_DEPTH"]
        },
        "FPN": {
            "IN_FEATURES": data["FPN_IN_FEATURES"]
        },
        "ANCHOR_GENERATOR": {
            "SIZES": data["ANCHOR_GENERATOR_SIZES"],
            "ASPECT_RATIOS": data["ANCHOR_GENERATOR_ASPECT_RATIOS"]
        },
        "RPN": {
            "IN_FEATURES": data["RPN_IN_FEATURES"],
            "PRE_NMS_TOPK_TRAIN": data["RPN_PRE_NMS_TOPK_TRAIN"],
            "PRE_NMS_TOPK_TEST": data["RPN_PRE_NMS_TOPK_TEST"],
            "POST_NMS_TOPK_TRAIN": data["RPN_POST_NMS_TOPK_TRAIN"],
            "POST_NMS_TOPK_TEST": data["RPN_POST_NMS_TOPK_TEST"]
        },
        "ROI_HEADS": {
            "NAME": data["ROI_HEADS_NAME"],
            "IN_FEATURES": data["ROI_HEADS_IN_FEATURES"]
        },
        "ROI_BOX_HEAD": {
            "NAME": data["ROI_BOX_HEAD_NAME"],
            "NUM_FC": data["ROI_BOX_HEAD_NUM_FC"],
            "POOLER_RESOLUTION": data["ROI_BOX_HEAD_POOLER_RESOLUTION"]
        },
        "ROI_MASK_HEAD": {
            "NAME": data["ROI_MASK_HEAD_NAME"],
            "NUM_CONV": data["ROI_MASK_HEAD_NUM_CONV"],
            "POOLER_RESOLUTION": data["ROI_MASK_HEAD_POOLER_RESOLUTION"]
        },
        "WEIGHTS": data["WEIGHTS"],
        "MASK_ON": data["MASK_ON"]
    },
    "DATASETS": {
        "TRAIN": list(data["DATASETS_TRAIN"]),
        "TEST": list(data["DATASETS_TEST"])
    },
    "SOLVER": {
        "IMS_PER_BATCH": data["SOLVER_IMS_PER_BATCH"],
        "BASE_LR": data["SOLVER_BASE_LR"],
        "STEPS": list(data["SOLVER_STEPS"]),
        "MAX_ITER": data["SOLVER_MAX_ITER"]
    },
    "INPUT": {
        "MIN_SIZE_TRAIN": list(data["INPUT_MIN_SIZE_TRAIN"])
    },
    "VERSION": data["VERSION"]
    }
    return yaml_structure
