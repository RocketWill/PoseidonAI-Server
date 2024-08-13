import os
import glob
import shutil
import random
import time
import copy

from celery import task
from app.models import TrainingTask
from utils.common import read_json, write_json, write_yaml, \
                         remove_unannotated_images, filter_coco_images_and_annotations
from utils.common.dataset_statistics import summarize_yolo_dataset, summarize_coco_dataset

# 定義任務進度步驟
progress_steps = [
    'Starting task',
    'Preparing task',
    'Reading COCO label file',
    'Splitting dataset',
    'Writing dataset and config files',
    'Task preparation complete',
    'Finishing task'
]

# Celery 任務，用於創建訓練任務
@task(bind=True, name='tasks.create.task')
def create_task(self, training_framework_name, args_file, epochs, gpu_id, val_ratio, dataset_dir, model, project_dir,
                name, user_id, algorithm_id, dataset_id, training_configuration_id, save_key, description):
    total_steps = len(progress_steps)  # 計算總步數
    current_step = 0

    def update_progress(step, description):
        nonlocal current_step
        current_step += step
        self.update_state(state='PROGRESS', meta={'current': current_step, 'total': total_steps, 'description': description, 'steps': progress_steps})

    update_progress(1, 'Starting task')
    time.sleep(1)

    # 根據不同框架名稱調用相應的任務創建函數
    if training_framework_name == 'YOLOv8':
        update_progress(1, 'Preparing task')
        time.sleep(1)
        [train_num, val_num], summary = create_yolov8_task(args_file, epochs, gpu_id, val_ratio, dataset_dir, model, project_dir, update_progress)
    elif training_framework_name == 'Detectron2-InstanceSegmentation':
        update_progress(1, 'Preparing task')
        time.sleep(1)
        [train_num, val_num], summary = create_d2_insseg_dataset(args_file, epochs, gpu_id, val_ratio, dataset_dir, model, project_dir, update_progress)
    else:
        raise NotImplementedError

    task = TrainingTask(name, user_id, algorithm_id, dataset_id, training_configuration_id, model, epochs, val_ratio, gpu_id, save_key, [train_num, val_num], description, summary)
    task.save()
    update_progress(1, 'Finishing task')
    time.sleep(2)

    return {'train_num': train_num, 'val_num': val_num}

# 創建 YOLOv8 訓練任務
def create_yolov8_task(args_file, epochs, gpu_id, val_ratio, dataset_dir, model, project_dir, update_progress):
    update_progress(1, 'Reading COCO label file')
    time.sleep(1)
    coco_label_file = os.path.join(dataset_dir, 'mscoco', 'annotations.json')
    class_names = get_class_names(coco_label_file)

    update_progress(1, 'Splitting dataset')
    time.sleep(1)
    dataset_file_content, (train_pairs, val_pairs) = split_yolov8_dataset(dataset_dir, val_ratio, class_names, os.path.join(project_dir, 'data'))
    dataset_summary, classes_summary = summarize_yolo_dataset(train_pairs, val_pairs, class_names)

    update_progress(1, 'Writing dataset and config files')
    time.sleep(1)
    dataset_file = os.path.join(project_dir, 'dataset.yaml')
    cfg_file = os.path.join(project_dir, 'cfg.yaml')
    training_args = read_json(args_file)
    task_dir = os.path.join(project_dir, 'project')
    training_args.update(
        {
            'epochs': int(epochs), 
            'device': 'cuda:{}'.format(int(gpu_id)), 
            'project': task_dir,
            'model': model,
            'data': dataset_file,
            'mask_ratio': int(training_args['mask_ratio']),
            'cache': ''
        }
    )
    write_yaml(dataset_file_content, dataset_file)
    write_yaml(training_args, cfg_file)

    update_progress(1, 'Task preparation complete.')
    time.sleep(1)
    return [len(train_pairs), len(val_pairs)], classes_summary

# 創建 Detectron2-InstanceSegmentation 訓練任務
def create_d2_insseg_dataset(args_file, epochs, gpu_id, val_ratio, dataset_dir, model, project_dir, update_progress):
    model_resnet_depth = int(model.split('___')[1]) # example: mask_rcnn_R_50_FPN_3x___18, to get the Resnet depth
    update_progress(1, 'Reading COCO label file')
    time.sleep(1)
    coco_label_file = os.path.join(dataset_dir, 'mscoco', 'annotations.json')
    class_names = get_class_names(coco_label_file)
    update_progress(1, 'Splitting dataset')
    time.sleep(1)
    train_coco_data, val_coco_data = split_d2_dataset(dataset_dir, val_ratio, os.path.join(project_dir, 'data'))
    dataset_summary, classes_summary = summarize_coco_dataset(train_coco_data, val_coco_data, class_names)

    update_progress(1, 'Writing dataset and config files')
    time.sleep(1)
    cfg_file = os.path.join(project_dir, 'cfg.yaml')
    training_args = read_json(args_file)
    task_dir = os.path.join(project_dir, 'project')
    training_args.update({'RESNETS_DEPTH': model_resnet_depth, 'SOLVER_MAX_ITER': epochs, 'DATASETS_TRAIN': ['train_dataset'], 'DATASETS_TEST': ['val_dataset'] if len(val_coco_data['images']) else []})
    training_args_yaml = d2_dict_to_yaml(training_args)
    training_args_yaml.update({'OUTPUT_DIR': task_dir})
    training_args_yaml['MODEL']['DEVICE'] = 'cuda:{}'.format(gpu_id)
    write_yaml(training_args_yaml, cfg_file)

    update_progress(1, 'Task preparation complete.')
    time.sleep(1)
    return [len(train_coco_data['images']), len(val_coco_data['images'])], classes_summary

# 提取 COCO 標籤文件中的類別名稱
def get_class_names(coco_label_file):
    data = read_json(coco_label_file)
    class_names = [category['name'] for category in data['categories']]
    return class_names

# 分割 YOLOv8 數據集
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

    # 生成文件名到絕對路徑的映射
    image_files_dict = {os.path.splitext(os.path.basename(f))[0]: os.path.abspath(f) for f in image_files}
    label_files_dict = {os.path.splitext(os.path.basename(f))[0]: os.path.abspath(f) for f in label_files}

    # 確保只有匹配的圖像和標籤文件
    common_basenames = set(image_files_dict.keys()).intersection(set(label_files_dict.keys()))

    # 生成匹配對
    data_pairs = [(image_files_dict[basename], label_files_dict[basename]) for basename in common_basenames]

    # 劃分訓練集和驗證集
    val_image_num = int(val_ratio * len(data_pairs))
    val_pairs = random.sample(data_pairs, val_image_num)
    train_pairs = list(set(data_pairs) - set(val_pairs))

    # 複製文件到目標目錄
    for image_file, label_file in train_pairs:
        shutil.copy(image_file, train_images_dir)
        shutil.copy(label_file, train_labels_dir)
    for image_file, label_file in val_pairs:
        shutil.copy(image_file, val_images_dir)
        shutil.copy(label_file, val_labels_dir)
        
    return dict(
        path=output_dir,
        train='images/train',
        val='images/val' if val_image_num else '',
        test='',
        names=class_names
    ), (train_pairs, val_pairs)

# 分割 COCO 數據集
def split_coco_dataset(coco_data, val_size):
    total_images = len(coco_data['images'])
    val_size = min(val_size, total_images)
    
    # 隨機選擇驗證集圖像
    val_images = random.sample(coco_data['images'], val_size)
    val_image_ids = {img['id'] for img in val_images}
    
    # 生成訓練集圖像
    train_images = [img for img in coco_data['images'] if img['id'] not in val_image_ids]
    train_image_ids = {img['id'] for img in train_images}
    
    # 根據圖像 ID 分配標註
    train_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in train_image_ids]
    val_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in val_image_ids]
    
    train_coco_data = copy.deepcopy(coco_data)
    val_coco_data = copy.deepcopy(coco_data)
    
    train_coco_data['images'] = train_images
    train_coco_data['annotations'] = train_annotations
    
    val_coco_data['images'] = val_images
    val_coco_data['annotations'] = val_annotations
    return train_coco_data, val_coco_data

# 複製圖像到訓練和驗證目錄
def copy_images_to_train_val_dirs(image_files, train_data, val_data, train_dir, val_dir):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    train_image_ids = {img['file_name'] for img in train_data['images']}
    val_image_ids = {img['file_name'] for img in val_data['images']}
    
    for image_file in image_files:
        image_name = os.path.basename(image_file)
        if image_name in train_image_ids:
            shutil.copy(image_file, os.path.join(train_dir, image_name))
        elif image_name in val_image_ids:
            shutil.copy(image_file, os.path.join(val_dir, image_name))

# 分割 Detectron2 數據集
def split_d2_dataset(dataset_dir, val_ratio, output_dir):
    coco_dataset_file = os.path.join(dataset_dir, 'mscoco', 'annotations.json')
    image_files = glob.glob(os.path.join(dataset_dir, 'images', '*'))
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    [os.makedirs(d, exist_ok=True) for d in [train_dir, val_dir]]
    coco_annos = remove_unannotated_images(coco_dataset_file)
    # 验证图像是否存在，并移除多余图像和标注
    coco_annos = filter_coco_images_and_annotations(coco_annos, image_files)
    train_data_file = os.path.join(output_dir, 'train.json')
    val_data_file = os.path.join(output_dir, 'val.json')
    val_image_num = int(val_ratio * len(coco_annos['images']))
    train_coco_data, val_coco_data = split_coco_dataset(coco_annos, val_image_num)
    write_json(train_coco_data, train_data_file)
    write_json(val_coco_data, val_data_file)
    copy_images_to_train_val_dirs(image_files, train_coco_data, val_coco_data, train_dir, val_dir)
    return train_coco_data, val_coco_data

# 將 Detectron2 字典轉換為 YAML 結構
def d2_dict_to_yaml(data):
    yaml_structure = {
        "MODEL": {
            "META_ARCHITECTURE": data["META_ARCHITECTURE"],
            "BACKBONE": {
                "NAME": data["BACKBONE_NAME"]
            },
            "RESNETS": {
                "OUT_FEATURES": data["RESNETS_OUT_FEATURES"],
                "DEPTH": data["RESNETS_DEPTH"],
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

    if (int(data.get('RESNETS_DEPTH')) in [18, 34]):
        yaml_structure['MODEL']['RESNETS']['RES2_OUT_CHANNELS'] = 64
    return yaml_structure
