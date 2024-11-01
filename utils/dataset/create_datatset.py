'''
Author: Will Cheng chengyong@pku.edu.cn
Date: 2024-08-05 19:04:56
LastEditors: Will Cheng (will.cheng@efctw.com)
LastEditTime: 2024-10-08 14:04:11
FilePath: /PoseidonAI-Server/utils/dataset/create_datatset.py
Description: 

Copyright (c) 2024 by chengyong@pku.edu.cn, All Rights Reserved. 
'''
import os
import json
import glob

from .tools.verify_coco_format import validate_coco_format, find_valid_images
from .tools.coco2yolo import convert_coco_json
from utils.common import remove_unannotated_images, filter_coco_images_and_annotations, \
    write_json, read_json
from utils.common.dataset_statistics import analyze_coco_annotation, analyze_classify_annotation
from .utils import validate_directory_structure, unzip_skip_first_level, get_image_filenames

def create_classify_dataset_helper(dataset_raw_root, user_id, save_key, zip_file):
    dataset_root = os.path.join(dataset_raw_root, user_id, save_key)
    os.makedirs(dataset_root)
    dataset_dir = os.path.join(dataset_root, 'dataset')
    dataset_file = os.path.join(dataset_root, 'dataset.zip')
    zip_file.save(dataset_file)
    valid, msg = validate_directory_structure(dataset_file)
    if not valid:
        raise ValueError(msg)
    unzip_skip_first_level(dataset_file, dataset_dir)
    class_names = get_classify_class_names(dataset_dir)
    filenames = get_image_filenames(dataset_dir)
    valid_images = len(filenames)
    dataset_statistics = analyze_classify_annotation(dataset_dir)
    return valid_images, class_names, dataset_statistics, filenames
    
def create_dataset_helper(dataset_raw_root, user_id, save_key, dataset_format, detect_type, r_image_list, label_file, image_files):
    output_image_dir = os.path.join(dataset_raw_root, user_id, save_key, 'images')
    output_coco_dir = os.path.join(dataset_raw_root, user_id, save_key, 'mscoco')
    coco_label_file = os.path.join(output_coco_dir, label_file.filename)
    valid_coco_images = create_coco(output_coco_dir, output_image_dir, r_image_list, label_file, image_files)
    filtered_coco_label_file = os.path.join(output_coco_dir, 'annotations.json')
    coco_data = read_json(filtered_coco_label_file)
    
    coco_image_filenames = list({image['file_name'] for image in coco_data['images']})
    dataset_statistics = analyze_coco_annotation(filtered_coco_label_file)
    if 'yolo' in dataset_format:
        output_yolo_dir = os.path.join(dataset_raw_root, user_id, save_key, 'yolo')
        use_segments = True if 'seg' in detect_type else False
        use_keypoints = True if 'kpts' in detect_type else False
        # label_file = os.path.join(output_coco_dir, label_file.filename)
        label_file = os.path.join(output_coco_dir, 'annotations.json')
        valid_yolo_images = create_yolo(label_file, output_yolo_dir, use_segments, use_keypoints)
        if len(valid_coco_images) != valid_yolo_images:
            print('valid images are different, coco: {}, yolo: {}.'.format(len(valid_coco_images), valid_yolo_images))
        return min(len(valid_coco_images), valid_yolo_images), get_class_names(coco_label_file), dataset_statistics, coco_image_filenames
    return len(valid_coco_images), get_class_names(coco_label_file), dataset_statistics, coco_image_filenames
            
def create_coco(output_coco_dir, output_image_dir, r_image_list, label_file, image_files):
    os.makedirs(output_coco_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)
    label_file_save_path = os.path.join(output_coco_dir, label_file.filename)
    label_file.save(label_file_save_path)
    
    with open(label_file_save_path, 'r') as f:
        json_data = f.read()
    
    # 验证格式
    valid, coco_data = validate_coco_format(json_data)
    coco_data = remove_unannotated_images(coco_data)
    
    if valid:
        valid_images = find_valid_images(r_image_list, coco_data)
    else:
        raise ValueError('不合规的 COCO 标注文件')
    
    # 获取COCO数据中的图像文件名列表
    coco_image_filenames = {image['file_name'] for image in coco_data['images']}
    
    for file in image_files:
        if file:
            filename = file.filename
            # 仅在文件名存在于COCO数据中的时候保存图像
            if filename in coco_image_filenames:
                file.save(os.path.join(output_image_dir, filename))

    image_files = sorted(glob.glob(os.path.join(output_image_dir, '*')))
    filtered_coco_data = filter_coco_images_and_annotations(coco_data, image_files)
    write_json(filtered_coco_data, os.path.join(output_coco_dir, 'annotations.json'))
    return valid_images

def create_yolo(label_file, output_labels_dir, use_segments, use_keypoints):
    return convert_coco_json(label_file, output_labels_dir, use_segments, use_keypoints)

def get_class_names(coco_label_file):
    print(coco_label_file)
    with open(coco_label_file, 'r') as f:
        json_str = f.read()
        data = json.loads(json_str)
    # Extract class names from the 'categories' field
    class_names = [category['name'] for category in data['categories']]
    return class_names

def get_classify_class_names(dataset_dir):
    subdirectories = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    return subdirectories