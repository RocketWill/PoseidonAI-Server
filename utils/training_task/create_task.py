'''
Author: Will Cheng (will.cheng@efctw.com)
Date: 2024-08-02 14:09:31
LastEditors: Will Cheng chengyong@pku.edu.cn
LastEditTime: 2024-08-03 21:40:57
FilePath: /PoseidonAI-Server/utils/training_task/create_task.py
'''
import os
import glob
import shutil
import random

from utils.common import read_json, write_json, write_yaml


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
        
    return dict(
        path=output_dir, # dataset root dir
        train='images/train', # train images (relative to 'path') 4 images
        val='images/val' if val_image_num else '', # val images (relative to 'path') 4 images
        test='',
        names=class_names
    )
        

def create_yolov8_task(args_file, epochs, gpu_id, val_ratio, dataset_dir, model, project_dir):
    coco_label_file = glob.glob(os.path.join(dataset_dir, 'mscoco', '*.json'))[0]
    class_names = get_class_names(coco_label_file)
    dataset_file = os.path.join(project_dir, 'dataset.yaml')
    cfg_file = os.path.join(project_dir, 'cfg.yaml')
    cfg_file_json = os.path.join(project_dir, 'cfg.json')
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
    dataset_file_content = split_yolov8_dataset(dataset_dir, val_ratio, class_names, os.path.join(project_dir, 'data'))
    write_yaml(dataset_file_content, dataset_file)
    write_yaml(tarining_args, cfg_file)
    write_json(tarining_args, cfg_file_json)
    
if __name__ == '__main__':
    args_file = '/Users/will/workspace/EFC/PoseidonAI-Server/data/configs/66a0c88e791831cbf7f535c1/9faa066c-a339-4bca-a660-075f19655053/args.json'
    epochs = 10
    gpu_id = 0
    val_ratio = 0.2
    dataset_dir = '/Users/will/workspace/EFC/PoseidonAI-Server/data/dataset_raw/66a0c88e791831cbf7f535c1/011ce856-7f27-4188-ab0b-dbb624c5dd5c'
    project_dir = '/Users/will/workspace/EFC/PoseidonAI-Server/utils/training_task/test_output'
    model = 'yolov8m.yaml'
    create_yolov8_task(args_file, epochs, gpu_id, val_ratio, dataset_dir, model, project_dir)