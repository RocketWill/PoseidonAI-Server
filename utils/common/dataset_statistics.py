'''
Author: Will Cheng (will.cheng@efctw.com)
Date: 2024-08-12 10:19:06
LastEditors: Will Cheng (will.cheng@efctw.com)
LastEditTime: 2024-08-12 15:27:16
FilePath: /PoseidonAI-Server/utils/common/dataset_statistics.py
'''
import os
import json
from collections import defaultdict

def analyze_coco_annotation(coco_annotation_file):
    with open(coco_annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    # Initialize counters
    category_counts = defaultdict(int)
    total_images = len(coco_data['images'])
    total_instances = len(coco_data['annotations'])

    # Count occurrences of each category in annotations
    for annotation in coco_data['annotations']:
        category_id = annotation['category_id']
        category_counts[category_id] += 1
    
    # Map category ids to their names
    category_id_to_name = {category['id']: category['name'] for category in coco_data['categories']}
    category_counts_list = []
    # Display results
    for category_id, count in category_counts.items():
        category_name = category_id_to_name.get(category_id, "Unknown")
        category_counts_list.append({'name': category_name, 'value': count})
    return dict(
        total_images=total_images,
        total_instances=total_instances,
        category_counts=category_counts_list
    )


def count_yolo_dataset_info(pairs, classnames):
    dataset_info = {
        'instances': 0,
        'images': 0
    }

    class_info = defaultdict(lambda: {'instances': 0, 'images': 0})

    for image_file, label_file in pairs:
        if not os.path.exists(label_file):
            continue
        dataset_info['images'] += 1
        with open(label_file, 'r') as f:
            labels = f.readlines()
            dataset_info['instances'] += len(labels)
            classes_in_image = set()
            for label in labels:
                class_id = int(label.split()[0])
                classname = classnames[class_id]
                classes_in_image.add(class_id)
                class_info[classname]['instances'] += 1
            
            for class_id in classes_in_image:
                classname = classnames[class_id]
                class_info[classname]['images'] += 1

    return dataset_info, class_info

def summarize_yolo_dataset(train_pairs, val_pairs, classnames):
    # 統計訓練集信息
    train_info, train_class_info = count_yolo_dataset_info(train_pairs, classnames)
    val_info, val_class_info = count_yolo_dataset_info(val_pairs, classnames)

    # 構建輸出格式
    dataset_summary = []
    class_summary = []

    # 訓練集總體信息
    dataset_summary.append({
        'dataset_type': 'train',
        'instances': train_info['instances'],
        'images': train_info['images']
    })

    # 驗證集總體信息
    dataset_summary.append({
        'dataset_type': 'val',
        'instances': val_info['instances'],
        'images': val_info['images']
    })

    # 各類別訓練集信息
    for classname in classnames:
        class_summary.append({
            'dataset_type': 'train',
            'classname': classname,
            'class_id': classnames.index(classname),
            'instances': train_class_info[classname]['instances'],
            'images': train_class_info[classname]['images']
        })

    # 各類別驗證集信息
    for classname in classnames:
        class_summary.append({
            'dataset_type': 'val',
            'classname': classname,
            'class_id': classnames.index(classname),
            'instances': val_class_info[classname]['instances'],
            'images': val_class_info[classname]['images']
        })

    return dataset_summary, class_summary


def count_coco_dataset_info(coco_data, classnames):
    dataset_info = {
        'instances': 0,
        'images': len(coco_data['images'])
    }

    class_info = defaultdict(lambda: {'instances': 0, 'images': set()})

    for annotation in coco_data['annotations']:
        class_id = annotation['category_id']
        image_id = annotation['image_id']
        classname = classnames[class_id]

        dataset_info['instances'] += 1
        class_info[classname]['instances'] += 1
        class_info[classname]['images'].add(image_id)

    # 將 set 轉換為數量
    for classname in class_info:
        class_info[classname]['images'] = len(class_info[classname]['images'])

    return dataset_info, class_info

def summarize_coco_dataset(train_coco_data, val_coco_data, classnames):
    # 統計訓練集信息
    train_info, train_class_info = count_coco_dataset_info(train_coco_data, classnames)
    val_info, val_class_info = count_coco_dataset_info(val_coco_data, classnames)

    # 構建輸出格式
    dataset_summary = []
    class_summary = []

    # 訓練集總體信息
    dataset_summary.append({
        'dataset_type': 'train',
        'instances': train_info['instances'],
        'images': train_info['images']
    })

    # 驗證集總體信息
    dataset_summary.append({
        'dataset_type': 'val',
        'instances': val_info['instances'],
        'images': val_info['images']
    })

    # 各類別訓練集信息
    for classname in classnames:
        class_summary.append({
            'dataset_type': 'train',
            'classname': classname,
            'class_id': classnames.index(classname),
            'instances': train_class_info[classname]['instances'],
            'images': train_class_info[classname]['images']
        })

    # 各類別驗證集信息
    for classname in classnames:
        class_summary.append({
            'dataset_type': 'val',
            'classname': classname,
            'class_id': classnames.index(classname),
            'instances': val_class_info[classname]['instances'],
            'images': val_class_info[classname]['images']
        })

    return dataset_summary, class_summary
