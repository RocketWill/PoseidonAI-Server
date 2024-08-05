'''
Author: Will Cheng chengyong@pku.edu.cn
Date: 2024-07-26 16:59:09
LastEditors: Will Cheng chengyong@pku.edu.cn
LastEditTime: 2024-08-05 19:24:08
FilePath: /PoseidonAI-Server/utils/common/__init__.py
Description: 

Copyright (c) 2024 by chengyong@pku.edu.cn, All Rights Reserved. 
'''
import json
import re

import yaml

def read_json(file):
    with open(file) as f:
        json_str = f.read()
        
        # 移除 // 注释
        json_str = re.sub(r'(?<!:)\/\/.*(?=[\n\r])', '', json_str)  # 确保 // 不在 URL 中
        # 移除 /* */ 注释
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.S)
        
        data = json.loads(json_str)
        return data

def write_json(data, file):
    with open(file, "w") as f:
        json.dump(data, f, indent=4)
    
def write_yaml(data, output_file):
    with open(output_file, 'w') as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False)
        
def remove_unannotated_images(coco_annotation_file):
    # 检查 coco_annotation_file 是否为字典类型
    if not isinstance(coco_annotation_file, dict):
        coco_data = read_json(coco_annotation_file)
    else:
        coco_data = coco_annotation_file

    # 获取有标注的图片 ID
    annotated_image_ids = {ann['image_id'] for ann in coco_data['annotations']}
    
    # 过滤有标注的图片
    annotated_images = [img for img in coco_data['images'] if img['id'] in annotated_image_ids]
    coco_data['images'] = annotated_images
    
    # 过滤没有标注的图片的标注
    annotated_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in annotated_image_ids]
    coco_data['annotations'] = annotated_annotations
    
    return coco_data