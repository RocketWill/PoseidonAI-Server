'''
Author: Will Cheng (will.cheng@efctw.com)
Date: 2024-07-29 08:28:38
LastEditors: Will Cheng (will.cheng@efctw.com)
LastEditTime: 2024-10-16 16:26:32
FilePath: /PoseidonAI-Server/tasks/dataset.py
'''
import os
import ntpath
import glob

import cv2

from utils.dataset.tools.visualize_coco_dataset import draw_annotations
from celery import task

# @shared_task(ignore_result=False)
@task(name='tasks.draw.annotations')
def draw_annotations_task(image_dir, label_file, vis_dir, draw_mask, draw_bbox):
    draw_annotations(image_dir, label_file, vis_dir, draw_mask=draw_mask, draw_bbox=draw_bbox)


@task(name='tasks.vis.classify.dataset')
def vis_classify_dataset(dataset_dir, vis_dir):
    # 定义常见的图片文件扩展名
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff')

    # 使用 glob 匹配 dataset_dir 下的所有图片
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(dataset_dir, '*', ext)))

    # 确保 vis_dir 存在
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    # 创建软链接
    for image_file in image_files:
        image = cv2.imread(image_file)
        image_filename = os.path.basename(image_file)  # 获取文件名
        category_name = ntpath.basename(os.path.dirname(image_file))
        symlink_path = os.path.join(vis_dir, image_filename)  # 生成软链接路径

        # 如果目标文件已经存在，删除软链接再创建
        if os.path.exists(symlink_path):
            os.remove(symlink_path)

        # os.symlink(image_file, symlink_path)  # 创建软链接
        x, y = 10, 30
        (text_width, text_height), baseline = cv2.getTextSize(category_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(image, (x, y - text_height - baseline), (x + text_width, y), (0, 0, 0), -1)
        cv2.putText(image, category_name, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(symlink_path, image)

    print(f"已将所有图片链接到 {vis_dir}")