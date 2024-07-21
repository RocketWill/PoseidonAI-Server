
from utils.dataset.tools.visualize_coco_dataset import draw_annotations
from celery import task

# @shared_task(ignore_result=False)
@task(name='tasks.draw.annotations')
def draw_annotations_task(image_dir, label_file, vis_dir, draw_mask, draw_bbox):
    draw_annotations(image_dir, label_file, vis_dir, draw_mask=draw_mask, draw_bbox=draw_bbox)
