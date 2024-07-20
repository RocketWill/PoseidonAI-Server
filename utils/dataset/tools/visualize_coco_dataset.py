import os
import random
import json
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

def random_color():
    levels = range(32, 256, 32)
    return tuple(random.choice(levels) for _ in range(3))

def draw_annotations(image_dir, annotation_file, output_dir, draw_mask=True, draw_bbox=True):
    try:
        # Load COCO annotation file
        coco = COCO(annotation_file)

        # Create output directory if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for img_id in coco.getImgIds():
            img_info = coco.loadImgs(img_id)[0]
            img_path = os.path.join(image_dir, img_info['file_name'])

            if not os.path.exists(img_path):
                print(f"Image {img_path} not found, skipping.")
                continue

            image = cv2.imread(img_path)
            if image is None:
                print(f"Could not read image {img_path}, skipping.")
                continue

            ann_ids = coco.getAnnIds(imgIds=img_info['id'])
            anns = coco.loadAnns(ann_ids)

            for ann in anns:
                color = random_color()
                category_id = ann['category_id']
                category_name = coco.loadCats(category_id)[0]['name']
                
                if draw_bbox and 'bbox' in ann:
                    bbox = ann['bbox']
                    x, y, w, h = map(int, bbox)
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    (text_width, text_height), baseline = cv2.getTextSize(category_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(image, (x, y - text_height - baseline), (x + text_width, y), (0, 0, 0), -1)
                    cv2.putText(image, category_name, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if draw_mask and 'segmentation' in ann:
                    if isinstance(ann['segmentation'], list):  # Polygon
                        for seg in ann['segmentation']:
                            poly = np.array(seg).reshape((int(len(seg) / 2), 2)).astype(np.int32)
                            overlay = image.copy()
                            cv2.fillPoly(overlay, [poly], color + (int(0.7 * 255),))
                            cv2.addWeighted(overlay, 0.7, image, 1 - 0.7, 0, image)
                            cv2.polylines(image, [poly], isClosed=True, color=color, thickness=2)
                    else:  # RLE
                        rle = ann['segmentation']
                        if isinstance(rle['counts'], list):
                            rle = maskUtils.frPyObjects(rle, rle['size'][0], rle['size'][1])
                        m = maskUtils.decode(rle)
                        m = (m * 255).astype(np.uint8)
                        contours, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                            overlay = image.copy()
                            cv2.fillPoly(overlay, [contour], color + (int(0.7 * 255),))
                            cv2.addWeighted(overlay, 0.7, image, 1 - 0.7, 0, image)
                            cv2.drawContours(image, [contour], -1, color, 2)
            
            output_path = os.path.join(output_dir, img_info['file_name'])
            cv2.imwrite(output_path, image)
            print(f"Saved annotated image to {output_path}")
    except Exception as e:
        raise ValueError(e)

if __name__ == "__main__":
    image_directory = 'image'
    coco_annotation_file = '/mnt/d/workspace/general/plot_coco_dataset/instances_val.json'
    output_directory = 'res'
    draw_masks = True  # Set to False if you don't want to draw masks
    draw_bboxes = False  # Set to True if you want to draw bounding boxes

    draw_annotations(image_directory, coco_annotation_file, output_directory, draw_mask=draw_masks, draw_bbox=draw_bboxes)
