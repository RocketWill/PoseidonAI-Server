import os

from .tools.verify_coco_format import validate_coco_format, find_valid_images
from .tools.coco2yolo import convert_coco_json

def create_dataset_helper(dataset_raw_root, user_id, save_key, dataset_format, detect_type, r_image_list, label_file, image_files):
    output_image_dir = os.path.join(dataset_raw_root, user_id, save_key, 'images')
    output_coco_dir = os.path.join(dataset_raw_root, user_id, save_key, 'mscoco')
    valid_coco_images = create_coco(output_coco_dir, output_image_dir, r_image_list, label_file, image_files)
    if 'yolo' in dataset_format:
        output_yolo_dir = os.path.join(dataset_raw_root, user_id, save_key, 'yolo')
        use_segments = True if 'seg' in detect_type else False
        use_keypoints = True if 'kpts' in detect_type else False
        label_file = os.path.join(output_coco_dir, label_file.filename)
        valid_yolo_images = create_yolo(label_file, output_yolo_dir, use_segments, use_keypoints)
        if len(valid_coco_images) != valid_yolo_images:
            print('valid images are different, coco: {}, yolo: {}.'.format(len(valid_coco_images), valid_yolo_images))
        return min(len(valid_coco_images), valid_yolo_images)
    return len(valid_coco_images)
            
def create_coco(output_coco_dir, output_image_dir, r_image_list, label_file, image_files):
    os.makedirs(output_coco_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)
    label_file_save_path = os.path.join(output_coco_dir, label_file.filename)
    label_file.save(label_file_save_path)
    with open(label_file_save_path, 'r') as f:
        json_data = f.read()
    # 验证格式
    valid, coco_data = validate_coco_format(json_data)
    if valid:
        valid_images = find_valid_images(r_image_list, coco_data)
    else:
        raise ValueError('不合規的 COCO 標注文件')
    
    for file in image_files:
        if file:
            filename = file.filename
            file.save(os.path.join(output_image_dir, filename))
    return valid_images

def create_yolo(label_file, output_labels_dir, use_segments, use_keypoints):
    return convert_coco_json(label_file, output_labels_dir, use_segments, use_keypoints, cls91to80=False)