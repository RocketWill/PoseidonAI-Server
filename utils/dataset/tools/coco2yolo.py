import json

import cv2
from collections import defaultdict
from pycocotools import mask

from .coco2yolo_utils import *

def convert_coco_json(json_file='../coco/annotations/instances_default.json', output_path="", use_segments=False, use_keypoints=False):
    save_dir = make_dirs(output_path)  # output directory

    try:
        # Check if the JSON file exists
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"JSON file '{json_file}' not found.")

        # Import JSON
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Create class mapping from the categories in the JSON file
        categories = data['categories']
        class_map = {category['id']: idx for idx, category in enumerate(categories)}
        print(f"Generated class map: {class_map}")

        # Create image dict
        images = {'%g' % x['id']: x for x in data['images']}
        # Create image-annotations dict
        imgToAnns = defaultdict(list)
        for ann in data['annotations']:
            imgToAnns[ann['image_id']].append(ann)

        # Write labels file
        compliant_files_count = 0
        for img_id, anns in tqdm(imgToAnns.items(), desc=f'Annotations {json_file}'):
            img = images['%g' % img_id]
            h, w, f = img['height'], img['width'], img['file_name']
            f = f.split('/')[-1]
            
            bboxes = []
            segments = []
            keypoints = []
            for ann in anns:
                # Use the dynamically generated class ID mapping
                cls = class_map[ann['category_id']]

                if len(ann['bbox']) == 0:
                    box = bbox_from_keypoints(ann)
                else:
                    box = np.array(ann['bbox'], dtype=np.float64)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= w
                box[[1, 3]] /= h
                if box[2] <= 0 or box[3] <= 0:
                    continue

                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)

                if use_segments:
                    if len(ann['segmentation']) == 0:
                        segments.append([])
                        continue
                    if isinstance(ann['segmentation'], dict):
                        ann['segmentation'] = rle2polygon(ann['segmentation'])
                    if len(ann['segmentation']) > 1:
                        s = merge_multi_segment(ann['segmentation'])
                        s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                    else:
                        s = [j for i in ann['segmentation'] for j in i]
                        s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                    s = [cls] + s
                    if s not in segments:
                        segments.append(s)

            # Write to file
            with open((save_dir / f).with_suffix('.txt'), 'a') as file:
                for i in range(len(bboxes)):
                    if use_keypoints:
                        line = *(keypoints[i]),
                    else:
                        try:
                            line = *(segments[i] if use_segments and len(segments[i]) > 0 else bboxes[i]),
                            file.write(('%g ' * len(line)).rstrip() % line + '\n')
                        except Exception as e:
                            print(f"Error writing annotation for image {img_id}: {e}")

            compliant_files_count += 1

        print(f"Conversion completed. Total {compliant_files_count} compliant annotation files written.")
        return compliant_files_count

    except FileNotFoundError as e:
        raise ValueError(e)
    except Exception as e:
        raise ValueError(e)

def bbox_from_keypoints(ann):
    if 'keypoints' not in ann:
        return
    k = np.array(ann['keypoints']).reshape(-1, 3)
    x_list, y_list, v_list = zip(*k)
    box = [min(x_list), min(y_list), max(x_list) - min(x_list), max(y_list) - min(y_list)]
    return np.array(box, dtype=np.float64)

def show_kpt_shape_flip_idx(data):
    for category in data['categories']:
        if 'keypoints' not in category:
            continue
        keypoints = category['keypoints']
        num = len(keypoints)
        print('kpt_shape: [' + str(num) + ', 3]')
        flip_idx = list(range(num))
        for i, name in enumerate(keypoints):
            name = name.lower()
            left_pos = name.find('left')
            if left_pos < 0:
                continue
            name_right = name.replace('left', 'right')
            for j, namej in enumerate(keypoints):
                namej = namej.lower()
                if namej == name_right:
                    flip_idx[i] = j
                    flip_idx[j] = i
                    break
        print('flip_idx: [' + ', '.join(str(x) for x in flip_idx) + ']')


def is_clockwise(contour):
    value = 0
    num = len(contour)
    for i, point in enumerate(contour):
        p1 = contour[i]
        if i < num - 1:
            p2 = contour[i + 1]
        else:
            p2 = contour[0]
        value += (p2[0][0] - p1[0][0]) * (p2[0][1] + p1[0][1]);
    return value < 0

def get_merge_point_idx(contour1, contour2):
    idx1 = 0
    idx2 = 0
    distance_min = -1
    for i, p1 in enumerate(contour1):
        for j, p2 in enumerate(contour2):
            distance = pow(p2[0][0] - p1[0][0], 2) + pow(p2[0][1] - p1[0][1], 2);
            if distance_min < 0:
                distance_min = distance
                idx1 = i
                idx2 = j
            elif distance < distance_min:
                distance_min = distance
                idx1 = i
                idx2 = j
    return idx1, idx2

def merge_contours(contour1, contour2, idx1, idx2):
    contour = []
    for i in list(range(0, idx1 + 1)):
        contour.append(contour1[i])
    for i in list(range(idx2, len(contour2))):
        contour.append(contour2[i])
    for i in list(range(0, idx2 + 1)):
        contour.append(contour2[i])
    for i in list(range(idx1, len(contour1))):
        contour.append(contour1[i])
    contour = np.array(contour)
    return contour

def merge_with_parent(contour_parent, contour):
    if not is_clockwise(contour_parent):
        contour_parent = contour_parent[::-1]
    if is_clockwise(contour):
        contour = contour[::-1]
    idx1, idx2 = get_merge_point_idx(contour_parent, contour)
    return merge_contours(contour_parent, contour, idx1, idx2)

def mask2polygon(image):
    contours, hierarchies = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    contours_approx = []
    polygons = []
    for contour in contours:
        epsilon = 0.001 * cv2.arcLength(contour, True)
        contour_approx = cv2.approxPolyDP(contour, epsilon, True)
        contours_approx.append(contour_approx)

    contours_parent = []
    for i, contour in enumerate(contours_approx):
        parent_idx = hierarchies[0][i][3]
        if parent_idx < 0 and len(contour) >= 3:
            contours_parent.append(contour)
        else:
            contours_parent.append([])

    for i, contour in enumerate(contours_approx):
        parent_idx = hierarchies[0][i][3]
        if parent_idx >= 0 and len(contour) >= 3:
            contour_parent = contours_parent[parent_idx]
            if len(contour_parent) == 0:
                continue
            contours_parent[parent_idx] = merge_with_parent(contour_parent, contour)

    contours_parent_tmp = []
    for contour in contours_parent:
        if len(contour) == 0:
            continue
        contours_parent_tmp.append(contour)

    polygons = []
    for contour in contours_parent_tmp:
        polygon = contour.flatten().tolist()
        polygons.append(polygon)
    return polygons 

def rle2polygon(segmentation):
    if isinstance(segmentation["counts"], list):
        segmentation = mask.frPyObjects(segmentation, *segmentation["size"])
    m = mask.decode(segmentation) 
    m[m > 0] = 255
    polygons = mask2polygon(m)
    return polygons

def min_index(arr1, arr2):
    """Find a pair of indexes with the shortest distance. 
    Args:
        arr1: (N, 2).
        arr2: (M, 2).
    Return:
        a pair of indexes(tuple).
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments):
    """Merge multi segments to one list.
    Find the coordinates with min distance between each segment,
    then connect these coordinates with one thin line to merge all 
    segments into one.

    Args:
        segments(List(List)): original segmentations in coco's json file.
            like [segmentation1, segmentation2,...], 
            each segmentation is a list of coordinates.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0]:idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


def delete_dsstore(path='../datasets'):
    # Delete apple .DS_store files
    from pathlib import Path
    files = list(Path(path).rglob('.DS_store'))
    print(files)
    for f in files:
        f.unlink()


if __name__ == '__main__':
    source = 'COCO'

    if source == 'COCO':
        convert_coco_json('/mnt/d/workspace/sam-labeling/dataset/instances_default.json',  # directory with *.json
                          '/mnt/d/workspace/sam-labeling/dataset/yolo',
                          use_segments=True,
                          use_keypoints=False,
                          cls91to80=False)
