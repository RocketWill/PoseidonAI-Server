import json

def validate_coco_format(json_data):
    try:
        data = json.loads(json_data)
        
        # 必须包含这些顶级字段
        required_fields = ['images', 'annotations', 'categories']
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # 进一步验证字段内容的格式
        # 这里可以根据具体的 COCO 数据集格式要求进行更详细的验证
        
        return True, data
    
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON format: {str(e)}"
    except Exception as e:
        return False, str(e)

def find_valid_images(image_names, coco_data):
    valid_images = []
    
    # 验证图像是否在 COCO 数据集中有对应的标注
    for image_name in image_names:
        found = False
        for image in coco_data['images']:
            if image['file_name'] == image_name:
                found = True
                break
        if found:
            valid_images.append(image_name)
    
    return valid_images

# 示例用法
if __name__ == "__main__":
    # 示例输入
    image_list = ['image1.jpg', 'image2.jpg', 'image3.jpg']
    
    # 读取 JSON 文件
    with open('your_coco_dataset.json', 'r') as f:
        json_data = f.read()
    
    # 验证格式
    valid, coco_data = validate_coco_format(json_data)
    if valid:
        # 查找有效的图像
        valid_images = find_valid_images(image_list, coco_data)
        print("Valid images found in COCO dataset:")
        print(valid_images)
    else:
        print(f"Validation failed: {coco_data}")  # coco_data 将包含错误信息
