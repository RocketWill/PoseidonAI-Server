import os
import sys
import zipfile
import tempfile
import shutil
from PIL import Image

def is_image_file(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False

def validate_directory_structure(zip_file):
    # 忽略的文件和目录
    ignored_files = ['.DS_Store', '__MACOSX']
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')

    # 使用临时目录解压文件
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 解压压缩包到临时目录
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)

        # 检查第0级目录
        top_level_items = [item for item in os.listdir(tmp_dir) if item not in ignored_files]
        if len(top_level_items) != 1 or not os.path.isdir(os.path.join(tmp_dir, top_level_items[0])):
            return False, "第0級目錄不合格：解壓後應該只有一個目錄。"

        # 获取第0级目录路径
        top_level_dir = os.path.join(tmp_dir, top_level_items[0])

        # 检查第1级目录
        first_level_dirs = [d for d in os.listdir(top_level_dir) if os.path.isdir(os.path.join(top_level_dir, d)) and d not in ignored_files]
        if len(first_level_dirs) == 0:
            return False, "第1級目錄不合格：應至少有一個子目錄。"

        # 检查第1级目录下的内容
        for first_level_dir in first_level_dirs:
            full_first_level_path = os.path.join(top_level_dir, first_level_dir)

            # 第1级目录下不能有任何文件
            first_level_items = [f for f in os.listdir(full_first_level_path) if f not in ignored_files]

            # 检查第2级目录中的文件，必须全部是图片，且不能包含子目录
            for item in first_level_items:
                item_path = os.path.join(full_first_level_path, item)

                if os.path.isdir(item_path):
                    return False, f"第2級目錄 '{item}' 不合格：應只包含圖片文件，不能有子目錄。"

                if not item.lower().endswith(image_extensions):
                    return False, f"第2級目錄 '{item}' 不合格：應只包含圖片文件。"

    # 如果所有检查通过
    return True, ""
    
def unzip_skip_first_level(zip_file, output_dir):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # 获取压缩文件中的所有文件名列表
        all_files = zip_ref.namelist()

        # 找到第0级目录名（假设只有一个顶层目录）
        top_level_dir = all_files[0].split('/')[0] + '/'

        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 解压时跳过第0级目录
        for file in all_files:
            # 去掉第0级目录路径
            file_without_top = file[len(top_level_dir):]

            if file_without_top:  # 确保不是空字符串
                output_path = os.path.join(output_dir, file_without_top)
                if file.endswith('/'):
                    # 如果是目录，则创建目录
                    os.makedirs(output_path, exist_ok=True)
                else:
                    # 如果是文件，解压文件
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with open(output_path, 'wb') as f_out:
                        f_out.write(zip_ref.read(file))

def get_image_filenames(directory):
    # 定义常见的图片文件扩展名
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')

    image_filenames = []

    # 使用 os.walk 遍历目录和子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):  # 检查文件是否为图片
                image_filenames.append(file)  # 只存储文件的基本名称

    return image_filenames
