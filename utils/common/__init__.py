'''
Author: Will Cheng chengyong@pku.edu.cn
Date: 2024-07-26 16:59:09
LastEditors: Will Cheng chengyong@pku.edu.cn
LastEditTime: 2024-08-03 21:32:00
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
        yaml.dump(data, file, default_flow_style=False)