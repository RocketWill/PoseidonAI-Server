'''
Author: Will Cheng chengyong@pku.edu.cn
Date: 2024-07-26 16:59:09
LastEditors: Will Cheng chengyong@pku.edu.cn
LastEditTime: 2024-07-27 12:01:23
FilePath: /PoseidonAI-Server/utils/training_configuration/tools/utils.py
Description: 

Copyright (c) 2024 by chengyong@pku.edu.cn, All Rights Reserved. 
'''
import json
import re

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
        
def convert_to_official_types(user_config, official_config):
    converted_config = {}
    for key, value in user_config.items():
        try:
            if key in official_config:
                official_type = type(official_config[key])
                if official_type == bool:
                    converted_config[key] = value.lower() in ("true", "1")
                else:
                    converted_config[key] = official_type(value)
            else:
                converted_config[key] = value  # Keep custom keys as they are
        except:
            converted_config[key] = value
    return converted_config