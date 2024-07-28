'''
Author: Will Cheng chengyong@pku.edu.cn
Date: 2024-07-27 13:16:53
LastEditors: Will Cheng chengyong@pku.edu.cn
LastEditTime: 2024-07-27 13:16:56
FilePath: /PoseidonAI-Server/routes/utils.py
Description: 

Copyright (c) 2024 by chengyong@pku.edu.cn, All Rights Reserved. 
'''
from collections import defaultdict

def parse_immutable_dict(immutable_dict):
    parsed_dict = defaultdict(list)
    
    for key, value in immutable_dict.items():
        if '.' in key:
            base_key, index = key.rsplit('.', 1)
            parsed_dict[base_key].append(value)
        else:
            parsed_dict[key] = value
    
    # Convert defaultdict to regular dict and parse values to the correct type
    final_dict = {}
    for key, value in parsed_dict.items():
        if isinstance(value, list) and len(value) == 1:
            final_dict[key] = value[0]
        elif isinstance(value, list):
            final_dict[key] = [float(v) if '.' in v else int(v) for v in value]
        else:
            final_dict[key] = float(value) if '.' in value else int(value) if value.isdigit() else value
    
    return final_dict