'''
Author: Will Cheng chengyong@pku.edu.cn
Date: 2024-07-27 13:16:53
LastEditors: Will Cheng (will.cheng@efctw.com)
LastEditTime: 2024-08-05 16:25:27
FilePath: /PoseidonAI-Server/routes/utils.py
Description: 

Copyright (c) 2024 by chengyong@pku.edu.cn, All Rights Reserved. 
'''
from collections import defaultdict
import re


def parse_immutable_dict(immutable_dict):
    parsed_dict = defaultdict(list)
    
    for key, value in immutable_dict.items():
        if '.' in key and not key.endswith('.pt'):
            base_key, index = key.rsplit('.', 1)
            parsed_dict[base_key].append(value)
        else:
            parsed_dict[key] = value
    
    # Regular expressions to match integers and floats
    int_pattern = re.compile(r'^\d+$')
    float_pattern = re.compile(r'^\d+\.\d+$')
    
    # Convert defaultdict to regular dict and parse values to the correct type
    final_dict = {}
    for key, value in parsed_dict.items():
        if isinstance(value, list) and len(value) == 1:
            final_dict[key] = value[0]
        elif isinstance(value, list):
            final_dict[key] = [float(v) if float_pattern.match(v) else int(v) if int_pattern.match(v) else v for v in value]
        else:
            if float_pattern.match(value):
                final_dict[key] = float(value)
            elif int_pattern.match(value):
                final_dict[key] = int(value)
            else:
                final_dict[key] = value
    
    return final_dict