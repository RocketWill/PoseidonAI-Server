'''
Author: Will Cheng chengyong@pku.edu.cn
Date: 2024-07-26 16:53:13
LastEditors: Will Cheng chengyong@pku.edu.cn
LastEditTime: 2024-07-27 21:50:53
FilePath: /PoseidonAI-Server/utils/training_configuration/tools/yolov8.py
Description: 

Copyright (c) 2024 by chengyong@pku.edu.cn, All Rights Reserved. 
'''
import os
from os.path import abspath, dirname
from copy import deepcopy

from .utils import read_json, write_json, convert_to_official_types

current_file_path = abspath(__file__)
current_dir = dirname(current_file_path)
data_dir = os.path.join(current_dir, os.pardir, 'data', 'configs', 'yolov8')

system_train_settings_file = \
    abspath(os.path.join(data_dir, 'system-train-settings.json'))

system_aug_settings_file = \
    abspath(os.path.join(data_dir, 'system-augmentation-settings.json'))
    
user_train_settings_file = \
    abspath(os.path.join(data_dir, 'user-train-settings.json'))
    
user_aug_settings_file = \
    abspath(os.path.join(data_dir, 'user-augmentation-settings.json'))
    
def save_config_file(user_config_content, output_path):
    system_train_data = read_json(system_train_settings_file)
    system_aug_data = read_json(system_aug_settings_file)
    user_train_data = read_json(user_train_settings_file)
    user_aug_data = read_json(user_aug_settings_file)
    default_data = {}
    default_data.update(system_train_data)
    default_data.update(system_aug_data)
    default_data.update(user_train_data)
    default_data.update(user_aug_data)
    args_data = deepcopy(default_data)
    args_data.update(user_config_content)
    args_data = convert_to_official_types(args_data, default_data)
    write_json(args_data, output_path)
    
    saved_args = deepcopy(user_train_data)
    saved_args.update(user_aug_data)
    saved_args.update(user_config_content)
    saved_args = convert_to_official_types(saved_args, default_data)
    return saved_args
    
