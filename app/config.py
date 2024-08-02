'''
Author: Will Cheng chengyong@pku.edu.cn
Date: 2024-07-24 16:53:06
LastEditors: Will Cheng (will.cheng@efctw.com)
LastEditTime: 2024-08-02 13:46:17
FilePath: /PoseidonAI-Server/app/config.py
Description: 

Copyright (c) 2024 by chengyong@pku.edu.cn, All Rights Reserved. 
'''
import os
from datetime import timedelta
from os.path import abspath, dirname, join

current_file_path = abspath(__file__)
current_dir = dirname(current_file_path)

class Config:
    STATIC_FOLDER = abspath(join(current_dir, os.pardir, 'data', 'static'))
    DATASET_RAW_FOLDER = abspath(join(current_dir, os.pardir, 'data', 'dataset_raw'))
    TRAINING_CONFIGS_FOLDER = abspath(join(current_dir, os.pardir, 'data', 'configs'))
    TRAINING_PROJECT_FOLDER = abspath(join(current_dir, os.pardir, 'data', 'projects'))
    
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your_secret_key_here'
    MONGO_URI = 'mongodb://admin:admin@localhost:27017/poseidon?authSource=admin&retryWrites=true&w=majority'
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or 'jwt_secret_key'  # JWT 签名的密钥
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=48)  # 访问令牌过期时间
