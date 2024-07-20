import os
from datetime import timedelta

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

class Config:
    STATIC_FOLDER = os.path.abspath(os.path.join(current_dir, os.pardir, 'static'))
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your_secret_key_here'
    MONGO_URI = 'mongodb://admin:admin@localhost:27017/poseidon?authSource=admin&retryWrites=true&w=majority'
    SECRET_KEY = 'your_secret_key_here'  # 用于签署 JWT 的密钥
    JWT_SECRET_KEY = 'jwt_secret_key'  # JWT 签名的密钥
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)  # 访问令牌过期时间