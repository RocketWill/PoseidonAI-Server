import os
from datetime import timedelta

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your_secret_key_here'
    MONGO_URI = 'mongodb://admin:admin@localhost:27017/poseidon?authSource=admin&retryWrites=true&w=majority'
    SECRET_KEY = 'your_secret_key_here'  # 用于签署 JWT 的密钥
    JWT_SECRET_KEY = 'jwt_secret_key'  # JWT 签名的密钥
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)  # 访问令牌过期时间