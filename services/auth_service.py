from app.models import User
from app import jwt
from werkzeug.security import check_password_hash
from flask import jsonify
from flask_jwt_extended import create_access_token, JWTManager


class AuthService:
    @staticmethod
    def register_user(username, password, email):
        # 检查用户名或邮箱是否已经存在
        existing_user = User.find_by_username(username)
        if existing_user:
            return jsonify({'message': '用户名已存在'}), 400
        
        existing_email = User.find_by_email(email)
        if existing_email:
            return jsonify({'message': '邮箱已被注册'}), 400

        # 创建新用户
        user = User(username=username, password_hash=password, email=email)
        user.save()
        return jsonify({'message': '用户注册成功'}), 201

    @staticmethod
    def find_user_by_username(username):
        return User.find_by_username(username)
    
    @staticmethod
    def find_user_by_id(user_id):
        return User.find_by_id(user_id)

    @staticmethod
    def authenticate_user(username, password):
        user = User.find_by_username(username)
        if user and check_password_hash(user.password_hash, password):
            return user
        return None

    @staticmethod
    def login_user(username, password):
        user = AuthService.authenticate_user(username, password)
        if user:
            # 创建访问令牌
            access_token = create_access_token(identity=str(user._id))
            return jsonify({
                'message': 'Login successful',
                'access_token': access_token,
                'username': user.username,
                'email': user.email,
                'status': 'ok'
            }), 200
        else:
            return jsonify({'error': 'Invalid username or password'}), 401
    
    @staticmethod
    def logout_user(user_id):
        # 在实际应用中，通常是删除或失效用户的认证令牌或会话信息
        # 这里仅为示例，清除会话信息
        user = AuthService.find_user_by_id(user_id)
        user.update_last_login()
        return jsonify({'message': '用户已退出'}), 200