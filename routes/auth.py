from bson.objectid import ObjectId
from flask import Blueprint, request, jsonify, session
from services.auth_service import AuthService
from functools import wraps
from flask_jwt_extended import decode_token, get_jwt_identity
from flask_jwt_extended.exceptions import JWTDecodeError

auth_bp = Blueprint('auth', __name__)

def jwt_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        authorization_header = request.headers.get('Authorization')

        if not authorization_header or not authorization_header.startswith('Bearer '):
            return jsonify({"message": "Missing or incorrect Authorization header"}), 401

        jwt_token = authorization_header.split(' ')[1]

        try:
            decoded_token = decode_token(jwt_token)
            user_id = decoded_token.get('sub')
            kwargs['user_id'] = user_id  # 将用户ID传递给被装饰函数
        except JWTDecodeError:
            return jsonify({"message": "Invalid JWT token"}), 401

        return fn(*args, **kwargs)

    return wrapper

@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    email = data.get('email')

    if not username or not password or not email:
        return jsonify({'error': 'Missing username, password, or email'}), 400

    if AuthService.find_user_by_username(username):
        return jsonify({'error': 'Username already exists'}), 400

    AuthService.register_user(username, password, email)
    return jsonify({'message': 'User registered successfully'}), 201

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    print(data)
    username = data.get('username')
    password = data.get('password')
    email = data.get('email')

    if not username or not password:
        return jsonify({'error': 'Missing username or password', 'status': 'failed'}), 400

    return AuthService.login_user(username, password)
    # if user:
    #     # 在这里可以设置用户的会话信息或生成认证令牌等
    #     session['user_id'] = str(user._id)  # 示例中使用 Flask 的 session 保存用户 ID
    #     return jsonify({
    #         'message': 'Login successful',
    #         'username': user.username,
    #         'email': user.email
    #     }), 200
    # else:
    #     return jsonify({'error': 'Invalid username or password'}), 401

@auth_bp.route('/logout', methods=['POST'])
@jwt_required
def logout(user_id):
    # 在实际应用中，通常是删除或失效用户的认证令牌或会话信息
    # 这里仅为示例，清除会话信息
    # session.clear()
    AuthService.logout_user(ObjectId(user_id))
    return jsonify({'message': 'User logged out'}), 200


@auth_bp.route('/profile', methods=['GET'])
@jwt_required
def profile(user_id):
    # dataset_id = ObjectId('60965e2c7c54665d2c7bca8b')
    user = AuthService.find_user_by_id(ObjectId(user_id))
    if not user:
        return jsonify({'error': 'User not found'}), 404

    return jsonify({
        'data': {
            'name': user.username,
            'email': user.email,
            'created_at': user.created_at,
            'last_login': user.last_login,
            'datasets': user.datasets,
            'tags': [],
            'userid': user_id
        },
        'status': 'ok'
    }), 200
