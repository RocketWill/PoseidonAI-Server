'''
Author: Will Cheng (will.cheng@efctw.com)
Date: 2024-07-29 08:28:38
LastEditors: Will Cheng (will.cheng@efctw.com)
LastEditTime: 2024-08-08 09:56:37
FilePath: /PoseidonAI-Server/app/__init__.py
'''
from flask import Flask
from flask_pymongo import PyMongo
from flask_jwt_extended import JWTManager
from flask_cors import CORS
from app.config import Config
from app.celery_utils import create_celery_app
from celery import Celery, platforms
from flask_redis import FlaskRedis


mongo = PyMongo()
jwt = JWTManager()
app = Flask(__name__, static_folder=Config.STATIC_FOLDER, static_url_path='/static')
app.config.from_object(Config)
app.config['SECRET_KEY'] = Config.SECRET_KEY
app.config['JWT_SECRET_KEY'] = Config.JWT_SECRET_KEY
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = Config.JWT_ACCESS_TOKEN_EXPIRES
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['REDIS_URL'] = Config.REDIS_URL
mongo.init_app(app)
jwt.init_app(app)
redis_client = FlaskRedis(app)

def create_app():
    app = Flask(__name__, static_folder=Config.STATIC_FOLDER, static_url_path='/static')
    app.config.from_object(Config)
    app.config['SECRET_KEY'] = Config.SECRET_KEY
    app.config['JWT_SECRET_KEY'] = Config.JWT_SECRET_KEY
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = Config.JWT_ACCESS_TOKEN_EXPIRES
    app.config['CORS_HEADERS'] = 'Content-Type'
    app.config['REDIS_URL'] = Config.REDIS_URL

    # Initialize extensions
    mongo.init_app(app)
    jwt.init_app(app)
    redis_client = FlaskRedis(app)
    CORS(app)
    
    return app
