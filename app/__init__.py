from flask import Flask
from flask_pymongo import PyMongo
from flask_jwt_extended import JWTManager
from flask_cors import CORS
from app.config import Config
from app.celery_utils import create_celery_app
from celery import Celery, platforms


mongo = PyMongo()
jwt = JWTManager()

def create_app():
    app = Flask(__name__, static_folder=Config.STATIC_FOLDER, static_url_path='/static')
    app.config.from_object(Config)
    app.config['SECRET_KEY'] = Config.SECRET_KEY
    app.config['JWT_SECRET_KEY'] = Config.JWT_SECRET_KEY
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = Config.JWT_ACCESS_TOKEN_EXPIRES
    app.config['CORS_HEADERS'] = 'Content-Type'

    # Initialize extensions
    mongo.init_app(app)
    jwt.init_app(app)
    CORS(app)
    
    return app
