from flask import Flask
from flask_pymongo import PyMongo
from .config import Config
from flask_jwt_extended import JWTManager
from flask_cors import CORS, cross_origin


jwt = JWTManager()
mongo = PyMongo()
app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config.from_object(Config)
app.config['SECRET_KEY'] = Config.SECRET_KEY
app.config['JWT_SECRET_KEY'] = Config.JWT_SECRET_KEY
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = Config.JWT_ACCESS_TOKEN_EXPIRES
jwt.init_app(app)
mongo.init_app(app)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def create_app():
    app = Flask(__name__, static_folder='static', static_url_path='/static')
    app.config.from_object(Config)
    app.config['SECRET_KEY'] = Config.SECRET_KEY
    app.config['JWT_SECRET_KEY'] = Config.JWT_SECRET_KEY
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = Config.JWT_ACCESS_TOKEN_EXPIRES
    cors = CORS(app)
    app.config['CORS_HEADERS'] = 'Content-Type'

    # Initialize extensions
    mongo.init_app(app)
    jwt.init_app(app)

    # Register blueprints
    from routes.auth import auth_bp
    from routes.datasets import datasets_bp
    # from routes.tasks import tasks_bp
    # from routes.models import models_bp

    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(datasets_bp, url_prefix='/api/datasets')
    # app.register_blueprint(tasks_bp, url_prefix='/tasks')
    # app.register_blueprint(models_bp, url_prefix='/models')

    return app
