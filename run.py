'''
Author: Will Cheng (will.cheng@efctw.com)
Date: 2024-07-29 08:28:38
LastEditors: Will Cheng (will.cheng@efctw.com)
LastEditTime: 2024-10-21 13:43:52
FilePath: /PoseidonAI-Server/run.py
'''
from app import create_app
from app.celery_utils import create_celery_app
from flask import jsonify
import time
from celery import task

app = create_app()
celery = create_celery_app(app)

# Register blueprints
from routes.auth import auth_bp
from routes.datasets import datasets_bp
from routes.detect_types import detect_types_bp
from routes.dataset_formats import dataset_formats_bp
from routes.training_configurations import training_configurations_bp
from routes.training_frameworks import training_frameworks_bp
from routes.algorithms import algorithms_bp
from routes.training_task import training_tasks_bp
from routes.user_logs import user_logs_bp

app.register_blueprint(auth_bp, url_prefix='/api/auth')
app.register_blueprint(datasets_bp, url_prefix='/api/datasets')
app.register_blueprint(detect_types_bp, url_prefix='/api/detect-types')
app.register_blueprint(dataset_formats_bp, url_prefix='/api/dataset-formats')
app.register_blueprint(training_configurations_bp, url_prefix='/api/training-configurations')
app.register_blueprint(training_frameworks_bp, url_prefix='/api/training-frameworks')
app.register_blueprint(algorithms_bp, url_prefix='/api/algorithms')
app.register_blueprint(training_tasks_bp, url_prefix='/api/training-tasks')
app.register_blueprint(user_logs_bp, url_prefix='/api/user-logs')

if __name__ == '__main__':
    app.run(debug=True)
