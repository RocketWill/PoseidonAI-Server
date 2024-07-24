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

app.register_blueprint(auth_bp, url_prefix='/api/auth')
app.register_blueprint(datasets_bp, url_prefix='/api/datasets')
app.register_blueprint(detect_types_bp, url_prefix='/api/detect-types')
app.register_blueprint(dataset_formats_bp, url_prefix='/api/dataset-formats')


if __name__ == '__main__':
    app.run(debug=True)
