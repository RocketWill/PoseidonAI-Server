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

app.register_blueprint(auth_bp, url_prefix='/api/auth')
app.register_blueprint(datasets_bp, url_prefix='/api/datasets')



if __name__ == '__main__':
    app.run(debug=True)
