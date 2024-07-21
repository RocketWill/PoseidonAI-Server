from celery import Celery
from app.celery_config import CeleryConfig

def create_celery_app(app):
    celery = Celery(app.import_name, backend=CeleryConfig.result_backend, broker=CeleryConfig.broker_url)
    celery.conf.update(app.config)
    celery.config_from_object(CeleryConfig)


    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery
