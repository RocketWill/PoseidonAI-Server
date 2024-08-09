'''
Author: Will Cheng (will.cheng@efctw.com)
Date: 2024-07-29 08:28:38
LastEditors: Will Cheng (will.cheng@efctw.com)
LastEditTime: 2024-08-08 15:14:33
FilePath: /PoseidonAI-Server/app/celery_config.py
'''
# celery_config.py
from datetime import timedelta

class CeleryConfig:
    broker_url = 'redis://127.0.0.1:6379/0'  # 使用 Redis 作为 broker
    result_backend = 'redis://127.0.0.1:6379/0'  # 使用 Redis 作为结果后端
    task_serializer = 'json'
    result_serializer = 'json'
    accept_content = ['json']
    timezone = 'UTC'
    enable_utc = True
    beat_schedule = {
        'add-every-30-seconds': {
            'task': 'tasks.add',
            'schedule': timedelta(seconds=30),
            'args': (16, 16)
        },
    }
    worker_max_memory_per_child=500000 # 500MB
