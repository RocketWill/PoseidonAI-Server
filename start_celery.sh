celery -A run.celery worker --loglevel=info -E --concurrency=1

# celery -A run.celery worker --loglevel=info -E --pool solo
# 不要使用--pool=solo，1. 進程殺不掉 2. 訓練後顯存無法釋放