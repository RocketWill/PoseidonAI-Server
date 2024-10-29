'''
Author: Will Cheng (will.cheng@efctw.com)
Date: 2024-10-21 13:09:09
LastEditors: Will Cheng (will.cheng@efctw.com)
LastEditTime: 2024-10-22 08:55:49
FilePath: /PoseidonAI-Server/services/user_log_service.py
'''
from app.models import UserLog

class UserLogService:
    @staticmethod
    def create(user_id, action, timestamp, level='INFO', details=None, browserInfo=None, url=None, referrer=None, deviceType=None, 
                 language=None, timezone=None, networkType=None, created_at=None):
        user_log = UserLog(user_id, action, timestamp, level, details, browserInfo, url, referrer, deviceType, 
                 language, timezone, networkType, created_at)
        result = user_log.save()
        return result

    @staticmethod
    def get_user_logs(user_id):
        return UserLog.find_by_user(user_id)

    @staticmethod
    def delete_user_logs(user_id):
        return UserLog.delete_by_user(user_id)
