import uuid
import traceback
from app import mongo
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import pymongo
from bson.objectid import ObjectId

class User:
    def __init__(self, username, password_hash, email, created_at=None, last_login=None, datasets=[]):
        self._id = None
        self.username = username
        self.password_hash = password_hash if 'scrypt:' in str(password_hash) else generate_password_hash(password_hash)
        self.email = email
        self.created_at = created_at if created_at is not None else datetime.utcnow()
        self.last_login = last_login
        self.datasets = datasets

    @staticmethod
    def from_dict(user_data):
        user = User(
            username=user_data['username'],
            password_hash=user_data['password'],
            email=user_data['email'],
            created_at=user_data['created_at'],
            last_login=user_data['last_login'],
            datasets=user_data['datasets']
        )
        user._id = user_data['_id']
        return user

    def save(self):
        try:
            mongo.db.users.insert_one({
                'username': self.username,
                'password': self.password_hash,
                'email': self.email,
                'created_at': self.created_at,
                'last_login': self.last_login,
                'datasets': self.datasets
            })
            return True
        except pymongo.errors.DuplicateKeyError as e:
            # 处理唯一索引重复异常，这里可以记录日志或者进行其他操作
            print(f"Duplicate email '{self.email}'. User data not inserted.")
        return False


    @staticmethod
    def find_by_username(username):
        # 根据用户名查询用户信息
        try:
            user_dict = mongo.db.users.find_one({'username': username})
            return User.from_dict(user_dict)
        except:
            return False
    
    def find_by_email(email):
        # 根据用户郵箱查询用户信息
        try:
            user_dict = mongo.db.users.find_one({'email': email})
            return User.from_dict(user_dict)
        except:
            return False

    @staticmethod
    def find_by_id(user_id):
        # 根据用户 ID 查询用户信息
        try:
            user_dict = mongo.db.users.find_one({'_id': user_id})
            return User.from_dict(user_dict)
        except:
            return False

    def check_password(self, password):
        # 检查密码是否正确
        return check_password_hash(self.password_hash, password)

    def update_last_login(self):
        # 更新最后登录时间为当前时间
        self.last_login = datetime.utcnow()
        mongo.db.users.update_one(
            {'username': self.username},
            {'$set': {'last_login': self.last_login}}
        )

    def add_dataset(self, dataset_id):
        # 添加数据集 ID 到用户的数据集列表中
        self.datasets.append(dataset_id)
        mongo.db.users.update_one(
            {'username': self.username},
            {'$set': {'datasets': self.datasets}}
        )

    def remove_dataset(self, dataset_id):
        # 从用户的数据集列表中移除指定的数据集 ID
        if dataset_id in self.datasets:
            self.datasets.remove(dataset_id)
            mongo.db.users.update_one(
                {'username': self.username},
                {'$set': {'datasets': self.datasets}}
            )


class Dataset:
    def __init__(self, user_id, name, description, detect_types, label_file, image_files, valid_images_num, save_key, format='MSCOCO', created_at=None):
        self._id = None
        self.user_id = user_id
        self.name = name
        self.description = description
        self.format = format
        self.detect_types = detect_types
        self.created_at = datetime.utcnow() if not created_at else created_at  # 数据集上传时间
        self.image_files = image_files  # 数据集文件的路径或存储信息
        self.label_file = label_file
        self.valid_images_num = valid_images_num
        self.save_key = save_key

    def save(self):
        # 将数据集信息存储到 MongoDB 中的 datasets 集合
        try:
            mongo.db.datasets.insert_one({
                'user_id': self.user_id,
                'name': self.name,
                'description': self.description,
                'format': self.format,
                'detect_types': self.detect_types,
                'created_at': self.created_at,
                'image_files': self.image_files,
                'label_file': self.label_file,
                'valid_images_num': self.valid_images_num,
                'save_key': self.save_key
            })
            return True
        except Exception as e:
            print(e)
            return False
    
    @staticmethod
    def from_dict(dataset_data):
        user = Dataset(
            user_id=dataset_data['user_id'],
            name=dataset_data['name'],
            description=dataset_data['description'],
            created_at=dataset_data['created_at'],
            detect_types=dataset_data['detect_types'],
            label_file=dataset_data['label_file'],
            image_files=dataset_data['image_files'],
            valid_images_num=dataset_data['valid_images_num'],
            format=dataset_data['format'],
            save_key=dataset_data['save_key']
        )
        user._id = dataset_data['_id']
        return user

    @staticmethod
    def find_by_id(dataset_id):
        # 根据数据集 ID 查询数据集信息
        try:
            dataset_dict = mongo.db.datasets.find_one({'_id': ObjectId(dataset_id)})
            return Dataset.from_dict(dataset_dict)
        except:
            return False
        

    @staticmethod
    def find_by_user(user_id):
        # 根据用户 ID 查询该用户上传的所有数据集信息
        try:
            datasets = list(mongo.db.datasets.find({'user_id': ObjectId(user_id)}))
            for dataset in datasets:
                dataset['_id'] = str(dataset['_id'])
                dataset['user_id'] = str(dataset['user_id'])
            return datasets
        except:
            print(traceback.print_exc())
            return False

    def update(self):
        # 更新数据集信息
        try:
            mongo.db.datasets.update_one(
                {'_id': self.dataset_id},
                {'$set': {
                    'name': self.name,
                    'description': self.description,
                    'format': self.format,
                    'files': self.files,
                    'image_files': self.image_files,
                    'label_file': self.label_file
                }}
            )
            return True
        except:
            return False

    @staticmethod
    def delete(dataset_id):
        try:
            mongo.db.datasets.delete_one({'_id': ObjectId(dataset_id)})
            return True
        except:
            return False
        

class TrainingTask:
    def __init__(self, user_id, dataset_id, algorithm, task_type, path, status='等待中', logs=None, metrics=None):
        self.user_id = user_id
        self.dataset_id = dataset_id
        self.algorithm = algorithm
        self.task_type = task_type
        self.path = path
        self.status = status
        self.created_at = datetime.utcnow()  # 任务创建时间
        self.logs = logs if logs else []
        self.metrics = metrics if metrics else {}

    def save(self):
        # 将训练任务信息存储到 MongoDB 中的 training_tasks 集合
        mongo.db.training_tasks.insert_one({
            'user_id': self.user_id,
            'dataset_id': self.dataset_id,
            'algorithm': self.algorithm,
            'task_type': self.task_type,
            'path': self.path,
            'status': self.status,
            'created_at': self.created_at,
            'logs': self.logs,
            'metrics': self.metrics
        })

    @staticmethod
    def find_by_id(task_id):
        # 根据任务 ID 查询训练任务信息
        return mongo.db.training_tasks.find_one({'_id': ObjectId(task_id)})

    @staticmethod
    def find_by_user(user_id):
        # 根据用户 ID 查询该用户创建的所有训练任务信息
        return list(mongo.db.training_tasks.find({'user_id': ObjectId(user_id)}))

    def update_status(self, new_status):
        # 更新任务状态
        self.status = new_status
        mongo.db.training_tasks.update_one(
            {'_id': self.task_id},
            {'$set': {'status': self.status}}
        )

    def add_log(self, log_entry):
        # 添加训练日志
        self.logs.append(log_entry)
        mongo.db.training_tasks.update_one(
            {'_id': self.task_id},
            {'$push': {'logs': log_entry}}
        )

    def add_metrics(self, metrics_dict):
        # 添加训练指标
        self.metrics.update(metrics_dict)
        mongo.db.training_tasks.update_one(
            {'_id': self.task_id},
            {'$set': {'metrics': self.metrics}}
        )

    def delete(self):
        # 删除训练任务信息
        mongo.db.training_tasks.delete_one({'_id': self.task_id})


class Model:
    def __init__(self, user_id, name, path, description, algorithm, model_type, files=None):
        self.user_id = user_id
        self.name = name
        self.path = path
        self.description = description
        self.algorithm = algorithm
        self.model_type = model_type
        self.created_at = datetime.utcnow()  # 模型创建时间
        self.files = files if files else []

    def save(self):
        # 将模型信息存储到 MongoDB 中的 models 集合
        mongo.db.models.insert_one({
            'user_id': self.user_id,
            'name': self.name,
            'path': self.path,
            'description': self.description,
            'algorithm': self.algorithm,
            'model_type': self.model_type,
            'created_at': self.created_at,
            'files': self.files
        })

    @staticmethod
    def find_by_id(model_id):
        # 根据模型 ID 查询模型信息
        return mongo.db.models.find_one({'_id': ObjectId(model_id)})

    @staticmethod
    def find_by_user(user_id):
        # 根据用户 ID 查询该用户创建的所有模型信息
        return list(mongo.db.models.find({'user_id': ObjectId(user_id)}))

    def update(self):
        # 更新模型信息
        mongo.db.models.update_one(
            {'_id': self.model_id},
            {'$set': {
                'name': self.name,
                'path': self.path,
                'description': self.description,
                'algorithm': self.algorithm,
                'model_type': self.model_type,
                'files': self.files
            }}
        )

    def delete(self):
        # 删除模型信息
        mongo.db.models.delete_one({'_id': self.model_id})
