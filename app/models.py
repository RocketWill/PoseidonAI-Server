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
    def __init__(self, user_id, name, description, detect_type_id, label_file, image_files, valid_images_num, save_key, dataset_format_ids, created_at=None):
        self._id = None
        self.user_id = user_id
        self.name = name
        self.valid_images_num = valid_images_num
        self.description = description
        self.dataset_format_ids = dataset_format_ids
        self.detect_type_id = detect_type_id
        self.created_at = datetime.utcnow() if not created_at else created_at  # 数据集上传时间
        self.image_files = image_files  # 数据集文件的路径或存储信息
        self.label_file = label_file
        self.save_key = save_key

    def save(self):
        # 将数据集信息存储到 MongoDB 中的 datasets 集合
        try:
            mongo.db.datasets.insert_one({
                'user_id': ObjectId(self.user_id),
                'name': self.name,
                'description': self.description,
                'dataset_format_ids': [ObjectId(d) for d in self.dataset_format_ids],
                'detect_type_id': ObjectId(self.detect_type_id),
                'created_at': self.created_at,
                'image_files': self.image_files,
                'label_file': self.label_file,
                'valid_images_num': self.valid_images_num,
                'save_key': self.save_key,
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
            detect_type_id=dataset_data['detect_type_id'],
            label_file=dataset_data['label_file'],
            image_files=dataset_data['image_files'],
            valid_images_num=dataset_data['valid_images_num'],
            dataset_format_ids=dataset_data['dataset_format_ids'],
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
            traceback.print_exc()
            return False
        

    @staticmethod
    def find_by_user(user_id):
        # 根据用户 ID 查询该用户上传的所有数据集信息
        try:
            datasets = list(mongo.db.datasets.find({'user_id': ObjectId(user_id)}))
            for dataset in datasets:
                dataset['_id'] = str(dataset['_id'])
                dataset['user_id'] = str(dataset['user_id'])
                dataset['dataset_format_ids'] = [str(d) for d in dataset['dataset_format_ids']]
                dataset['detect_type_id'] = str(dataset['detect_type_id'])
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
                    'dataset_format_ids': self.dataset_format_ids,
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
        
        
class DetectType:
    def __init__(self, name, tag_name, description='', created_at=None):
        self._id = None
        self.name = name
        self.tag_name = tag_name
        self.description = description
        self.created_at = datetime.utcnow() if not created_at else created_at
        
    @staticmethod
    def from_dict(detect_type_data):
        detect_type = DetectType(
            name=detect_type_data['name'],
            tag_name=detect_type_data['tag_name'],
            description=detect_type_data['description'],
            created_at=detect_type_data['created_at'],
        )
        detect_type._id = detect_type_data['_id']
        return detect_type
    
    def save(self):
        try:
            mongo.db.detect_types.insert_one({
                'name': self.name,
                'tag_name': self.tag_name,
                'description': self.description,
                'created_at': self.created_at,
            })
            return True
        except Exception as e:
            print(e)
        return False

    @staticmethod
    def find_by_id(detect_type_id):
        data = mongo.db.detect_types.find_one({'_id': ObjectId(detect_type_id)})
        if data:
            return DetectType.from_dict(data)
        return None

    @staticmethod
    def list_all():
        detect_types = mongo.db.detect_types.find()
        return [DetectType.from_dict(dt) for dt in detect_types]

    def delete(self):
        if self.id:
            mongo.db.detect_types.delete_one({'_id': ObjectId(self.id)})

    def __repr__(self):
        repr_str = f"_id: {self._id}\n"
        repr_str += f"name: {self.name}\n"
        repr_str += f"tag_name: {self.tag_name}\n"
        repr_str += f"description: {self.description}\n"
        repr_str += f"created_at: {self.created_at}\n"
        return repr_str
    
# For training a model 具体到一个训练任务
class Algorithm:
    def __init__(self, name, training_framework_id, detect_type_id, description='', created_at=None):
        self._id = None
        self.name = name
        self.training_framework_id = training_framework_id
        self.detect_type_id = detect_type_id # DetectType _id
        self.description = description
        self.created_at = datetime.utcnow() if not created_at else created_at
    
    @staticmethod
    def from_dict(algorithm_data):
        algorithm = Algorithm(
            name=algorithm_data['name'],
            training_framework_id=algorithm_data['training_framework_id'],
            detect_type_id=algorithm_data['detect_type_id'],
            created_at=algorithm_data['created_at'],
            description=algorithm_data['description'],
        )
        algorithm._id = algorithm_data['_id']
        return algorithm

    def save(self):
        try:
            mongo.db.algorithms.insert_one({
                'name': self.name,
                'training_framework_id': self.training_framework_id,
                'detect_type_id': self.detect_type_id,
                'description': self.description,
                'created_at': self.created_at,
            })
            return True
        except Exception as e:
            print(e)
        return False

    @staticmethod
    def find_by_id(algorithm_id):
        return mongo.db.algorithms.find_one({'_id': ObjectId(algorithm_id)})
    
    def delete(self):
        mongo.db.algorithms.delete_one({'_id': self._id})
    
    @staticmethod
    def list_all():
        algorithms = mongo.db.algorithms.find()
        return [Algorithm.from_dict(algo) for algo in algorithms]
    
    def __repr__(self):
        repr_str = f"_id: {self._id}\n"
        repr_str += f"name: {self.name}\n"
        repr_str += f"training_framework_id: {self.training_framework_id}\n"
        repr_str += f"detect_type_id: {self.detect_type_id}\n"
        repr_str += f"description: {self.description}\n"
        repr_str += f"created_at: {self.created_at}\n"
        return repr_str

class DatasetFormat:
    def __init__(self, name, description='', created_at=None):
        self._id = None
        self.name = name
        self.description = description
        self.created_at = datetime.utcnow() if not created_at else created_at
    
    @staticmethod
    def from_dict(format_data):
        dataset_format = DatasetFormat(
            name=format_data['name'],
            created_at=format_data['created_at'],
            description=format_data['description'],
        )
        dataset_format._id = format_data['_id']
        return dataset_format

    def save(self):
        try:
            mongo.db.dataset_formats.insert_one({
                'name': self.name,
                'description': self.description,
                'created_at': self.created_at,
            })
            return True
        except Exception as e:
            print(e)
        return False

    @staticmethod
    def find_by_id(format_id):
        try:
            data = mongo.db.dataset_formats.find_one({'_id': ObjectId(format_id)})
            return DatasetFormat.from_dict(data)
        except:
            return None
    
    def delete(self):
        mongo.db.dataset_formats.delete_one({'_id': self._id})
    
    @staticmethod
    def list_all():
        dataset_formats = mongo.db.dataset_formats.find()
        print(dataset_formats[0]['_id'])
        return [DatasetFormat.from_dict(df) for df in dataset_formats]
    
    def __repr__(self):
        repr_str = f"_id: {self._id}\n"
        repr_str += f"name: {self.name}\n"
        repr_str += f"description: {self.description}\n"
        repr_str += f"created_at: {self.created_at}\n"
        return repr_str

# For creating configuration files
class TrainingFramework:
    def __init__(self, name, dataset_format_id, description='', created_at=None):
        self._id = None
        self.dataset_format_id = dataset_format_id
        self.name = name
        self.description = description
        self.created_at = datetime.utcnow() if not created_at else created_at
    
    @staticmethod
    def from_dict(framework_data):
        training_framework = TrainingFramework(
            name=framework_data['name'],
            dataset_format_id=framework_data['dataset_format_id'],
            created_at=framework_data['created_at'],
            description=framework_data['description'],
        )
        training_framework._id = training_framework['_id']
        return training_framework

    def save(self):
        try:
            mongo.db.training_frameworks.insert_one({
                'name': self.name,
                'dataset_format_id': self.dataset_format_id,
                'description': self.description,
                'created_at': self.created_at,
            })
            return True
        except Exception as e:
            print(e)
        return False

    @staticmethod
    def find_by_id(format_id):
        return mongo.db.training_frameworks.find_one({'_id': ObjectId(format_id)})
    
    def delete(self):
        mongo.db.training_frameworks.delete_one({'_id': self._id})
    
    @staticmethod
    def list_all():
        training_frameworks = mongo.db.training_frameworks.find()
        return [DatasetFormat.from_dict(tf) for tf in training_frameworks]
    
    def __repr__(self):
        repr_str = f"_id: {self._id}\n"
        repr_str += f"name: {self.name}\n"
        repr_str += f"dataset_format_id: {self.dataset_format_id}\n"
        repr_str += f"description: {self.description}\n"
        repr_str += f"created_at: {self.created_at}\n"
        return repr_str