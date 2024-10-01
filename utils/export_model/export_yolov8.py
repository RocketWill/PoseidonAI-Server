'''
Author: Will Cheng (will.cheng@efctw.com)
Date: 2024-09-11 15:51:27
LastEditors: Will Cheng (will.cheng@efctw.com)
LastEditTime: 2024-09-18 17:19:20
FilePath: /PoseidonAI-Server/utils/export_model/export_yolov8.py
'''
import os
import shutil
from datetime import datetime
import logging

from ultralytics import YOLO
from celery import Task, task

from .common import ExporterError, ExporterStatus, zip_model_and_deps
from utils.common import read_yaml

logger = logging.getLogger(__name__)

def generate_datetime():
    return datetime.now().strftime("%Y%m%d%H%M")

class ExportYOLOV8Model:

    def __init__(self, project_root) -> None:
        self.training_dir = os.path.join(project_root, 'project', 'exp')
        self.weights_dir = os.path.join(self.training_dir, 'weights')
        self.weights_file = os.path.join(self.weights_dir, 'best.pt')
        self.output_file = os.path.join(self.weights_dir, 'model_{}.torchscript'.format(generate_datetime()))
        self.cfg_file = os.path.join(project_root, 'cfg.yaml')
        self.cfg = read_yaml(self.cfg_file)

        self.status = ExporterStatus.IDLE
        self.error_detail = None

    def convert(self):
        try:
            # 状态更新为正在处理
            self.status = ExporterStatus.PROCESSING

            # 检查权重文件是否存在
            if not os.path.exists(self.weights_file):
                self.status = ExporterStatus.FAILURE
                self.error_detail = (ExporterError.FILE_NOT_FOUND, "Weights file not found.")
                logger.error(f"Error: {self.error_detail[1]}")
                return False

            model = YOLO(self.weights_file)

            # 导出模型为torchscript格式
            export_path = model.export(format="torchscript", simplify=True, 
                                       dynamic=False, imgsz=int(self.cfg['imgsz']))

            # 检查导出结果并移动文件
            if isinstance(export_path, str) and export_path:
                shutil.move(export_path, self.output_file)
                self.status = ExporterStatus.SUCCESS
                print(f"Model successfully converted to TorchScript format and saved as {self.output_file}")
                return self.output_file
            else:
                self.status = ExporterStatus.FAILURE
                self.error_detail = (ExporterError.UNEXPECTED_ERROR, "Export failed: No output file generated.")
                logger.error(f"Error: {self.error_detail[1]}")
                return False

        except AssertionError as e:
            self.status = ExporterStatus.ERROR
            self.error_detail = (ExporterError.ASSERTION_FAILED, str(e))
            logger.error(f"Assertion failed: {str(e)}")
            return False
        except FileNotFoundError as e:
            self.status = ExporterStatus.ERROR
            self.error_detail = (ExporterError.FILE_NOT_FOUND, str(e))
            logger.error(f"File not found: {str(e)}")
            return False
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                self.status = ExporterStatus.ERROR
                self.error_detail = (ExporterError.CUDA_OUT_OF_MEMORY, str(e))
                logger.error(f"CUDA out of memory: {str(e)}")
            else:
                self.status = ExporterStatus.ERROR
                self.error_detail = (ExporterError.RUNTIME_ERROR, str(e))
                logger.error(f"Runtime error: {str(e)}")
            return False
        except Exception as e:
            self.status = ExporterStatus.ERROR
            self.error_detail = (ExporterError.UNEXPECTED_ERROR, str(e))
            logger.error(f"Unexpected error: {str(e)}")
            return False


class ExportYolov8ModelTask(Task):
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f'Task {task_id} failed: {exc}')
        return super().on_failure(exc, task_id, args, kwargs, einfo)
    

@task(bind=True, base=ExportYolov8ModelTask, name='tasks.export.yolov8')
def start_export_model(self, project_root, output_zip, format, content):
    from .constant import TORCHSCRIPT_DEPS_DIR, YOLOV8_DETNET_TS_DLL

    self.status = ExporterStatus.PENDING
    self.update_state(state=ExporterStatus.PENDING.name, meta={'status': self.status.name, 'exc_type': '', 'exc_message': ''})
    
    exporter = ExportYOLOV8Model(project_root)
    
    try:
        success = exporter.convert()
        if success:
            converted_model_file = success
            output_file = zip_model_and_deps(output_zip, content, converted_model_file, TORCHSCRIPT_DEPS_DIR, YOLOV8_DETNET_TS_DLL) # may be False
            self.status = ExporterStatus.SUCCESS
            self.update_state(state=ExporterStatus.SUCCESS.name, meta={'status': self.status.name, 'output_file': output_file, 'exc_type': '', 'exc_message': ''})
            logger.info(f"Task status: {self.status.name}. Model successfully exported to {exporter.output_file}")
            logger.info(f"Task status: {self.status.name}. Model & deps exported to {output_file}")
            return {'status': self.status.name, 'output_file': output_file}
        else:
            self.status = ExporterStatus.FAILURE
            error_detail_str = f"{exporter.error_detail[0].name}: {exporter.error_detail[1]}"
            self.update_state(state=ExporterStatus.FAILURE.name, meta={'status': self.status.name, 'error_detail': error_detail_str, 'exc_type': exporter.error_detail[0].name, 'exc_message': exporter.error_detail[1]})
            logger.error(f"Task status: {self.status.name}. Export failed with error: {error_detail_str}")
            return {'status': self.status.name, 'error_detail': error_detail_str}
    
    except Exception as e:
        self.status = ExporterStatus.ERROR
        self.update_state(state=ExporterStatus.ERROR.name, meta={'status': self.status.name, 'error_detail': str(e), 'exc_type': 'UnexpectedError', 'exc_message': str(e)})
        logger.error(f"Task status: {self.status.name}. Unexpected error: {str(e)}")
        return {'status': self.status.name, 'error_detail': str(e)}
    
    finally:
        if self.status == ExporterStatus.PROCESSING:
            self.status = ExporterStatus.REVOKED
            self.update_state(state=ExporterStatus.REVOKED.name, meta={'status': self.status.name, 'exc_type': '', 'exc_message': ''})
            logger.warning(f"Task status: {self.status.name}. Task was revoked.")
