'''
Author: Will Cheng (will.cheng@efctw.com)
Date: 2024-10-04 13:42:14
LastEditors: Will Cheng (will.cheng@efctw.com)
LastEditTime: 2024-10-04 14:36:45
FilePath: /PoseidonAI-Server/utils/export_model/export_detectron2.py
'''
import os
import shutil
import glob
from datetime import datetime
import logging

from celery import Task, task
from typing import Dict, List, Tuple
import torch
from torch import Tensor, nn

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, detection_utils
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from detectron2.export import (
    STABLE_ONNX_OPSET_VERSION,
    TracingAdapter,
    dump_torchscript_IR,
    scripting_with_instances,
)
from detectron2.modeling import GeneralizedRCNN, RetinaNet, build_model
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.projects.point_rend import add_pointrend_config
from detectron2.structures import Boxes
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

from .common import ExporterError, ExporterStatus, zip_model_and_deps
from utils.common import read_yaml

logger = logging.getLogger(__name__)

def generate_datetime():
    return datetime.now().strftime("%Y%m%d%H%M")


def setup_cfg(model_file, cfg_file):
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.WEIGHTS = model_file
    cfg.MODEL.DEVICE = 'cpu'
    cfg.freeze()
    return cfg

def export_scripting(torch_model, output_file):
    assert TORCH_VERSION >= (1, 8)
    fields = {
        "proposal_boxes": Boxes,
        "objectness_logits": Tensor,
        "pred_boxes": Boxes,
        "scores": Tensor,
        "pred_classes": Tensor,
        "pred_masks": Tensor,
        "pred_keypoints": torch.Tensor,
        "pred_keypoint_heatmaps": torch.Tensor,
    }

    class ScriptableAdapterBase(nn.Module):
        # Use this adapter to workaround https://github.com/pytorch/pytorch/issues/46944
        # by not retuning instances but dicts. Otherwise the exported model is not deployable
        def __init__(self):
            super().__init__()
            self.model = torch_model
            self.eval()

    if isinstance(torch_model, GeneralizedRCNN):

        class ScriptableAdapter(ScriptableAdapterBase):
            def forward(self, inputs: Tuple[Dict[str, torch.Tensor]]) -> List[Dict[str, Tensor]]:
                instances = self.model.inference(inputs, do_postprocess=False)
                return [i.get_fields() for i in instances]

    else:

        class ScriptableAdapter(ScriptableAdapterBase):
            def forward(self, inputs: Tuple[Dict[str, torch.Tensor]]) -> List[Dict[str, Tensor]]:
                instances = self.model(inputs)
                return [i.get_fields() for i in instances]

    ts_model = scripting_with_instances(ScriptableAdapter(), fields)
    with PathManager.open(output_file, "wb") as f:
        torch.jit.save(ts_model, f)
    dump_torchscript_IR(ts_model, os.path.dirname(output_file))
    # TODO inference in Python now missing postprocessing glue code
    return None

class ExportDetectron2TSModel:

    def __init__(self, project_root) -> None:
        self.training_dir = os.path.join(project_root, 'project')
        self.cfg_file = os.path.join(self.training_dir, 'cfg.yaml')
        self.weights_files = sorted(glob.glob(os.path.join(self.training_dir, '*.pth')))
        self.weights_file = self.weights_files[-1]
        self.output_file = os.path.join(self.training_dir, 'model_{}.torchscript'.format(generate_datetime()))
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

            torch._C._jit_set_bailout_depth(1)
            cfg = setup_cfg(self.weights_file, self.cfg_file)
            torch_model = build_model(cfg)
            DetectionCheckpointer(torch_model).resume_or_load(cfg.MODEL.WEIGHTS)
            torch_model.eval()
            export_scripting(torch_model, self.output_file)

            # 检查导出结果并移动文件
            if isinstance(self.output_file, str) and self.output_file:
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
        
class ExportDetectron2TSModelTask(Task):
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f'Task {task_id} failed: {exc}')
        return super().on_failure(exc, task_id, args, kwargs, einfo)
    

@task(bind=True, base=ExportDetectron2TSModelTask, name='tasks.export.detectron2.ts')
def start_export_model(self, project_root, output_zip, format, content):
    from .constant import TORCHSCRIPT_DEPS_DIR, DETECTRON2_SEGNET_TS_DLL

    self.status = ExporterStatus.PENDING
    self.update_state(state=ExporterStatus.PENDING.name, meta={'status': self.status.name, 'exc_type': '', 'exc_message': ''})
    
    exporter = ExportDetectron2TSModel(project_root)
    
    try:
        success = exporter.convert()
        if success:
            converted_model_file = success
            output_file = zip_model_and_deps(output_zip, content, converted_model_file, TORCHSCRIPT_DEPS_DIR, DETECTRON2_SEGNET_TS_DLL) # may be False
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
