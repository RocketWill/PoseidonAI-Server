from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os
# import PointRend project
from detectron2.projects import point_rend
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader, DatasetMapper
import detectron2.utils.comm as comm
import torch
import time
import datetime
import numpy as np
import logging
import os
from detectron2.data.datasets import register_coco_instances
register_coco_instances("wafer1_train", {}, "w1_dataset/train.json", "w1_dataset")
register_coco_instances("wafer1_val", {}, "w1_dataset/val.json", "w1_dataset")
class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                logging.info(
                    "Loss on Validation  Done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    )
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return mean_loss
            
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            val_loss = self._do_loss_eval()
            self.trainer.storage.put_scalar('validation_loss', val_loss)
            print(f"Iteration {next_iter}: Validation Loss: {val_loss:.4f}")

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
                    
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks

    def after_step(self):
        super().after_step()
        
        # Print training loss
        if (self.iter + 1) % 20 == 0:  # adjust this number to control print frequency
            train_loss = self.storage.latest()['total_loss']
            
            if isinstance(train_loss, (tuple, list)):
                train_loss_str = ", ".join([f"{l:.4f}" for l in train_loss])
            else:
                train_loss_str = f"{train_loss:.4f}"
            
            print(f"Iteration {self.iter + 1}: Training Loss: {train_loss_str}")

cfg = get_cfg()
point_rend.add_pointrend_config(cfg)
cfg.merge_from_file("projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
cfg.DATASETS.TRAIN = ("wafer1_train",)
cfg.DATASETS.TEST = ("wafer1_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl" 
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 #nr classes + 1
cfg.MODEL.POINT_HEAD.NUM_CLASSES = 2
cfg.SOLVER.MAX_ITER=100
cfg.SOLVER.CHECKPOINT_PERIOD = 500
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = Trainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# After training, you can access the losses:
train_loss = trainer.storage.history('total_loss').values()
val_loss = trainer.storage.history('validation_loss').values()

print("Training loss:", train_loss)
print("Validation loss:", val_loss)