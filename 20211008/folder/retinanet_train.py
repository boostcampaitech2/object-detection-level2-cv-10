from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
import copy
cfg = Config.fromfile('./configs/custom/final.py')

cfg.seed = 2021
cfg.gpu_ids = [0]
cfg.work_dir = './work_dirs/retinanet_trash'
cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
#cfg.model.roi_head.bbox_head.num_classes = 10
datasets = [build_dataset(cfg.data.train)]
model = build_detector(cfg.model,
                       train_cfg=cfg.get('train_cfg'),
                       test_cfg=cfg.get('test_cfg'))
model.init_weights()
train_detector(model, datasets[0], cfg, distributed=False, validate=True)