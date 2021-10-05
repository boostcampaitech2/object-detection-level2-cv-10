# built-in
import os
import logging
import time
from functools import partial

# 3rd
from tqdm import tqdm

# torch
import torch
import torch.nn as nn

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Custom
# from utils import MetricDict
from references.utils import MetricLogger, SmoothedValue
from references.coco_eval import CocoEvaluator
# from references.coco_utils import get_coco_api_from_dataset

logger = logging.getLogger('model')


class TorchVisionFasterRCNNwFPN(nn.Module):
    def __init__(self, num_classes=10):
        super(TorchVisionFasterRCNNwFPN, self).__init__()
        _time = time.perf_counter()
        self.backbone = None
        logger.info(f"Load backbone. {time.perf_counter() - _time:.4f}s")

        _time = time.perf_counter()
        self.head = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True)
        # self.head._has_warned = True
        logger.info(f"Load head. {time.perf_counter() - _time:.4f}s")

        _time = time.perf_counter()
        self.num_classes = num_classes + 1  # class 개수= 10 + background
        # get number of input features for the classifier
        in_features = self.head.roi_heads.box_predictor.cls_score.in_features
        self.head.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, self.num_classes)
        logger.info(
            f"Load roi_head predictor. {time.perf_counter() - _time:.4f}s")

        # Overwrite
        """
        ### Original function
        @torch.jit.unused
        def eager_outputs(self, losses, detections):
            # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
            if self.training:
                return losses

            return detections
        """
        def eager_outputs(self, losses, detections):
            return losses, detections

        self.head.eager_outputs = partial(eager_outputs, self.head)

    def get_loss_and_stat(self, images, targets, phase):
        loss_dict, detections = self.head(images, targets)

        if phase == "train":
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            losses.backward()
        elif phase == "val":
            loss_value = None
        else:
            raise PermissionError

        return loss_value, detections

    def train_model(self,
                    dataloaders,
                    optimizer,
                    device,
                    wandb_obj=None,
                    scheduler=None,
                    num_epochs=5,
                    epoch_div=5,
                    train_process=['train', 'val']):

        best_loss = 1000
        # metric_dict = MetricDict(train_process=train_process)
        cpu_device = torch.device("cpu")
        metric_train_logger = MetricLogger(delimiter="  ")
        # metric_train_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_val_logger = MetricLogger(delimiter="  ")

        coco = dataloaders['val'].dataset.coco
        # iou_types = _get_iou_types(model)
        iou_types = ['bbox']
        coco_evaluator = CocoEvaluator(coco, iou_types)

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1:>2}/{num_epochs} ----------")
            coco_evaluator.eval_imgs = {k: [] for k in iou_types}
            # print(f"Epoch {epoch + 1:>2}/{num_epochs} ----------")
            # metric_dict.reset()
            for phase in train_process:

                if phase == 'train':
                    self.train()  # Set model to training mode
                else:
                    self.eval()  # Set model to evaluate mode

                for images, targets, image_ids in tqdm(dataloaders[phase]):

                    # gpu 계산을 위해 image.to(device)
                    images = list(image.float().to(device) for image in images)
                    targets = [{k: v.to(device)
                                for k, v in t.items()} for t in targets]

                    # calculate loss
                    model_time = time.time()
                    with torch.set_grad_enabled(phase == 'train'):
                        optimizer.zero_grad()
                        losses, detections = self.get_loss_and_stat(
                            images, targets, phase)
                        optimizer.step()
                    model_time = time.time() - model_time

                    if phase == 'train':
                        # reduce losses over all GPUs for logging purposes
                        metric_train_logger.update(loss=losses)
                        metric_train_logger.update(
                            lr=optimizer.param_groups[0]["lr"])
                    elif phase == 'val':
                        outputs = detections
                        outputs = [{k: v.to(cpu_device)
                                    for k, v in t.items()} for t in outputs]

                        res = {
                            target["image_id"].item(): output
                            for target, output in zip(targets, outputs)
                        }
                        evaluator_time = time.time()
                        coco_evaluator.update(res)
                        evaluator_time = time.time() - evaluator_time
                        metric_val_logger.update(model_time=model_time,
                                                 evaluator_time=evaluator_time)

                # print(f"Epoch #{epoch+1} {phase} loss: {loss_hist.value}")
                # gather the stats from all processes
                if phase == 'val':
                    metric_val_logger.synchronize_between_processes()
                    print("Averaged stats:", metric_val_logger)
                    coco_evaluator.synchronize_between_processes()

                    # accumulate predictions from all images
                    coco_evaluator.accumulate()
                    coco_evaluator.summarize()

                # if loss_hist.value < best_loss and phase == 'val':
                #     save_path = './checkpoints/faster_rcnn_torchvision_checkpoints.pth'
                #     save_dir = os.path.dirname(save_path)
                #     if not os.path.exists(save_dir):
                #         os.makedirs(save_dir)

                #     torch.save(self.head.state_dict(), save_path)
                #     best_loss = loss_hist.value
