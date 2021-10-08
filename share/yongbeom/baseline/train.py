# built-in
import pdb
import time
import logging
from datetime import datetime

# 3rd

# torch
import torch

# from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader

import torch.optim as optim
# from torch.optim.lr_scheduler import StepLR
# from torch.optim import lr_scheduler

# Custom
from models.torchvision_faster_rcnn import TorchVisionFasterRCNNwFPN
from datasets import RecycleDetDataset, get_train_transform, get_valid_transform
from utils import GetConfig

now = datetime.now()
cur_time_str = now.strftime("%d%m%Y_%H%M")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)-8s %(levelname)-6s %(message)s',
    datefmt='%m-%d %H:%M',
    filename=f'FocalLoss_SGD1e-3_{cur_time_str}.wandbtest.log',
    filemode='w')

logger = logging.getLogger("train")


def collate_fn(batch):
    return tuple(zip(*batch))


def main():

    config_defaults = {
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 1e-4,
        # 'scheduler_step': 1,
        # 'optimizer': 'sgd',
        # 'optimizer': 'adam',
        # 'fc_layer_size': 128,
        # 'image_max_size': 224,
        # 'freeze_layers': [None],
        # 'init_layers': [4, 5],
    }
    config = GetConfig(config_defaults)

    NUM_CLASS = 10
    TRAIN_PROCESS = ['train', 'val']
    # TRAIN_PROCESS = ['val']
    TRAIN_DATA_PATH = "/opt/ml/Git/object-detection-level2-cv-10/data"
    IMAGE_DATA_PATH = "/opt/ml/detection/dataset"
    TRAIN_JSON = {
        # "train": f"{TRAIN_DATA_PATH}/stratified_train.10fold.json",
        # "val": f"{TRAIN_DATA_PATH}/stratified_valid.10fold.json",
        "train": f"{TRAIN_DATA_PATH}/stratified_train.10fold.wArea.json",
        "val": f"{TRAIN_DATA_PATH}/stratified_valid.10fold.wArea.json",
    }
    TRAIN_CSV = {
        "train": f"{TRAIN_DATA_PATH}/stratified_train.10fold.csv",
        "val": f"{TRAIN_DATA_PATH}/stratified_valid.10fold.csv",
    }

    defined_transforms = {
        'train': get_train_transform(),
        'val': get_valid_transform()
    }

    _time = time.perf_counter()
    recycle_dataset = {
        x: RecycleDetDataset(
            annotation=TRAIN_JSON[x],
            data_dir=IMAGE_DATA_PATH,
            transforms=defined_transforms[x],
        )
        for x in TRAIN_PROCESS
    }
    logger.info(f"Dataset progress. {time.perf_counter() - _time:.4f}s")

    _time = time.perf_counter()
    dataloaders = {
        x: DataLoader(
            recycle_dataset[x],
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn,
            # sampler=sampler[x],
        )
        for x in TRAIN_PROCESS
    }
    logger.info(f"Dataloader progress. {time.perf_counter() - _time:.4f}s")

    _time = time.perf_counter()
    model = TorchVisionFasterRCNNwFPN(num_classes=NUM_CLASS, pretrained=True)
    model.to(device)
    logger.info(f"Model progress. {time.perf_counter() - _time:.4f}s")

    _time = time.perf_counter()
    # params = (p for p in model.parameters() if p.requires_grad)
    logger.info(f"Define grad_params. {time.perf_counter() - _time:.4f}s")

    optimizer = optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=0.9,
    )
    #   weight_decay=0.0005)

    # training
    model.train_model(
        dataloaders=dataloaders,
        optimizer=optimizer,
        device=device,
        #   scheduler=scheduler,
        #   wandb_obj=wandb_obj,
        num_epochs=config.epochs,
        train_process=TRAIN_PROCESS,
    )


if __name__ == "__main__":

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    main()

    # import wandb

    # api = wandb.Api()

    # run is specified by <entity>/<project>/<run_id>
    # run = api.run("booduck4ai/effv2s_pretrained/")

    # sweep_config = {
    #     'method': 'grid',  #grid, random
    #     'metric': {
    #         'name': 'person_loss',
    #         # 'name': 'label_loss',
    #         'goal': 'minimize'
    #     },
    #     'parameters': {
    #         'epochs': {
    #             'values': [20]
    #         },
    #         'batch_size': {
    #             'values': [244]
    #         },
    #         # 'dropout': {
    #         #     'values': [0.3, 0.4, 0.5]
    #         # },
    #         'fc_layer_size': {
    #             'values': [128]
    #             # 'values': [512]
    #         },
    #         'learning_rate': {
    #             'values': [1e-2]
    #             # 'values': [1e-2]
    #         },
    #         'image_max_size': {
    #             'values': [224]
    #         },
    #         'scheduler_step': {
    #             'values': [2, 4, 6]
    #         },
    #         'freeze_layers': {
    #             'values': [
    #                 [None],
    #                 # ['0'],
    #                 # ['1', '0'],
    #                 # ['2', '1', '0'],
    #                 # ['3', '2', '1', '0'],
    #                 # ['0', '1', '2', '3', '4'],
    #             ]
    #         },
    #         'init_layers': {
    #             'values': [
    #                 # [None],
    #                 ['5'],
    #                 # ['4', '5'],
    #                 # ['3', '4', '5'],
    #             ]
    #         },
    #     }
    # }
    # sweep_id = wandb.sweep(sweep_config, project='test')

    # wandb.agent(sweep_id, main)
