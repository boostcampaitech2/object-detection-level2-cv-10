import os

import numpy as np
import pandas as pd

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def get_train_transform():
    return A.Compose(
        [A.Resize(1024, 1024),
         A.Flip(p=0.5), ToTensorV2(p=1.0)],
        bbox_params={
            'format': 'pascal_voc',
            'label_fields': ['labels']
        })


def get_valid_transform():
    return A.Compose([ToTensorV2(p=1.0)],
                     bbox_params={
                         'format': 'pascal_voc',
                         'label_fields': ['labels']
                     })


class RecycleDetDataset(Dataset):
    '''
      data_dir: data가 존재하는 폴더 경로
      transforms: data transform (resize, crop, Totensor, etc,,,)
    '''
    def __init__(self, annotation, data_dir, transforms=None, n_data=None):
        super().__init__()
        self.data_dir = data_dir
        # coco annotation 불러오기 (coco API)
        self.coco = COCO(annotation)
        self.predictions = {
            "images": self.coco.dataset["images"].copy(),
            "categories": self.coco.dataset["categories"].copy(),
            "annotations": None
        }
        self.transforms = transforms
        self.index = self.coco.getImgIds()
        self.index = self.index[:n_data]

    def __getitem__(self, index: int):

        image_id = self.index[index]
        # image_id = [image_id]

        image_info = self.coco.loadImgs(image_id)[0]

        image = cv2.imread(os.path.join(self.data_dir,
                                        image_info['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        ann_ids = self.coco.getAnnIds(imgIds=image_info['id'])
        anns = self.coco.loadAnns(ann_ids)

        boxes = np.array([x['bbox'] for x in anns])

        # boxex (x_min, y_min, x_max, y_max)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        # torchvision faster_rcnn은 label=0을 background로 취급
        # class_id를 1~10으로 수정
        labels = np.array([x['category_id'] + 1 for x in anns])
        labels = torch.as_tensor(labels, dtype=torch.int64)

        areas = np.array([x['area'] for x in anns])
        areas = torch.as_tensor(areas, dtype=torch.float32)

        is_crowds = np.array([x['iscrowd'] for x in anns])
        is_crowds = torch.as_tensor(is_crowds, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            # 'image_id': torch.tensor([index]),
            'image_id': torch.tensor([image_id]),
            'area': areas,
            'iscrowd': is_crowds
        }

        # transform
        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            target['boxes'] = torch.tensor(sample['bboxes'],
                                           dtype=torch.float32)

        return image, target, image_id

    def __len__(self) -> int:
        # return len(self.coco.getImgIds())
        return len(self.index)