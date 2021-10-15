# Swin Transformer

## Data

total: train 50%, test 50%

train: train 90%, validation 10%

test: public evaluation 25%, private evaluation 25%

### Stratified group k-fold
10 fold로 데이터를 나누었습니다.

category와 area를 고려하여 train/validation 세트를 구성했습니다.

- train

- val

## Model

|          | model          |
| -------- | -------------- |
| backbone | Swin-S         |
| neck     | FPN            |
| head     | HTC (soft-nms) |

### config
```bash
# based on: mmdetection
config
|-- datasets
|   `-- recycle_dataset_albu.py
|-- default_runtime.py
`-- models
    |-- htc
    |   `-- htc_soft-nms_without_mask_r50_fpn_1x_coco.py
    `-- swin
        `-- swin-s_img-768_AdamW-20e.py
```

### training

```
config/schedules/schedule_20e_AdamW.py  # config
```

- epoch: 30 epoch
- Optimizer: AdamW
- scheduler: manual

### augmentation
- Resize (multi-scale training)
  - img_scale `[(1024, 768), (768, 1024), (896, 1024), (1024, 1024)]`
- RandomFlip
- ShiftScaleRotate
- RandomBrightnessContrast
- ImageCompression
- Blur
- MedianBlur
- Normalize
