# MMDetection

## Config
```
mmdet_config
│  └─default_runtime.py
├─datasets
│  ├─recycle_dataset.py
│  ├─recycle_dataset_albu.py
│  ├─recycle_dataset_cascade.py
│  └─recycle_dataset_pseudo_labeling.py
├─models
│  ├─cascade_rcnn
│  │  └─cascade_rcnn_r50_fpn.py
│  ├─htc
│  │  └─htc_soft-nms_without_mask_r50_fpn_1x_coco.py
│  └─swin
│     ├─swin-s_fpn_htc_soft-nms_AdamW-2x.py
│     ├─swin-t_fpn_cascade_rcnn_AdamW-24e.py
│     └─swin-t_fpn_cascade_rcnn_pseudo_labeling.py
└─schedules
   ├─schedule_1x.py
   ├─schedule_1x_AdamW.py
   ├─schedule_20e.py
   ├─schedule_20e_AdamW.py
   ├─schedule_2x.py
   └─schedule_3x_AdamW.py
```

### TTA를 위한 코드 수정
일부 Aug가 `samples_per_gpu`가 1인 경우에만 작동을 하여 해당 코드를 수정합니다.
- mmdet/apis/train.py#L144
  ```diff
    val_dataloader = build_dataloader(
    val_dataset,
  - samples_per_gpu=val_samples_per_gpu,
  + samples_per_gpu= 1,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=distributed,
    shuffle=False)
  ```

## Model
MMDetection에서는 3개 모델을 이용하여 학습을 진행하였습니다.

1. swin-t_img-768_AdamW-24e.py

2. swin-t_img-768_AdamW-24e_pseudo_labeling.py

3. swin-s_fpn_htc_soft-nms_AdamW-2x.py

## Backbone - Swin Transformer
backbone 모델로 모두 swin transformer를 기반으로 하여 학습을 진행하였습니다.

## Neck and Head
### Cascade R-CNN

1. pseudo labeling

    필요한 것
    - pseudo_labeling.py를 통해 얻은 이미지 폴더
    - 만들어진 이미지와 matching되는 pseudo labeling된 json file
    실행 방법
    ```bash
    python mmdet_train.py -c mmdet_config/models/swin/swin-t_img-768_AdamW-24e_pseudo_labeling.py
    ```
2. normal

    실행 방법
    ```bash
    python mmdet_train.py -c mmdet_config/models/swin/swin-t_img-768_AdamW-24e.py
    ```
### HTC
    ```bash
    python mmdet_train.py -c mmdet_config/models/swin/swin-s_fpn_htc_soft-nms_AdamW-2x.py
    ```
