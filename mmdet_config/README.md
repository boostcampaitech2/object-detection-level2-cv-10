
```
mmdet_config
│  default_runtime.py
│
├─datasets
│      recycle_dataset.py
│      recycle_dataset_albu.py
│      recycle_dataset_cascade.py
│      recycle_dataset_pseudo_labeling.py
│
├─models
│  ├─cascade_rcnn
│  │      cascade_rcnn_r50_fpn.py
│  │
│  ├─htc
│  │      htc_soft-nms_without_mask_r50_fpn_1x_coco.py
│  │
│  └─swin
│          swin-s_fpn_htc_soft-nms_AdamW-2x.py
│          swin-t_img-768_AdamW-24e.py
│          swin-t_img-768_AdamW-24e_pseudo_labeling.py
│
└─schedules
        schedule_1x.py
        schedule_1x_AdamW.py
        schedule_20e.py
        schedule_20e_AdamW.py
        schedule_2x.py
        schedule_3x_AdamW.py
```
# mmdetection
  - mmdet/apis/train.py 142 line
    ```python
    #수정 전
        val_dataloader = build_dataloader(
        val_dataset,
        samples_per_gpu=val_samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)
    ```
    ```python
    #수정 후
        val_dataloader = build_dataloader(
        val_dataset,
        samples_per_gpu= 1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)
    ```
## cascade r-cnn

- pseudo labeling
  - 필요한 것
    - pseudo labeling.py를 통해 얻은 이미지 폴더
    - 만들어진 이미지와 matching되는 pseudo labeling된 json file
  - commend
    ```python
    python mmdet_train.py -c mmdet_config/models/swin/swin-t_img-768_AdamW-24e_pseudo_labeling.py
    ```
- normal
  - commend
    ```python
    python mmdet_train.py -c mmdet_config/models/swin/swin-t_img-768_AdamW-24e.py
    ```
### hrnet
- hrnet
  - commend
    ```python
    python mmdet_train.py -c mmdet_config/models/swin/swin-s_fpn_htc_soft-nms_AdamW-2x.py
    ```
