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