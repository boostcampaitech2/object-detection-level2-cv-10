<h1 align="center">
<p>object-detection-level2-cv-10
</h1>

<h3 align="center">
<p>권예환_T2012, 권용범_T2013, 김경민_T2018, 방누리_T2098, 심재빈_T2124, 정현종_T2208
</h3>

## Overview
자원의 대부분을 수입하는 우리나라는 특히 자원 재활용이 중요합니다. 국내에서 버려지는 쓰레기종량제 봉투 속을 살펴보면 70%는 재활용이 가능한 자원입니다. 분리수거에 대한 접근성 향상을 위해 딥러닝 모델을 이용해 분리수거를 돕고자 합니다.

## Usage
해당 이미지에 포함된 객체들이 어떤 쓰레기인지 파악해주는 명령어입니다. 사용 방법은 아래와 같습니다.
```bash
./separate_trash -img image_sample1.jpg image_sample2.jpg
```
`--image, (-img)`: 이미지들을 입력 받습니다.

해당 명령어를 통해 본 팀에서 사용한 모델들을 앙상블 하여 이미지 결과를 도출합니다.

### 시연결과
<p float="left">
  <img src="/images/0000.jpg" width="250" />
  <img src="/images/0002.jpg" width="250" />
  <img src="/images/0008.jpg" width="250" />
</p>


## Model infomation

### [YOLOv5](/yolov5/)
```
optimizer: AdamW
scheduler: one_cycle
epoch: 200
loss: Yolov5 default loss
```

### [Swin-t FPN Cascade R-CNN](/mmdet_config/)
```
optimizer: AdamW (lr = 1 X 1e-4)
scheduler: StepLR (gamma = 0.1, 16 epoch, 22 epoch)
epoch: 24
loss
    classification : CrossEntropy
    bounding box regression : Smooth L1
```
### [Swin-t FPN Cascade R-CNN Pseudo labeling](/mmdet_config/)
```
Pseudo labeling (label data Learning)
model = yolov5x
optimizer: SGD (lr = 0.001)
scheduler: OnecycleLR
epoch: 44/50(early stop)
loss : Yolov5 default loss
```

```
Pseudo labeling (Semi-Supervised Learning)
model
    backbone : swin-t
    neck : FPN
    head : Cascade R-CNN
optimizer: AdamW (lr = 1e-4)
scheduler: StepLR (gamma = 0.1, 16 epoch, 22 epoch)
epoch: 23
loss
    classification : CrossEntropy
    bounding box regression : Smooth L1
```
### [Swin-s FPN HTC](/mmdet_config/)
```
model
    backbone : swin-s
    neck : FPN
    head : HTC
optimizer: AdamW (lr = 1e-4)
scheduler: StepLR (gamma = 0.1, 12 epoch, 22 epoch, 28 epoch)
epoch: 30
loss: HTC default loss
```

## Ensemble
```
python ensemble_inference.py ensemble_inference_cfg.json
```
