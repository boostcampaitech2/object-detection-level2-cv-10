<h1 align="center">
<p>object-detection-level2-cv-10
</h1>

<h3 align="center">
<p>권예환_T2012, 권용범_T2013, 김경민_T2018, 방누리_T2098, 심재빈_T2124, 정현종_T2208
</h3>

## Overview

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

1. [YOLOv5](/yolov5/README.MD)

2. [swin-t_Cascade_R-CNN](/mmdet_config/README.md)

2. [swin-t_HTC_R-CNN](/mmdet_config/README.md)

## Ensemble
```
python ensemble_inference.py ~
```
