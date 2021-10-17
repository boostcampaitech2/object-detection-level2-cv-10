# utils
## Ensemble
### ensemble_config1.json
- public LB : 0.627
- 사용 모델
  - yolov5x6
  - Hybrid task cascade(backbone = Swin-S) 
  - cascade R-CNN (backbone = Swin-T)
  - Semi-Supervised Learning
    - Supervised Learning(yolov5x)
    - Semi-Supervised Learning(cascade R-CNN, backbone = Swin-T)
- ensemble
  - non-maximum weighted(nmw)
### ensemble_config2.json
- public LB : 0.625
- 사용 모델
  - yolov5x6
  - Hybrid task cascade(backbone = Swin-S) 
  - cascade R-CNN (backbone = Swin-T)
- ensemble
  - non-maximum weighted(nmw)
### ensemble_inference.py

```bash
python ensemble_inference.py ensemble_config1.json
```

## Pseudo labeling
**yolov5에서 "pseudo labeling을 위한 train/val/test 방법"으로 crop된 이미지를 사용합니다.**
1. pseudo_labeling.py의 line 145를 crop된 이미지들이 모인 폴더로 지정합니다.
2. '/opt/ml/detection/dataset/'에 train_plus폴더를 생성합니다.
```python
for idx in categories.keys():
        crop_image_path = './yolov5/runs/detect/exp10/crops/'
        path = crop_image_path + idx + '/'
        file_list = os.listdir(path)
```
3. pseudo_labeling.py를 실행합니다.
4. 생성된 json 파일과 image를 확인합니다.