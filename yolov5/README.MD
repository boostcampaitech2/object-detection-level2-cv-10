# yolov5 사용법

## COCO data format -> yolov5 data format으로 변경
### Git clone https://github.com/ssaru/convert2Yolo
### 실행 방법
실행시 에러가 발생하는 부분은 파일 경로에 맞게 수정해야 합니다.
1. root folder인 convert2Yolo에 train folder 생성
2. class 종류를 정의하는 names.txt를 복사
3. command 실행하여 COCO data format json을 yolov5 label로 변환한 파일과 image path를 담고 있는 manifest 파일을 생성
각각 train/val 파일을 생성해야 합니다.
```python
python3 example.py --datasets COCO --img_path /opt/ml/detection/dataset/train --label /opt/ml/detection/dataset/train.json --convert_output_path ./ --img_type ".jpg" --manifest_path ./ --cls_list_file names.txt
```
4. manifest 파일을 이용하는 copy_manifest_image_path_to_dir.py를 복사
5. command를 실행하여 manifest파일 안에 있는 image path 경로에 해당하는 image 파일을 복사
각각 train/val 파일을 생성해야 합니다.
```python
python3 copy_manifest_image_path_to_dir.py
```
6. train 실행전에 images와 labels 폴더에 각각 test/train/val 폴더/파일이 존재하는지 확인
images(폴더)
  test(폴더-test image 파일들)
  train(폴더-train image 파일들)
  val(폴더-validation image 파일들)
labels(폴더)
  test(폴더-필요 없음)
  train(폴더-train label 파일들)
  val(폴더-validation label 파일들)



## yolov5 data를 이용한 train/val/test 방법
### Git clone https://github.com/open-mmlab/mmdetection.git
### 실행 방법
실행시 에러가 발생하는 부분은 파일 경로에 맞게 수정해야 합니다.
1. path와 class 정보를 갖고 있는 trash_10.yaml 파일을 복사
2. augmentation 정보를 갖고 있는 hyp.finetune_custom.yaml 파일을 복사
3. default arguments를 수정한 train.py 파일을 복사
4. inference를 위한 detect.py 파일을 복사
5. COCO data format에 맞게 box(xmin, ymin, xmax, ymax)좌표를 변환해주고 제출 양식에 맞춰주는 submission_converter.py 파일을 복사
6. command를 실행하여 train
```python
python train.py --img 1024 --batch 16 --epoch 50 --data ./data/trash_10.yaml
```
7. command를 실행하여 inference
```python
python detect.py
```
8. command를 실행하여 inference 결과를 submission 형식으로 변환
```python
submission_converter.py
```


