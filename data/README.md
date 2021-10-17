# Data
## Stratified group 10-Fold
Kaggle의 [Stratified Group k-Fold Cross-Validation](https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation) 노트북 코드를 기반하여 train/val dataset을 나누었습니다.

해당 방식은 이미지를 하나의 그룹으로 보고 이미지를 여러 fold에 넣어보고 fold의 category에 대한 표준편차가 가장 낮은 fold에 해당 이미지를 추가하는 방식으로 진행됩니다.

- annotations.csv: 전체 annotation들에 대한 정보를 csv format으로 저장되어 있습니다.

### Consider Categories and Areas.
이미지에 포함된 객체들의 area와 category를 고려하여 data를 분리하였습니다.

`json` format의 경우 COCO dataset format을 따르고 있습니다.

`csv` format의 경우 box들의 position information이 추가되어 있습니다.

```
Train
  stratified_train.10fold.wArea.csv
  stratified_train.10fold.wArea.json

Validation
  stratified_valid.10fold.wArea.csv
  stratified_valid.10fold.wArea.json
```