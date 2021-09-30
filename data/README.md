# Data

## Stratified 5-Fold

### csv format

<img src="/images/annotations_df_head.png">

csv format은 json 파일의 annotations 데이터만 포함하고 있습니다.

`bbox` list를 분해하여 다음을 추가하였습니다.

- `x`, `y` 좌표에 대하여 `min`, `max`, `center`를 추가하였습니다.
- `w`, `h` 를 따로 column을 추가하였습니다.

구성

- stratified_train.csv
- stratified_valid.csv

### json format
json format은 기존에 주어진 cocodataset 포맷과 동일합니다.

- stratified_train.json
- stratified_valid.json

### 
| category_id | train | test |
| :---------: | ----: | ---: |
|      0      |  3187 |  779 |
|      1      |  5102 | 1250 |
|      2      |   708 |  189 |
|      3      |   772 |  164 |
|      4      |   780 |  202 |
|      5      |  2293 |  650 |
|      6      |   965 |  298 |
|      7      |  4204 |  974 |
|      8      |   104 |   55 |
|      9      |   387 |   81 |

