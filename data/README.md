# Data

## 5-Fold
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

### annotations_train.csv

<img src="/images/annotations_df_head.png">

annotations 데이터에 대해서만 csv format으로 변경하였습니다.

`class label`은 모두 +1 씩 더했습니다. `faster_rcnn_torchvision_train.ipynb`을 참조하면 `class 0`은 background를 의미한다고 합니다.

`x_max` 와 `y_max`를 추가하였습니다.

