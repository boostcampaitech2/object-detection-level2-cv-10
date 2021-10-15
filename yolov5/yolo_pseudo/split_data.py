import cv2
import json
def get_file_path(json_file, image_num):
    file_path = json_file['images'][image_num]['file_name']
    return file_path

def read_image(dataset_path, file_path):
    image = cv2.imread(dataset_path + file_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

n = [0 for i in range(4884)]
with open('/opt/ml/detection/dataset/stratified_train.json', 'r') as f:
    an = json.load(f)
for i in range(len(an['images'])):   
    temp = an['images'][i]['file_name'].split('/')
    n[int(temp[1][:4])] = 1

dataset_path = '/opt/ml/detection/dataset/'
with open('/opt/ml/detection/dataset/train.json', 'r') as f:
    an = json.load(f)
for i in range(len(an['images'])):   
    temp = an['images'][i]['file_name'].split('/')
    print(dataset_path + an['images'][i]['file_name'])
    image = cv2.imread(dataset_path + an['images'][i]['file_name'], cv2.IMREAD_COLOR)
    if n[int(temp[1][:4])] == 0:
        cv2.imwrite(f'/opt/ml/detection/dataset/yolo/valid/{temp[1]}', image)
    else:
        cv2.imwrite(f'/opt/ml/detection/dataset/yolo/train/{temp[1]}', image)