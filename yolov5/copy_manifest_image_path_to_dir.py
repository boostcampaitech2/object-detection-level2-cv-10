import shutil
import os

#train / val 각각 해줘야 함

# dest_dir = '/opt/ml/detection/dataset/yolov5_dataset/images/train/'
# mainfest_file_dir = '/opt/ml/detection/yolov5/convert2Yolo/out.txt'

dest_dir = '/opt/ml/detection/dataset/yolov5_dataset/images/val/'
mainfest_file_dir = '/opt/ml/detection/yolov5/convert2Yolo/out.txt'

for line in open(mainfest_file_dir, "r"):
    line = line.strip()
    file_name = os.path.basename(line)
    shutil.copyfile(line, dest_dir + file_name)
    