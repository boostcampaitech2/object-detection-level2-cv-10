import json
import random
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import json
from collections import OrderedDict
import os
import copy

def get_file_path(json_file, image_num):
    file_path = json_file['images'][image_num]['file_name']
    return file_path


def get_annotations(json_file, image_num):
    anns = [ann for ann in json_file['annotations'] if ann['image_id'] == image_num]
    return anns


def read_image(dataset_path, file_path):
    image = cv2.imread(dataset_path + file_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def plot_bbox(image, anns):
    for ann in anns:
        xmin, ymin, w, h = map(int, ann['bbox'])
        image = cv2.rectangle(image, (xmin, ymin), (xmin + w, ymin + h), (0, 0, 255), 3)
    plt.imshow(image)

def make_imagedf(name, data):
    data['images'].append({})
    data['images'][-1]['width'] = 1024
    data['images'][-1]['height'] = 1024
    data['images'][-1]['file_name'] = 'train_plus/' + str(name) + '.jpg'
    data['images'][-1]['license'] = 0
    data['images'][-1]['flickr_url'] = None
    data['images'][-1]['coco_url'] = None
    data['images'][-1]['data_captured'] = None
    data['images'][-1]['id'] = name
    return data
    
def make_1annodf(name,
                 anno_list,
                 category_id, 
                 i, 
                 j,
                 id):
    data['annotations'].append({})
    data['annotations'][-1]['image_id'] = name
    data['annotations'][-1]['category_id'] = category_id
    data['annotations'][-1]['area'] = i * j
    data['annotations'][-1]['bbox'] = [512-int(j/2), 512-int(i/2), j, i]
    data['annotations'][-1]['iscrowd'] = 0
    data['annotations'][-1]['id'] = id
    return anno_list

def make_4annodf(name,
                 anno_list,
                 category_id,
                 i,
                 j,
                 idx,
                 id):
    ix = [0, 1, 1, 0]
    jx = [0, 1, 0, 1]
    data['annotations'].append({})
    data['annotations'][-1]['image_id'] = name
    data['annotations'][-1]['category_id'] = category_id
    data['annotations'][-1]['area'] = i * j
    data['annotations'][-1]['bbox'] = [512 * jx[idx] + (256-int(j/2)), 512 * ix[idx] + (256-int(i/2)), j, i]
    data['annotations'][-1]['iscrowd'] = 0
    data['annotations'][-1]['id'] = id
    return anno_list

def make_1object(img_inform, data, cnt):
    save_img = np.full((1024,1024,3), 255)
    img, i, j, current_categories = img_inform
    cnt_image, cnt_anno = cnt
    for i2 in range(i):
        for j2 in range(j):
            save_img[512-int(i/2) + i2][512-int(j/2) + j2] = img[i2][j2]
    #plt.imshow(save_img)
    cv2.imwrite(f'../dataset/train_plus/{cnt_image}.jpg', save_img)
    data = make_imagedf(cnt_image, data)
    data = make_1annodf(cnt_image,
                        data,
                        current_categories, 
                        i, 
                        j, 
                        cnt_anno)
    return data

def make_4object(img_q, data, cnt):
    ix = [0, 1, 1, 0]
    jx = [0, 1, 0, 1]
    save_img = np.full((1024,1024,3), 255)
    cnt_image, cnt_anno = cnt
    for idx in range(len(img_q)):
        img, i, j, current_categories = img_q[idx]
        for i2 in range(i):
            for j2 in range(j):
                save_img[512 * ix[idx] + (256-int(i/2) + i2)][512 * jx[idx] + (256-int(j/2) + j2)] = img[i2][j2]
        #plt.imshow(save_img)
        cv2.imwrite(f'../dataset/train_plus/{cnt_image}.jpg', save_img)
        data = make_4annodf(cnt_image,
                        data,
                        current_categories, 
                        i, 
                        j,
                        idx, 
                        cnt_anno + idx)
    data = make_imagedf(cnt_image, data)
    return data

if __name__ == '__main__':
    with open('/opt/ml/detection/dataset/stratified_train.json', 'r') as f:
        json_data = json.load(f)
    data = copy.deepcopy(json_data)
    print(len(data['images']))
    for i in range(len(data['images'])):
        word = data['images'][i]['file_name'].split('/')
    # path ../dataset/train_plus/{4882 + cnt}.jpg
    categories = {"General trash" : 0,
        "Paper" : 1 ,
        "Paper pack" : 2,
        "Metal" : 3,
        "Glass" : 4, 
        "Plastic" : 5,
        "Styrofoam" : 6,
        "Plastic bag" : 7,
        "Battery" : 8,
        "Clothing" : 9}
    #plt.figure(figsize=(15,15))
    q_1object = []
    q_4object = []
    images_list = OrderedDict()
    annotations_list = OrderedDict()
    cnt = 0
    for idx in categories.keys():
        path = './yolov5/runs/detect/exp10/crops/' + idx + '/'
        file_list = os.listdir(path)
        print(len(file_list))
        for title in file_list:
            cnt += 1
            img = read_image(path, title)
            i, j = img.shape[0], img.shape[1]
            if i >= 512 or j >= 512:
                q_1object.append([img, i, j, categories[idx]])
                continue
            else:
                q_4object.append([img, i, j, categories[idx]])

    complete_make = 0
    cnt_local_image = 4883
    cnt_local_anno = 23144
    cnt_image = cnt_local_image
    cnt_anno = cnt_local_anno
    while len(q_4object) > complete_make:
        q = copy.deepcopy(q_4object[complete_make : complete_make + 4])
        data = make_4object(q, data, [cnt_image, cnt_anno])
        complete_make += 4
        cnt_image += 1
        cnt_anno += len(q)
        if complete_make % 100 == 0:
            print(f'4object {complete_make}')
            
    for current_image_inform in q_1object:
        data = make_1object(current_image_inform, data, [cnt_image, cnt_anno])
        cnt_image += 1
        cnt_anno += 1
        if cnt_image % 100 == 0:
            print(cnt_image)

    with open('../dataset/labeling_data.json', 'w', encoding = 'utf-8') as make_file:
        json.dump(data, make_file, ensure_ascii = False, indent = "\t")