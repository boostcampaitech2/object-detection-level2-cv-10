# reference 
# https://towardsdatascience.com/how-to-analyze-the-coco-dataset-for-pose-estimation-7296e2ffb12e
# https://gist.github.com/michalfaber/8093a8383d8b3fb71cc876d830039b76#file-code2-py

# pip install pycocotools
from posixpath import ismount
from pycocotools.coco import COCO
import pandas as pd
import os

class COCO_ImageAnnotation:
    __save_path = None
    __images_df = pd.DataFrame()
    __annotations_df = pd.DataFrame()
    __train_coco_df = pd.DataFrame()

    def __init__(self, save_path=None) -> None:
        self.__save_path = save_path

    def convert_to_df(self, coco) -> None:
        images_data = []
        annotations_data = []
        # iterate over all images
        for img_id, img_fname, w, h, annotations in self.__get_meta(coco):
            images_data.append({
                'image_id': int(img_id),
                'file_name': img_fname,
                'width': int(w),
                'height': int(h)
            })
            # iterate over all annotations
            for annotation in annotations:
                annotations_data.append({
                    'image_id': annotation['image_id'],
                    'id': annotation['id'],
                    'category_id': annotation['category_id'],
                    'is_crowd': annotation['iscrowd'],
                    'bbox': annotation['bbox'],
                    'area': annotation['area'],                
                })

        self.__images_df = pd.DataFrame(images_data)
        self.__images_df.set_index('image_id', inplace=True)
        self.__annotations_df = pd.DataFrame(annotations_data)
        self.__annotations_df.set_index('image_id', inplace=True)

        return self.__images_df, self.__annotations_df


    # function iterates ofver all ocurrences of a  person and returns relevant data row by row
    def __get_meta(self, coco:COCO) -> list:
        ids = list(coco.imgs.keys())
        for i, img_id in enumerate(ids):
            img_meta = coco.imgs[img_id]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            # basic parameters of an image
            img_file_name = img_meta['file_name']
            w = img_meta['width']
            h = img_meta['height']
            # retrieve metadata for all annotations in the current image
            anns = coco.loadAnns(ann_ids)

            yield [img_id, img_file_name, w, h, anns]

    def save_to_csv(self, save_path : str) -> None:
        if(self.__images_df.empty or self.__annotations_df.empty):
            print("Check convert_to_df before call")
        else:           
            if(save_path != None):
                self.__save_path = save_path
            directory, _ = os.path.split(self.__save_path)
            if os.path.isdir(directory) and os.access(directory, os.R_OK):
                self.__train_coco_df = pd.merge(self.__images_df, self.__annotations_df, right_index=True, left_index=True)
                self.__train_coco_df.to_csv(self.__save_path, sep=',', na_rep='NaN')
            else:
                print("Check either the file is missing or not readable")

    def get_df_or_none(self):
        if(self.__train_coco_df.empty):
            print("Check run convert_to_df before use it")
            return None
        else:
            return self.__train_coco_df

if __name__ == '__main__':
    root='../dataset/'
    train_annot_path = root + 'train.json' # train json 정보
    train_coco = COCO(train_annot_path) # load annotations for training set
    
    save_path = root + 'train_coco_df.csv'
    coco_img_annotation = COCO_ImageAnnotation(None)
    coco_img_annotation.convert_to_df(train_coco)
    coco_img_annotation.save_to_csv(save_path)
    train_df = coco_img_annotation.get_df_or_none()
    print(train_df)
