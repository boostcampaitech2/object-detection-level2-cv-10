import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import cv2

from pycocotools.coco import COCO

# TRAIN_DATA_PATH = "/opt/ml/Git/object-detection-level2-cv-10/data"
# coco = COCO(f"{TRAIN_DATA_PATH}/stratified_valid.10fold.wArea.json")


class DrawBBox:
    def __init__(self, coco_data, save_path='valid_pred_images'):
        self.data_path = '/opt/ml/detection/dataset/train'
        if type(coco_data) == COCO:
            self.coco = coco_data
        elif type(coco_data) == str:
            self.coco = coco_data

        self.maxCol = 6
        self.maxRow = 6
        self.maxImg = self.maxCol * self.maxRow / 2
        self.save_path = save_path
        self.nIter = 0
        self.epoch = 0

        self.pltColorDict = {
            0: plt.cm.tab20(1),
            1: plt.cm.tab20(3),
            2: plt.cm.tab20(5),
            3: plt.cm.tab20(7),
            4: plt.cm.tab20(9),
            5: plt.cm.tab20(11),
            6: plt.cm.tab20(13),
            7: plt.cm.tab20(15),
            8: plt.cm.tab20(17),
            9: plt.cm.tab20(19),
        }

        self.catDict = {
            0: 'General trash',
            1: 'Paper',
            2: 'Paper pack',
            3: 'Metal',
            4: 'Glass',
            5: 'Plastic',
            6: 'Styrofoam',
            7: 'Plastic bag',
            8: 'Battery',
            9: 'Clothing',
        }

    def pltColor2RGB(self, color):
        _color = np.array(color) * 255
        return _color[:3]

    def draw_bboxes_from_coco(self, image, image_id, ax):
        # output: [x_min, y_min, w, h]
        cocoImgAnns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))
        _image = image.copy()
        for ann in cocoImgAnns:
            x_min = int(ann['bbox'][0])
            y_min = int(ann['bbox'][1])
            x_max = int(x_min + ann['bbox'][2])
            y_max = int(y_min + ann['bbox'][3])
            catId = ann['category_id']
            _color = self.pltColorDict[catId]
            _image = cv2.rectangle(_image, (x_min, y_min), (x_max, y_max),
                                   self.pltColor2RGB(_color), 3)
            # ax.text(x_min, y_min, cat_dict[_bbox[4]], color='w', fontsize=6, bbox=dict(facecolor=_color, lw=0))
            ax.text(x_min,
                    y_min,
                    str(catId),
                    ha='center',
                    va='center',
                    weight='bold',
                    fontsize=6,
                    bbox=dict(facecolor=_color, lw=0, pad=1))
        ax.imshow(_image)
        ax.text(0, -1, f"{image_id:04d}.jpg", size=8)
        ax.axis("off")

    def draw_bboxes_from_output(self, image, image_id, ax, output):
        # output: [x_min, y_min, x_max, y_max]
        _image = image.copy()
        for bbox, catId, score in zip(output['boxes'], output['labels'],
                                      output['scores']):
            x_min = int(bbox[0])
            y_min = int(bbox[1])
            x_max = int(bbox[2])
            y_max = int(bbox[3])
            catId -= 1
            _color = self.pltColorDict[catId]
            _image = cv2.rectangle(_image, (x_min, y_min), (x_max, y_max),
                                   self.pltColor2RGB(_color), 3)
            # ax.text(x_min, y_min, cat_dict[_bbox[4]], color='w', fontsize=6, bbox=dict(facecolor=_color, lw=0))
            ax.text(x_min,
                    y_min,
                    f"{catId}: {score:.3f}",
                    ha='center',
                    va='center',
                    weight='bold',
                    fontsize=5,
                    bbox=dict(facecolor=_color, lw=0, pad=1))
        ax.imshow(_image)
        ax.text(0, -1, f"pred: {image_id:04d}.jpg", size=8)
        ax.axis("off")

    def draw_bbox(self, epoch, image_ids, outputs):
        fig, axes = plt.subplots(
            self.maxRow,
            self.maxCol,
            figsize=(16, 16),
            #  dpi=200,
            sharey=True,
            sharex=True)

        nCol = 0
        nRow = 0
        self.nIter += 1
        for image_id, output in zip(image_ids, outputs):
            # for image_id in coco.getImgIds()[:5]:
            image = cv2.imread(f"{self.data_path}/{image_id:04d}.jpg",
                               cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.draw_bboxes_from_coco(image, image_id, axes[nRow, nCol])
            self.draw_bboxes_from_output(image, image_id, axes[nRow + 1, nCol],
                                         output)
            nCol += 1
            if nCol == self.maxCol:
                nCol = 0
                nRow += 2

        fig.tight_layout()
        fig.savefig(f"{self.save_path}/{epoch}/valid.{epoch}_{self.nIter}.jpg",
                    dpi=200)
        # fig.savefig(f"valid.{epoch}_{nIter}.jpg", dpi=300)
        plt.close()

    def batch_draw(self, epoch, image_ids, outputs):

        if self.epoch != epoch:
            self.nIter = 0
            self.epoch = epoch

        outputs = [{k: v.numpy() for k, v in t.items()} for t in outputs]

        Path(f"{self.save_path}/{epoch}").mkdir(parents=True, exist_ok=True)

        batch_len = len(image_ids)

        q = int(batch_len // self.maxImg)
        r = batch_len % self.maxImg

        prev_ind = 0
        for i in range(1, q + 1):
            ind = int(i * self.maxImg)
            self.draw_bbox(epoch, image_ids[prev_ind:ind],
                           outputs[prev_ind:ind])
            prev_ind = ind

        if r > 0:
            self.draw_bbox(epoch, image_ids[prev_ind:], outputs[prev_ind:])