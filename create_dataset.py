import enum
import glob
from utils import get_iou_score, read_image_cv2, get_ss_boxes, save_img
import cv2
import pandas as pd
from ast import literal_eval
import os
import shutil

MAX_RECT = 2000
MAX_NEG = 2000


class DataGeneratorRCNN:

    def __init__(self, id_to_class: dict, img_base_dir: str, pd_csv_path: str) -> None:
        self.id_to_class = id_to_class
        self.img_base_dir = img_base_dir
        self.pd_csv_path = pd_csv_path
        self.filenames = []
        self.bbox_labels = []

    def convert_csv(self):
        df = pd.read_csv(self.pd_csv_path, usecols=[
                         'filename', 'bbox_label'], converters={'bbox_label': literal_eval})
        self.filenames = df['filename'].to_list()
        self.bbox_labels = df['bbox_label'].to_list()

    def create(self):

        shutil.rmtree('dataset')
        
        os.mkdir('dataset')
        os.mkdir('dataset/burger')
        os.mkdir('dataset/drinks')
        os.mkdir('dataset/fries')
        os.mkdir('dataset/background')

        img_paths = glob.glob(self.img_base_dir+'/*')

        total_neg = 0

        for idx, img in enumerate(img_paths):
            image = read_image_cv2(img)
            ssboxes = get_ss_boxes(image)[: MAX_RECT]

            for i, y in enumerate(ssboxes):
                neg_count = 0
                bbox_i = self.bbox_labels[idx]

                for j, x in enumerate(bbox_i):
                    iou_score, is_overlap = get_iou_score(x, y)

                    if iou_score >= 0.7:
                        roi = image[y[1]:y[3], y[0]:y[2]] 
                        save_img(roi, f'dataset/{self.id_to_class[x[4]]}/{i, j}.png')
                        continue

                    if iou_score < 0.05:
                        total_neg += 1
                        neg_count += 1
                            
                    if neg_count == len(bbox_i) and not is_overlap and total_neg <= MAX_NEG:
                        roi = image[y[1]:y[3], y[0]:y[2]] 
                        save_img(roi, f'dataset/{self.id_to_class[0]}//{i, j}.png')


# id_to_label = {1: 'burger', 2: 'drinks', 3: 'fries', 0: 'background'}
# data_gen = DataGeneratorRCNN(id_to_label, 'data', 'df2.csv')
# data_gen.convert_csv()
# data_gen.create()