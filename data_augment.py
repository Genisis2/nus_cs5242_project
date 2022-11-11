from bs4 import BeautifulSoup
import glob
import albumentations as A
import cv2
import os
import pandas as pd
from utils import plot_image_with_bb, read_image_cv2


class ParseAnno:

    def __init__(self, path: str, class_to_id: dict, img_base_dir: str) -> None:
        self.path = path + '/*' if path[-1] != '/' else path + '*'
        self.annotations = glob.glob(self.path)
        self.class_to_id = class_to_id
        self.filenames = []
        self.bbox_label = []
        self.img_base_dir = img_base_dir
        self.final_dataframe = None

    def parse(self, save_csv=False):
        for anno in self.annotations:
            soup = BeautifulSoup(open(anno).read(), 'html.parser')
            self.filenames.append(soup.find('filename').string)
            objects = soup.find_all("object")
            bbox_label_i = []
            for o in objects:
                bbox_label_i.append((int(o.find('xmin').string), int(o.find('ymin').string), int(o.find(
                    'xmax').string), int(o.find('ymax').string), self.class_to_id[o.find('name').string]))
            self.bbox_label.append(bbox_label_i)

        assert len(self.filenames) == len(self.bbox_label)

        self.final_dataframe = pd.DataFrame(
            list(zip(self.filenames, self.bbox_label)), columns=['filename', 'bbox_label'])

        if save_csv:
            self.final_dataframe.to_csv(os.getcwd()+'/df.csv')

        return self.filenames, self.bbox_label

    def augment(self, transform_list: list, format='pascal_voc'):
        assert len(self.filenames) != 0

        final_transform = []
        for l in transform_list:
            final_transform.append(
                A.Compose([l], bbox_params=A.BboxParams(format=format)))

        temp_filenames = []
        temp_bbox_label = []

        for i, img in enumerate(self.filenames):
            image = read_image_cv2(self.img_base_dir+'/'+img)
            bboxes = self.bbox_label[i]
            for j, transform in enumerate(final_transform):
                transformed = transform(image=image, bboxes=bboxes)
                transformed_image = transformed['image']
                transformed_image = cv2.cvtColor(
                    transformed_image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(self.img_base_dir +
                            f'/{img.split(".")[0]}_{j}.png', transformed_image)
                transformed_bboxes = transformed['bboxes']

                temp_filenames.append(f'{img.split(".")[0]}_{j}.png')
                temp_bbox_label.append(transformed_bboxes)
                # plot_image_with_bb(transformed_image, transformed_bboxes)

        temp_dataframe = pd.DataFrame(
            list(zip(temp_filenames, temp_bbox_label)), columns=['filename', 'bbox_label'])
        self.final_dataframe = self.final_dataframe.append(temp_dataframe)
        return


# class_to_id = {'burger': 1, 'drinks': 2, 'fries': 3}
# parser = ParseAnno('annotations', class_to_id, 'data')
# parser.parse()
# parser.augment([A.HorizontalFlip(), A.RandomBrightnessContrast()])
# parser.final_dataframe.to_csv(os.getcwd()+'/df2.csv')
