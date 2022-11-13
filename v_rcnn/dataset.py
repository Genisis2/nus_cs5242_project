import glob
import time
from typing import Dict, List, Tuple
from utils import get_iou_score, read_image_cv2, get_ss_boxes, save_img, plot_image_with_bb
import pandas as pd
from ast import literal_eval
import os
import shutil
from torch_snippets import *
from sklearn.model_selection import train_test_split
from data_augment import class_to_id
import pickle

MAX_RECT = 2000
_SAMPLE_SEED = 42

class RCNNDataset(Dataset):

    def __init__(self, img_base_dir: str, data_df:pd.DataFrame, saved_ds_processing_fp:str=None):
        """
        Parameters:
        - `img_base_dir`:str
            - Path to the base directory that holds images
        - `data_df`:pd.DataFrame
            - DataFrame holding information from df2.csv
            - Set to `None` if will be using `saved_ds_processing_fp`
        - `saved_ds_processing_fp`
            - Path to the pickled dict holding data of this dataset
        """
        
        self.img_base_dir = img_base_dir
        
        # Takes very long to process so use saved processing if possible
        if saved_ds_processing_fp is not None:
            with open(saved_ds_processing_fp, 'rb') as fs:
                saved_ds_processing = pickle.load(fs)
            # Use the saved processing here
            self.filenames = saved_ds_processing['filenames']
            self.bbox_labels = saved_ds_processing['bbox_labels']
            self.gt_bbs = saved_ds_processing['gt_bbs']
            self.ssboxes = saved_ds_processing['ssboxes']
            self.rois = saved_ds_processing['rois']
            self.roi_deltas = saved_ds_processing['roi_deltas']
            self.roi_classes = saved_ds_processing['roi_classes']
        
        else:
            self.filenames = data_df['filename'].to_list()
            self.bbox_labels = data_df['bbox_label'].to_list()
            # Extract this information in self.process_df()
            # Similar to code in Lab for Lecture 9
            self.gt_bbs = []
            self.ssboxes = []
            self.rois = []
            self.roi_deltas = [] 
            self.roi_classes = []
            self._process_df(self.filenames, self.bbox_labels)

    def _process_df(self, filenames:List[str], bbox_labels:List[List[int]]):

        total_files = len(filenames) # Used for tracking progress

        _func_counter_start = time.perf_counter()
        print(f"========== Processing {total_files} files for R-CNN dataset start ==========")
        
        # Process each image
        for idx, img_fn in enumerate(filenames):

            _loop_counter_start = time.perf_counter()

            # Open image
            img = read_image_cv2(self.img_base_dir + img_fn)
            H, W, _ = img.shape
            img_area = H * W

            # Get the bbs and label for bbs of the image
            img_bbox_labels = np.array(bbox_labels[idx])
            # First 4 columns are bx, by, bX, bY
            img_bboxs = img_bbox_labels[:,:-1]
            # Last column is the label for the bbox
            img_labels = img_bbox_labels[:,-1:]

            # Get ss proposals
            ssboxes = get_ss_boxes(img)[:MAX_RECT]

            # Process ss proposals
            rois = []
            roi_deltas = []
            roi_classes = []
            for ssbox in ssboxes:

                # Get only ssbox coords
                ssbox = ssbox[:4]
                ss_x, ss_y, ss_X, ss_Y = ssbox

                # Skip ssboxes that are too small or too large
                ssbox_area = (ss_X - ss_x) * (ss_Y - ss_y)
                if ssbox_area < 0.05*img_area or ssbox_area > img_area:
                    continue
                
                # Find bbox that this ssbox best matches
                best_iou = -1
                best_is_overlap = False
                best_bbox_idx = -1
                for bb_idx, bb in enumerate(img_bboxs):
                    iou_score, is_overlap = get_iou_score(bb, ssbox)
                    if iou_score > best_iou:
                        best_iou = iou_score
                        best_is_overlap = is_overlap
                        best_bbox_idx = bb_idx

                best_bb = img_bboxs[best_bbox_idx]
                best_label = img_labels[best_bbox_idx].item()

                # Ignore neutrals
                if best_is_overlap and best_iou < 0.5:
                    continue

                # Get roi bounds as % of width and height
                roi = ssbox / np.array([W,H,W,H])
                rois.append(roi)

                # Get delta of roi from bb as % of width and height
                _x, _y, _X, _Y = best_bb
                delta = np.array([_x-ss_x, _y-ss_y, _X-ss_X, _Y-ss_Y]) / np.array([W,H,W,H])
                roi_deltas.append(delta)
                
                # Positive match
                if best_iou >= 0.5:
                    roi_classes.append(best_label)
                # Negative match (background)
                else:
                    roi_classes.append(class_to_id['background'])

            # Store in lists
            self.gt_bbs.append(img_bboxs/np.array([W,H,W,H])) # get gt bbox as % of width and height
            self.ssboxes.append(ssboxes)
            self.rois.append(rois)
            self.roi_deltas.append(roi_deltas)
            self.roi_classes.append(roi_classes)
            
            # Log current progress
            fn_display_name = img_fn[:15] + "..." if len(img_fn) > 15 else img_fn
            _loop_counter_elapsed_time = time.perf_counter() - _loop_counter_start
            print(f"Processed {idx+1}/{total_files} file {fn_display_name}. Time elapsed: {_loop_counter_elapsed_time}")

        # Log total time elapsed
        _func_counter_elapsed_time = time.perf_counter() - _func_counter_start
        print(f"========== Finished processing data for R-CNN dataset. Total time elapsed: {_func_counter_elapsed_time} ==========")

    def save(self, save_fp:str=None):
        """Saves the data held in this object"""

        if str is None:
            return

        # Create dict holding data to save
        data = {
            'filenames': self.filenames,
            'bbox_labels': self.bbox_labels,
            'gt_bbs': self.gt_bbs,
            'ssboxes': self.ssboxes,
            'rois': self.rois,
            'roi_deltas': self.roi_deltas, 
            'roi_classes': self.roi_classes
        }

        # Save to file
        os.makedirs(os.path.dirname(save_fp), exist_ok=True)
        with open(save_fp, 'wb') as fs:
            pickle.dump(data, fs)

    def __getitem__(self, ix):
        pass

    def collate_fn(self, batch):
        pass
    
    def __len__(self): 
        return len(self.data_df)

def create_train_test_dataset(img_root_dir:str, pd_csv_path:str, limit:int=None) -> Tuple[Dataset, Dataset]:
    """Creates the train and test datasets for V-RCNN training and eval

    Parameters:
    - img_root_dir:str
        - String path to the directory holding the files mentioned in pd_csv_path
    - pd_csv_path:str
        - String path to the CSV file holding information on the dataset
    - limit:int
        - Limit the number of images to include in the dataset. Default None meaning all images

    Returns:
    - train_dataset, train_dataset both of type VRCNNDataset
    """
    # Read the df2.csv file
    data_df = pd.read_csv(pd_csv_path, usecols=[
                        'filename', 'bbox_label'], converters={'bbox_label': literal_eval})

    # Artificially limit the data if limit set
    if limit is not None:
        data_df = data_df.iloc[:limit]
    
    # Split the images to train and test
    train_data_df, test_data_df = train_test_split(
        data_df, 
        train_size=0.8, 
        random_state=_SAMPLE_SEED, 
        shuffle=True,
    )

    # Create datasets for train and test
    train_dataset = RCNNDataset(img_root_dir, train_data_df)
    test_dataset = RCNNDataset(img_root_dir, test_data_df)

    return train_dataset, test_dataset

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