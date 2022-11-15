import time
from typing import Dict, List, Tuple
from utils import get_iou_score, read_image_cv2, get_ss_boxes, device, plot_image_with_bb
import pandas as pd
from ast import literal_eval
import os
from torch_snippets import *
from sklearn.model_selection import train_test_split
from data_augment import class_to_id
import pickle

IOU_POSITIVE_MATCH = 0.3
MAX_RECT = 2000
FE_INPUT_W = 224
FE_INPUT_H = 224

_SAMPLE_SEED = 42

# For use in normalization
# Uses ImageNet values for normalization since we are using a pretrained ResNet50 network
# Addtnl Ref: https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

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
                if best_is_overlap and best_iou < IOU_POSITIVE_MATCH:
                    continue

                # Get roi bounds as % of width and height
                roi = ssbox / np.array([W,H,W,H])
                rois.append(roi)

                # Get delta of roi from bb as % of width and height
                _x, _y, _X, _Y = best_bb
                delta = np.array([_x-ss_x, _y-ss_y, _X-ss_X, _Y-ss_Y]) / np.array([W,H,W,H])
                roi_deltas.append(delta)
                
                # Positive match
                if best_iou >= IOU_POSITIVE_MATCH:
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

    def __getitem__(self, img_idx):
        raise NotImplementedError() # Override in subclasses

    def collate_fn(self, batch):
        raise NotImplementedError() # Override in subclasses
    
    def __len__(self): 
        return len(self.filenames)

class VRCNNDataset(RCNNDataset):
    """Variant of RCNNDataset for vanilla RCNN

    Main property of this dataset is the idx to img,roi mapping
    """
    def __init__(self, img_base_dir: str, data_df:pd.DataFrame, saved_ds_processing_fp:str=None, cached_images_limit:int=0):
        super().__init__(img_base_dir, data_df, saved_ds_processing_fp)

        # Form a flattened list of the roi crops
        self.img_roi_idxs = []
        for img_idx in range(len(self.filenames)):
            # Get rois of image
            img_rois = self.rois[img_idx]
            for roi_idx in range(len(img_rois)):
                # Track the image source and the roi information
                self.img_roi_idxs.append((img_idx, roi_idx))

        # Dict to hold cached images
        self.cached_images_limit = cached_images_limit
        self.cached_images = {}

    def _get_image(self, crop_img_fn):
        """Gets the image from disk or from cache if cached"""

        if crop_img_fn in self.cached_images:
            return self.cached_images[crop_img_fn]

        # Get the image
        img_filepath = os.path.join(self.img_base_dir, crop_img_fn)
        img = read_image_cv2(img_filepath)

        # If limit is < 1, then that means don't cache
        if self.cached_images_limit < 1:
            return img

        # Else, cache
        if len(self.cached_images) >= self.cached_images_limit:
            #  Remove oldest cached if limit reached
            self.cached_images.pop(next(iter(self.cached_images)))
        # Now, actually store in cache
        self.cached_images[crop_img_fn] = img

        return img

    def _clear_cached_images(self):
        """Empties the cached data"""
        self.cached_images = {}

    def __getitem__(self, idx):

        # Get the roi crop at this idx
        img_idx, roi_idx = self.img_roi_idxs[idx]

        # Get the image to crop
        crop_img_fn = self.filenames[img_idx]
        img = self._get_image(crop_img_fn)
        H, W, _ = img.shape

        # Get the region proposal crop from the image
        crop_roi = self.rois[img_idx][roi_idx]
        x,y,X,Y = (crop_roi * np.array([W,H,W,H])).astype(np.uint16)
        roi_crop = img[y:Y,x:X]

        # Get the roi classes and roi offsets
        roi_class = self.roi_classes[img_idx][roi_idx]
        roi_delta = self.roi_deltas[img_idx][roi_idx]

        return roi_crop, roi_class, roi_delta

    def collate_fn(self, batch):

        roi_crops = [] 
        roi_classes = []
        roi_deltas = []

        # Process batch
        for ix in range(len(batch)):
            _roi_crop, _roi_class, _roi_delta = batch[ix]

            # Resize to input size of 224x224
            _roi_crop = cv2.resize(_roi_crop, (FE_INPUT_W, FE_INPUT_H))
            # Turn to (C,H,W) from (H,W,C)
            _roi_crop = torch.tensor(_roi_crop).permute(2,0,1)
            # Turn pixel values as a % of 255's
            _roi_crop = _roi_crop/255.
            # Normalize using normalization values of ImageNet
            _roi_crop = imagenet_normalize(_roi_crop)
            # Make sure we are dealing with a float tensor
            _roi_crop = _roi_crop.to(device).float()
            # Expand first dim so torch.cat works later
            _roi_crop = torch.unsqueeze(_roi_crop, 0)
            # Append to list
            roi_crops.append(_roi_crop)
            
            # Append other roi info into the lists
            roi_classes.append(_roi_class)
            roi_deltas.append(_roi_delta)
        
        # Check. Expect all lists to have the len of rois
        assert (len(roi_crops) == len(roi_classes) 
                and len(roi_classes) == len(roi_deltas))

        # Create batched tensors
        roi_crops = torch.cat(roi_crops).to(device)
        roi_classes = torch.Tensor(roi_classes).long().to(device)
        roi_deltas = torch.Tensor(roi_deltas).float().to(device)

        return roi_crops, roi_classes, roi_deltas
    
    def __len__(self): 
        return len(self.img_roi_idxs)

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
    train_dataset = VRCNNDataset(img_root_dir, train_data_df)
    test_dataset = VRCNNDataset(img_root_dir, test_data_df)

    return train_dataset, test_dataset

def create_train_test_dataset_from_pickle(
        img_root_dir:str, pickle_train_ds:str, pickle_test_ds:str) -> Tuple[Dataset, Dataset]:
    """Creates the train and test datasets for V-RCNN training and eval

    Parameters:
    - img_root_dir:str
        - String path to the directory holding the files mentioned in pd_csv_path
    - pickle_train_ds:str
        - String path to the pickled file for train dataset
    - pickle_test_ds:str
        - String path to the pickled file for test dataset

    Returns:
    - train_dataset, train_dataset both of type VRCNNDataset
    """
    # Create datasets for train and test
    train_dataset = VRCNNDataset(img_root_dir, None, pickle_train_ds)
    test_dataset = VRCNNDataset(img_root_dir, None, pickle_test_ds)

    return train_dataset, test_dataset
