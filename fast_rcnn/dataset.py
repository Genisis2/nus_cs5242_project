import time
from typing import Dict, List, Tuple
from utils import get_iou_score, read_image_cv2, get_ss_boxes, device, plot_image_with_bb
import pandas as pd
from ast import literal_eval
import os
from torch_snippets import *
from sklearn.model_selection import train_test_split
from v_rcnn.dataset import RCNNDataset, FE_INPUT_H, FE_INPUT_W, imagenet_normalize

class FastRCNNDataset(RCNNDataset):
    """Variant of RCNNDataset for Fast RCNN

    Doesn't crop image like RCNNDataset. Only gets the roi and roi_src_idxs.
    """
    def __init__(self, img_base_dir: str, data_df:pd.DataFrame, saved_ds_processing_fp:str=None):
        super().__init__(img_base_dir, data_df, saved_ds_processing_fp)

    def __getitem__(self, img_idx):
        
        # Get the image at this idx
        img_filepath = os.path.join(self.img_base_dir, self.filenames[img_idx])
        image = read_image_cv2(img_filepath)

        # Get the rois, roi classes, and roi offsets
        rois = self.rois[img_idx]
        roi_classes = self.roi_classes[img_idx]
        roi_deltas = self.roi_deltas[img_idx]

        return image, rois, roi_classes, roi_deltas

    def collate_fn(self, batch):

        images = []
        rois = []
        roi_src_idxs = [] 
        roi_classes = []
        roi_deltas = []

        # Process batch
        for batch_idx in range(len(batch)):
            _image, _rois, _roi_classes, _roi_deltas = batch[batch_idx]

            # Resize to input size of 224x224
            _image = cv2.resize(_image, (FE_INPUT_W, FE_INPUT_H))
            # Turn to (C,H,W) from (H,W,C)
            _image = torch.tensor(_image).permute(2,0,1)
            # Turn pixel values as a % of 255's
            _image = _image/255.
            # Normalize using normalization values of ImageNet
            _image = imagenet_normalize(_image)
            # Make sure we are dealing with a float tensor
            _image = _image.to(device).float()
            # Expand first dim so torch.cat works later
            _image = torch.unsqueeze(_image, 0)
            # Append to list
            images.append(_image)

            # Get rois and the idx of the source image for each roi
            rois.extend(_rois)
            roi_src_idxs.extend([batch_idx] * len(_rois))

            # Just extend the lists with the rois
            roi_classes.extend(_roi_classes)
            roi_deltas.extend(_roi_deltas)
        
        # Check
        assert (len(images) == len(batch)
                and len(rois) == len(roi_src_idxs)
                and len(roi_src_idxs) == len(roi_classes) 
                and len(roi_classes) == len(roi_deltas))

        # Create batched tensors
        images = torch.cat(images).to(device)
        rois = torch.Tensor(rois).float().to(device)
        roi_src_idxs = torch.Tensor(roi_src_idxs).float().to(device)
        roi_classes = torch.Tensor(roi_classes).long().to(device)
        roi_deltas = torch.Tensor(roi_deltas).float().to(device)

        return images, rois, roi_src_idxs, roi_classes, roi_deltas

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
    - train_dataset, train_dataset both of type FastRCNNDataset
    """
    # Create datasets for train and test
    train_dataset = FastRCNNDataset(img_root_dir, None, pickle_train_ds)
    test_dataset = FastRCNNDataset(img_root_dir, None, pickle_test_ds)

    return train_dataset, test_dataset
