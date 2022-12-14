from typing import Tuple
from torchvision import datasets, transforms
import torch
from torch.utils.data import random_split
from torch_snippets import *
from sklearn.model_selection import KFold
from v_rcnn.dataset import _SAMPLE_SEED
from v_rcnn.model import RCNN
from utils import device
import math

N_EPOCHS = 5
BATCH_SIZE = 32
LR = 1e-3

def train_fastrcnn(fastrcnn_model, train_ds, num_epochs=N_EPOCHS, 
                batch_size=BATCH_SIZE, learning_rate=LR) -> Report:
    """Trains a vanilla R-CNN model
    
    Parameters:
    - fastrcnn_model
        - FastRCNN model to train
    - train_ds
        - Dataset to train on
    - num_epochs
        - Number of epochs for training. Defaults to 5
    - batch_size
        - Size of each batch. Defaults to 32
    - learning_rate
        - Learning rate to use while training. Default to 1e-3

    Returns:
    - log report
    """

    # Split to train and validation
    total_ds_len = len(train_ds)
    train_ds_len = math.floor(0.9*total_ds_len)
    val_ds_len = total_ds_len - train_ds_len
    train_subset, val_subset = random_split(train_ds, [train_ds_len, val_ds_len], 
            generator=torch.Generator().manual_seed(_SAMPLE_SEED))

    # Define data loaders for training and validation data
    train_loader = torch.utils.data.DataLoader(
                    train_subset, batch_size=batch_size, 
                    collate_fn=train_ds.collate_fn,
                    drop_last=False)
    val_loader = torch.utils.data.DataLoader(
                    val_subset, batch_size=batch_size, 
                    collate_fn=train_ds.collate_fn,
                    drop_last=False)

    optimizer = torch.optim.SGD(fastrcnn_model.parameters(), lr=learning_rate)

    # Start training
    log = Report(num_epochs)
    for epoch in range(num_epochs):

        # Train model
        total_inputs = len(train_loader)
        pos = epoch
        for inputs in train_loader:
            
            # Unpack batch
            images, rois, roi_src_idxs, roi_classes, roi_deltas = inputs
            
            # Train model on batch
            fastrcnn_model.train()
            optimizer.zero_grad()
            _roi_classes, _roi_deltas = fastrcnn_model(images, rois, roi_src_idxs)
            
            # Calculate loss and bp
            loss, cls_loss, regr_loss = fastrcnn_model.calc_loss(
                    _roi_classes, _roi_deltas, roi_classes, roi_deltas)
            loss.backward()
            optimizer.step()

            # Calculate accuracy of predictions
            _, best_roi_class_preds  = _roi_classes.max(-1)
            accs = (roi_classes == best_roi_class_preds).cpu().numpy()

            pos += 1 / total_inputs
            log.record(pos, trn_loss=loss.item(), trn_cls_loss=cls_loss, 
                    trn_regr_loss=regr_loss, 
                    trn_acc=accs.mean(), end='\r')
        
        # Validate
        total_inputs = len(val_loader)
        pos = epoch
        for inputs in val_loader:
            
            # Unpack batch
            images, rois, roi_src_idxs, roi_classes, roi_deltas = inputs
            with torch.no_grad():
                # Get predictions
                fastrcnn_model.eval()
                _roi_classes, _roi_deltas = fastrcnn_model(images, rois, roi_src_idxs)
                
                # Calculate loss
                loss, cls_loss, regr_loss = fastrcnn_model.calc_loss(
                        _roi_classes, _roi_deltas, roi_classes, roi_deltas)
                
                # Calculate accuracy of predictions
                _, best_roi_class_preds  = _roi_classes.max(-1)
                accs = (roi_classes == best_roi_class_preds).cpu().numpy()
            
            pos += 1 / total_inputs
            log.record(pos, val_loss=loss.item(), val_cls_loss=cls_loss, 
                    val_regr_loss=regr_loss, 
                    val_acc=accs.mean(), end='\r')
        
        print("")
        print(f"Current ave.: trn_loss={np.mean([v for _,v in log.trn_loss])} val_loss={np.mean([v for _,v in log.val_loss])} trn_acc={np.mean([v for _,v in log.trn_acc])} val_acc={np.mean([v for _,v in log.val_acc])}")

    return log


