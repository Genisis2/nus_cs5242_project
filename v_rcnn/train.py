from typing import Tuple
from torchvision import datasets, transforms
import torch
from torch.utils.data import random_split
from torch_snippets import *
from sklearn.model_selection import KFold
from v_rcnn.dataset import _SAMPLE_SEED
from v_rcnn.model import RCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_EPOCHS = 5
BATCH_SIZE = 32
LR = 1e-3

def train_vrcnn(train_ds, num_epochs=N_EPOCHS, 
                batch_size=BATCH_SIZE, learning_rate=LR) -> Tuple[nn.Module, Report]:
    """Trains a vanilla R-CNN model
    
    Parameters:
    - train_ds
        - Dataset to train on
    - num_epochs
        - Number of epochs for training. Defaults to 5
    - batch_size
        - Size of each batch. Defaults to 32
    - learning_rate
        - Learning rate to use while training. Default to 1e-3

    Returns:
    - trained model
    - log report
    """

    # Split to train and validation
    train_subset, val_subset = random_split(train_ds, [0.9, 0.1], 
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

    rcnn = RCNN().to(device)
    optimizer = torch.optim.SGD(rcnn.parameters(), lr=learning_rate)

    # Start training
    log =  Report(num_epochs)
    for epoch in range(num_epochs):

        # Train model
        total_inputs = len(train_loader)
        pos = epoch
        for inputs in train_loader:
            
            # Unpack batch
            roi_crops, roi_classes, roi_deltas = inputs
            
            # Train model on batch
            rcnn.train()
            optimizer.zero_grad()
            _roi_classes, _roi_deltas = rcnn(roi_crops)
            
            # Calculate loss and bp
            loss, loc_loss, regr_loss = rcnn.calc_loss(
                    _roi_classes, _roi_deltas, roi_classes, roi_deltas)
            loss.backward()
            optimizer.step()

            # Calculate accuracy of predictions
            _, best_roi_class_pred  = _roi_classes.max(-1)
            accs = (roi_classes == best_roi_class_pred).cpu().numpy()

            pos += 1 / total_inputs
            log.record(pos, trn_loss=loss.item(), trn_loc_loss=loc_loss, 
                    trn_regr_loss=regr_loss, 
                    trn_acc=accs.mean(), end='\r')
        
        # Validate
        total_inputs = len(val_loader)
        pos = epoch
        for inputs in val_loader:
            
            # Unpack batch
            roi_crops, roi_classes, roi_deltas = inputs
            with torch.no_grad():
                # Get predictions
                rcnn.eval()
                _roi_classes, _roi_deltas = rcnn(roi_crops)
                
                # Calculate loss
                loss, loc_loss, regr_loss = rcnn.calc_loss(
                        _roi_classes, _roi_deltas, roi_classes, roi_deltas)
                
                # Calculate accuracy of predictions
                _, best_roi_class_pred  = _roi_classes.max(-1)
                accs = (roi_classes == best_roi_class_pred).cpu().numpy()
            
            pos += 1 / total_inputs
            log.record(pos, val_loss=loss.item(), val_loc_loss=loc_loss, 
                    val_regr_loss=regr_loss, 
                    val_acc=accs.mean(), end='\r')

    return rcnn, log


