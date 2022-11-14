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

def decode(_y):
    _, preds = _y.max(-1)
    return preds

def train_batch(inputs, model, optimizer, criterion):
    input, rois, rixs, clss, deltas = inputs
    model.train()
    optimizer.zero_grad()
    _clss, _deltas = model(input, rois, rixs)
    loss, loc_loss, regr_loss = criterion(_clss, _deltas, clss, deltas)
    accs = clss == decode(_clss)
    loss.backward()
    optimizer.step()
    return loss.detach(), loc_loss, regr_loss, accs.cpu().numpy()

def validate_batch(inputs, model, criterion):
    input, rois, rixs, clss, deltas = inputs
    with torch.no_grad():
        model.eval()
        _clss,_deltas = model(input, rois, rixs)
        loss, loc_loss, regr_loss = criterion(_clss, _deltas, clss, deltas)
        _clss = decode(_clss)
        accs = clss == _clss
    return _clss, _deltas, loss.detach(), loc_loss, regr_loss, accs.cpu().numpy()

def train_fastrcnn(fastrcnn_model, train_ds, num_epochs=N_EPOCHS, 
                batch_size=BATCH_SIZE, learning_rate=LR) -> Tuple[nn.Module, Report]:
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
    criterion = fastrcnn_model.calc_loss

    # Start training
    log =  Report(num_epochs)
    for epoch in range(num_epochs):

        _n = len(train_loader)
        for ix, inputs in enumerate(train_loader):
            loss, loc_loss, regr_loss, accs = train_batch(inputs, fastrcnn_model, 
                                                        optimizer, criterion)
            pos = (epoch + (ix+1)/_n)
            log.record(pos, trn_loss=loss.item(), trn_loc_loss=loc_loss, 
                    trn_regr_loss=regr_loss, 
                    trn_acc=accs.mean(), end='\r')
            
        _n = len(val_loader)
        for ix,inputs in enumerate(val_loader):
            _clss, _deltas, loss, \
            loc_loss, regr_loss, accs = validate_batch(inputs, 
                                                    fastrcnn_model, criterion)
            pos = (epoch + (ix+1)/_n)
            log.record(pos, val_loss=loss.item(), val_loc_loss=loc_loss, 
                    val_regr_loss=regr_loss, 
                    val_acc=accs.mean(), end='\r')
    
    print("train_acc:", np.mean([v for pos, v in log.trn_acc]), "|val_acc:", np.mean([v for pos, v in log.val_acc]))
    return log


