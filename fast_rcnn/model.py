import torch.nn as nn
import torch.nn.functional as F
import torch
from data_augment import class_to_id, id_to_class
from torchvision import models
from utils import device
from torchvision.ops import RoIPool

background_class = class_to_id['background']

class FastRCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # Use a resnet backbone
        resnet_backbone = models.resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V2)
        # Output shape will be (1024, 14, 14)
        model = torch.nn.Sequential(*(list(resnet_backbone.children())[:-3]))
        for param in model.parameters():
            param.requires_grad = False
        model.eval().to(device)
        self.backbone = model

        # ROIPool
        roi_output_size = (7,7) # height, width
        roi_spatial_scale = 14 / 224  # 0.0625 scaling of the original image
        self.roipool = RoIPool(output_size=roi_output_size, spatial_scale=roi_spatial_scale) # 224/14 = 16; 1/16 = 0.0625
        
        # Flattened shape of output from RoiPool
        feature_dim = 1024 * roi_output_size[0] * roi_output_size[1]
        # 2 FC layers
        self.fc1 = nn.Linear(feature_dim, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.relu = nn.ReLU()

        # Classifier to label
        self.cls_score = nn.Linear(4096, len(id_to_class))

        # BBox offset linear regressor
        self.bbox = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.Tanh(),
        )

        self.bb_loss_weight = 10
        self.cel = nn.CrossEntropyLoss()
        self.sl1 = nn.L1Loss()

    def forward(self, images, rois, roi_src_idxs):

        # Extract features using resnet backbone
        fmap = self.backbone(images)

        # Get roi coord in shape of 224, the resized input image
        rois = rois * 224
        # Add the batch_idx as an element to rois to satisfy RoiPool inputs
        rois = torch.concat((roi_src_idxs.unsqueeze(1), rois), -1)
        roi_pool_output = self.roipool(fmap, rois)
        # Flatten output for passing through FC layers
        roi_pool_output_flat = torch.flatten(roi_pool_output, start_dim = 1)
        
        # Pass features through FC layers
        fc1_op = self.fc1(roi_pool_output_flat) 
        fc1_op = self.relu(fc1_op)
        fc2_op = self.fc2(fc1_op)
        fc2_op = self.relu(fc2_op)
        
        # Get estimated classification probabilities
        cls_score = self.cls_score(fc2_op)
        # Get estimated bbox offset
        bbox = self.bbox(fc2_op)

        return cls_score, bbox

    def calc_loss(self, pred_classes, pred_bb_offset, gt_classes, gt_bb_offset):

        # Get classification loss
        classification_loss = self.cel(pred_classes, gt_classes)

        # Keep only bb's that aren't background
        bb_idxs = torch.where(gt_classes != background_class)[0]
        # If have bb's that aren't background, calculate bb offset loss
        if len(bb_idxs) > 0:
            pred_bb_offset = pred_bb_offset[bb_idxs]
            gt_bb_offset = gt_bb_offset[bb_idxs]
            regression_loss = self.sl1(pred_bb_offset, gt_bb_offset)
        # Else no bb offset regression loss for all-backgound proposals
        else:
            regression_loss = torch.zeros(1, device=device)

        # Calculate all_loss
        all_loss = classification_loss + self.bb_loss_weight * regression_loss

        # Detach non-all loss losses
        classification_loss = classification_loss.detach()
        regression_loss = regression_loss.detach()

        return all_loss, classification_loss, regression_loss
