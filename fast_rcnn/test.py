from torchvision.ops import nms
from data_augment import class_to_id
from torch_snippets import *
from utils import device, read_image_cv2
import time

def predict_on_test_set(frcnn_model, test_ds):

    gts = []
    preds = []

    total_images = len(test_ds.filenames)
    print(f"========== Predicting on test set with {total_images} images ==========")
    # For each image in the test dataset
    func_start_time = time.perf_counter()
    for img_idx in range(total_images):

        loop_start_time = time.perf_counter()

        # Open image
        img = read_image_cv2(test_ds.img_base_dir + test_ds.filenames[img_idx])
        H, W, _ = img.shape

        # Get the bbs and label for bbs of the image
        img_bbox_labels = np.array(test_ds.bbox_labels[img_idx])
        # First 4 columns are bx, by, bX, bY
        # Get in the 224x224 size since bbs will come from that
        img_bboxs = ((img_bbox_labels[:,:-1] / np.array([W,H,W,H])) * 224).astype(np.int16)
        # Last column is the label for the bbox
        img_labels = img_bbox_labels[:,-1:]
        img_labels = img_labels.squeeze() if img_labels.size > 1 else img_labels.squeeze(0)
        # This is ground truth
        gt = {
            'boxes': torch.from_numpy(img_bboxs).float().to(device), # Use float for map calc later
            'labels': torch.from_numpy(img_labels).to(device)
        }
        gts.append(gt)

        # Get prediction
        # Get the region proposals for the test image
        rois = test_ds.rois[img_idx]
        # These are just dummies filled with 0. Deltas and classes are not calculated for test datset
        roi_classes = test_ds.roi_classes[img_idx]
        roi_deltas = test_ds.roi_deltas[img_idx]
        # Pre-process data
        processed = test_ds.collate_fn([(img, rois, roi_classes, roi_deltas)])
        images, rois, roi_src_idxs, _, _ = processed
        with torch.no_grad():
            # Get predictions
            frcnn_model.eval()
            pred_classes, pred_deltas = frcnn_model(images, rois, roi_src_idxs)
            # Get only the most-likely class for each roi
            pred_classes = torch.nn.functional.softmax(pred_classes, -1)
            pred_classes_conf, pred_classes = torch.max(pred_classes, -1)

        # Detach and turn to numpy
        rois = np.array(rois.detach().cpu())
        pred_classes_conf = pred_classes_conf.detach().cpu().numpy()
        pred_classes = pred_classes.detach().cpu().numpy()
        pred_deltas = pred_deltas.detach().cpu().numpy()

        # Get only non-background rois
        non_bg_roi_idx = pred_classes != class_to_id['background']
        rois = rois[non_bg_roi_idx]
        pred_classes_conf = pred_classes_conf[non_bg_roi_idx]
        pred_classes = pred_classes[non_bg_roi_idx]
        pred_deltas = pred_deltas[non_bg_roi_idx] 

        # Get final bounding boxes
        pred_bboxes = (rois + pred_deltas)*224

        # Get only the max bbs using non-max suppression
        max_bb_ixs = nms(torch.tensor(pred_bboxes.astype(np.float32)), torch.tensor(pred_classes_conf), 0.05)
        pred_bboxes = pred_bboxes[max_bb_ixs]
        pred_classes_conf = pred_classes_conf[max_bb_ixs]
        pred_classes = pred_classes[max_bb_ixs]
        # If only one max bb, unsqueeze
        if len(max_bb_ixs) == 1:
            pred_classes_conf, pred_classes, pred_bboxes = [tensor[None] for tensor in [pred_classes_conf, pred_classes, pred_bboxes]]

        # Get bbs as ints
        pred_bboxes = pred_bboxes.astype(np.int16)

        # These are predictions
        pred = {
            'boxes': torch.from_numpy(pred_bboxes).float().to(device), # Use float for map calc later
            'scores': torch.from_numpy(pred_classes_conf).to(device),
            'labels': torch.from_numpy(pred_classes).to(device),
        }
        preds.append(pred)

        loop_time_elapsed = time.perf_counter() - loop_start_time
        print(f"{img_idx+1}/{len(test_ds.filenames)}: Processed in {loop_time_elapsed}")

    func_time_elapsed = time.perf_counter() - func_start_time
    print(f"========== Prediction finished. Time elapsed: {func_time_elapsed} ==========")

    return gts, preds