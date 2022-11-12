import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch_snippets as snippets

def save_img(img, path, size=(224, 224)):

  img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  cv2.imwrite(path, img)

  return

def plot_image_with_bb(image, bboxs: list):
    id_to_color = {1: 'r', 2: 'y', 3:'b', 0: 'g'}
    fig, ax = plt.subplots()
    ax.imshow(image)
    for bbox in bboxs:
        xmin, ymin, xmax, ymax, label = bbox
        rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor=id_to_color[label], facecolor='none')
        ax.add_patch(rect)

    plt.show()

def plot_image_with_bb_and_label(image, bbox_labels: list, id_to_label: dict, title=None):
    bbs = []
    labels = []
    for bbox_label in bbox_labels:
        xmin, ymin, xmax, ymax, label = bbox_label
        bbs.append((xmin, ymin, xmax, ymax))
        labels.append(id_to_label[label])
    snippets.show(image, bbs=bbs, texts=labels, sz=10, title=title)

def get_iou_score(bbox, ssbox):
    xmin_bb, ymin_bb, xmax_bb, ymax_bb= bbox[:4]
    xmin_ss, ymin_ss, xmax_ss, ymax_ss= ssbox[:4]

    x1, y1, x2, y2 = max(xmin_bb, xmin_ss), max(ymin_bb, ymin_ss), min(xmax_bb, xmax_ss), min(ymax_bb, ymax_ss)

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    area_bb = abs(xmax_bb - xmin_bb) * abs(ymax_bb - ymin_bb)
    area_ss = abs(xmax_ss - xmin_ss) * abs(ymax_ss - ymin_ss)

    iou = inter_area / (area_bb + area_ss - inter_area)

    is_overlap = xmin_ss >= xmin_bb and ymin_ss >= ymax_bb and xmax_ss <= xmax_bb and ymax_ss <= ymax_bb

    return iou, is_overlap

def get_ss_boxes(image: bytes, ):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    ssbox= []
    for (x, y, w, h) in rects:
        ssbox.append((x, y, x + w, y + h, 0))
        
    return ssbox

def read_image_cv2(path: str) -> bytes:
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image