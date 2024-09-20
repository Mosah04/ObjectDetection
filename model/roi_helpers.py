import numpy as np
import math
from typing import Tuple, List, Optional
import copy
from . import data_generators


def calc_iou(R: np.ndarray, img_data: dict, C, class_mapping: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bboxes = img_data['bboxes']
    width, height = img_data['width'], img_data['height']
    resized_width, resized_height, _ = data_generators.get_new_img_size(width, height, C.im_size)

    gta = np.zeros((len(bboxes), 4))

    for bbox_num, bbox in enumerate(bboxes):
        gta[bbox_num, 0] = int(round(bbox['x1'] * (resized_width / width) / C.rpn_stride))
        gta[bbox_num, 1] = int(round(bbox['x2'] * (resized_width / width) / C.rpn_stride))
        gta[bbox_num, 2] = int(round(bbox['y1'] * (resized_height / height) / C.rpn_stride))
        gta[bbox_num, 3] = int(round(bbox['y2'] * (resized_height / height) / C.rpn_stride))

    x_roi = []
    y_class_num = []
    y_class_regr_coords = []
    y_class_regr_label = []

    for ix in range(R.shape[0]):
        x1, y1, x2, y2 = map(int, map(round, R[ix, :]))
        best_iou = 0.0
        best_bbox = -1

        for bbox_num, bbox in enumerate(bboxes):
            curr_iou = data_generators.iou(
                [gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]],
                [x1, y1, x2, y2]
            )
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_bbox = bbox_num

        if best_iou < C.classifier_min_overlap:
            continue

        w, h = x2 - x1, y2 - y1
        x_roi.append([x1, y1, w, h])

        if C.classifier_min_overlap <= best_iou < C.classifier_max_overlap:
            cls_name = 'bg'
        elif best_iou >= C.classifier_max_overlap:
            cls_name = bboxes[best_bbox]['class']
            cxg = (gta[best_bbox, 0] + gta[best_bbox, 1]) / 2.0
            cyg = (gta[best_bbox, 2] + gta[best_bbox, 3]) / 2.0
            cx, cy = x1 + w / 2.0, y1 + h / 2.0

            tx = (cxg - cx) / w
            ty = (cyg - cy) / h
            tw = np.log((gta[best_bbox, 1] - gta[best_bbox, 0]) / w)
            th = np.log((gta[best_bbox, 3] - gta[best_bbox, 2]) / h)
        else:
            raise RuntimeError(f'roi = {best_iou}')

        class_num = class_mapping[cls_name]
        class_label = [0] * len(class_mapping)
        class_label[class_num] = 1
        y_class_num.append(class_label)

        coords = [0] * 4 * (len(class_mapping) - 1)
        labels = [0] * 4 * (len(class_mapping) - 1)
        if cls_name != 'bg':
            label_pos = 4 * class_num
            sx, sy, sw, sh = C.classifier_regr_std
            coords[label_pos:label_pos + 4] = [sx * tx, sy * ty, sw * tw, sh * th]
            labels[label_pos:label_pos + 4] = [1, 1, 1, 1]
        y_class_regr_coords.append(copy.deepcopy(coords))
        y_class_regr_label.append(copy.deepcopy(labels))

    if not x_roi:
        return None, None, None

    X = np.expand_dims(np.array(x_roi), axis=0)
    Y1 = np.expand_dims(np.array(y_class_num), axis=0)
    Y2 = np.expand_dims(np.concatenate([np.array(y_class_regr_label), np.array(y_class_regr_coords)], axis=1), axis=0)

    return X, Y1, Y2


def apply_regr(x: int, y: int, w: int, h: int, tx: float, ty: float, tw: float, th: float) -> Tuple[int, int, int, int]:
    try:
        cx, cy = x + w / 2., y + h / 2.
        cx1, cy1 = tx * w + cx, ty * h + cy
        w1, h1 = np.exp(tw) * w, np.exp(th) * h
        x1, y1 = int(round(cx1 - w1 / 2.)), int(round(cy1 - h1 / 2.))
        w1, h1 = int(round(w1)), int(round(h1))

        return x1, y1, w1, h1

    except (ValueError, OverflowError):
        return x, y, w, h


def apply_regr_np(X: np.ndarray, T: np.ndarray) -> np.ndarray:
    try:
        x, y, w, h = X[0, :, :], X[1, :, :], X[2, :, :], X[3, :, :]
        tx, ty, tw, th = T[0, :, :], T[1, :, :], T[2, :, :], T[3, :, :]

        cx, cy = x + w / 2., y + h / 2.
        cx1, cy1 = tx * w + cx, ty * h + cy
        w1, h1 = np.exp(tw.astype(np.float64)) * w, np.exp(th.astype(np.float64)) * h

        x1, y1 = np.round(cx1 - w1 / 2.), np.round(cy1 - h1 / 2.)
        w1, h1 = np.round(w1), np.round(h1)

        return np.stack([x1, y1, w1, h1])
    except Exception as e:
        print(f"Error in apply_regr_np: {e}")
        return X


def non_max_suppression_fast(boxes: np.ndarray, probs: np.ndarray, overlap_thresh: float = 0.9, max_boxes: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    if len(boxes) == 0:
        return np.array([]), np.array([])

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    np.testing.assert_array_less(x1, x2)
    np.testing.assert_array_less(y1, y2)

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(probs)

    while idxs.size > 0:
        last = idxs[-1]
        i = idxs[last]
        pick.append(i)

        xx1_int = np.maximum(x1[i], x1[idxs[:-1]])
        yy1_int = np.maximum(y1[i], y1[idxs[:-1]])
        xx2_int = np.minimum(x2[i], x2[idxs[:-1]])
        yy2_int = np.minimum(y2[i], y2[idxs[:-1]])

        ww_int = np.maximum(0, xx2_int - xx1_int + 0.5)
        hh_int = np.maximum(0, yy2_int - yy1_int + 0.5)

        area_int = ww_int * hh_int
        area_union = area[i] + area[idxs[:-1]] - area_int
        overlap = area_int / (area_union + 1e-6)

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

        if len(pick) >= max_boxes:
            break

    return boxes[pick].astype("int"), probs[pick]


def rpn_to_roi(rpn_layer: np.ndarray, regr_layer: np.ndarray, C, dim_ordering: str, use_regr: bool = True, max_boxes: int = 300, overlap_thresh: float = 0.9) -> np.ndarray:
    regr_layer /= C.std_scaling

    anchor_sizes = C.anchor_box_scales
    anchor_ratios = C.anchor_box_ratios

    assert rpn_layer.shape[0] == 1

    if dim_ordering == 'channels_first':
        rows, cols = rpn_layer.shape[2:]
    elif dim_ordering == 'channels_last':
        rows, cols = rpn_layer.shape[1:3]

    A = np.zeros((4, *rpn_layer.shape[1:3], rpn_layer.shape[3]))

    curr_layer = 0
    for anchor_size in anchor_sizes:
        for anchor_ratio in anchor_ratios:
            anchor_x = (anchor_size * anchor_ratio[0]) / C.rpn_stride
            anchor_y = (anchor_size * anchor_ratio[1]) / C.rpn_stride

            if dim_ordering == 'channels_first':
                regr = regr_layer[0, 4 * curr_layer:4 * curr_layer + 4, :, :]
            else:
                regr = np.transpose(regr_layer[0, :, :, 4 * curr_layer:4 * curr_layer + 4], (2, 0, 1))

            X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
            A[0, :, :, curr_layer], A[1, :, :, curr_layer] = X - anchor_x / 2, Y - anchor_y / 2
            A[2, :, :, curr_layer], A[3, :, :, curr_layer] = anchor_x, anchor_y

            if use_regr:
                A[:, :, :, curr_layer] = apply_regr_np(A[:, :, :, curr_layer], regr)

            A[2:, :, :, curr_layer] = np.maximum(1, A[2:, :, :, curr_layer])
            A[2:, :, :, curr_layer] += A[:2, :, :, curr_layer]
            A[:2, :, :, curr_layer] = np.maximum(0, A[:2, :, :, curr_layer])
            A[2:, :, :, curr_layer] = np.minimum([cols - 1, rows - 1], A[2:, :, :, curr_layer])

            curr_layer += 1

    all_boxes = np.reshape(A.transpose((0, 3, 1, 2)), (4, -1)).transpose((1, 0))
    all_probs = rpn_layer.transpose((0, 3, 1, 2)).reshape((-1))

    idxs = np.where((all_boxes[:, 0] >= all_boxes[:, 2]) | (all_boxes[:, 1] >= all_boxes[:, 3]))
    all_boxes, all_probs = np.delete(all_boxes, idxs, axis=0), np.delete(all_probs, idxs, axis=0)

    return non_max_suppression_fast(all_boxes, all_probs, overlap_thresh=overlap_thresh, max_boxes=max_boxes)[0]
