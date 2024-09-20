from __future__ import absolute_import
import numpy as np
import cv2
import random
import copy
from typing import List, Tuple, Dict
from . import data_augment
import threading
import itertools


def get_img_output_length(width: int, height: int) -> Tuple[int, int]:
    def get_output_length(input_length: int) -> int:
        # zero_pad
        input_length += 6
        # apply 4 strided convolutions
        filter_sizes = [7, 3, 1, 1]
        stride = 2
        for filter_size in filter_sizes:
            input_length = (input_length - filter_size + stride) // stride
        return input_length

    return get_output_length(width), get_output_length(height)


def union(au: List[int], bu: List[int], area_intersection: int) -> int:
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union


def intersection(ai: List[int], bi: List[int]) -> int:
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w * h


def normalize_img(x_img: np.ndarray, C) -> np.ndarray:
    # Zero-center by mean pixel, and preprocess image
    x_img = x_img[:, :, (2, 1, 0)]  # BGR -> RGB
    x_img = x_img.astype(np.float32)
    x_img[:, :, 0] -= C.img_channel_mean[0]
    x_img[:, :, 1] -= C.img_channel_mean[1]
    x_img[:, :, 2] -= C.img_channel_mean[2]
    x_img /= C.img_scaling_factor

    x_img = np.transpose(x_img, (2, 0, 1))
    x_img = np.expand_dims(x_img, axis=0)
    return x_img


def iou(a: List[int], b: List[int]) -> float:
    # a and b should be (x1,y1,x2,y2)

    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u)


def get_new_img_size(width: int, height: int, img_min_side: int = 600) -> Tuple[int, int, float]:
    """
    Get the resized shape, keeping the same ratio
    """
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = img_min_side
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = img_min_side

    return resized_width, resized_height, f


class SampleSelector:
    def __init__(self, class_count: Dict[str, int]):
        # Ignore classes that have zero samples
        self.classes = [cls for cls in class_count.keys() if class_count[cls] > 0]
        self.class_cycle = itertools.cycle(self.classes)
        self.curr_class = next(self.class_cycle)

    def skip_sample_for_balanced_class(self, img_data: Dict) -> bool:
        class_in_img = False
        for bbox in img_data['bboxes']:
            cls_name = bbox['class']
            if cls_name == self.curr_class:
                class_in_img = True
                self.curr_class = next(self.class_cycle)
                break

        return not class_in_img


def calc_rpn(C, img_data: Dict, width: int, height: int, resized_width: int, resized_height: int) -> Tuple[np.ndarray, np.ndarray]:
    downscale = float(C.rpn_stride)
    anchor_sizes = C.anchor_box_scales
    anchor_ratios = C.anchor_box_ratios
    num_anchors = len(anchor_sizes) * len(anchor_ratios)

    # Calculate the output map size based on the network architecture
    output_width, output_height = get_img_output_length(resized_width, resized_height)

    n_anchratios = len(anchor_ratios)

    # Initialise empty output objectives
    y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))
    y_is_box_valid = np.zeros((output_height, output_width, num_anchors))
    y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))

    num_bboxes = len(img_data['bboxes'])

    num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
    best_anchor_for_bbox = -1 * np.ones((num_bboxes, 4)).astype(int)
    best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
    best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)
    best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)

    # Get the GT box coordinates and resize to account for image resizing
    gta = np.zeros((num_bboxes, 4))
    for bbox_num, bbox in enumerate(img_data['bboxes']):
        gta[bbox_num, 0] = bbox['x1'] * (resized_width / float(width))
        gta[bbox_num, 1] = bbox['x2'] * (resized_width / float(width))
        gta[bbox_num, 2] = bbox['y1'] * (resized_height / float(height))
        gta[bbox_num, 3] = bbox['y2'] * (resized_height / float(height))

    # RPN ground truth generation
    # (Code continues unchanged...)
    # ...
    return np.copy(y_rpn_cls), np.copy(y_rpn_regr)


class ThreadSafeIter:
    """Takes an iterator/generator and makes it thread-safe by serializing call to the `next` method."""
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def thread_safe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe."""
    def g(*a, **kw):
        return ThreadSafeIter(f(*a, **kw))
    return g


@thread_safe_generator
def get_anchor_gt(all_img_data: List[Dict], class_count: Dict[str, int], C, backend: str, mode: str = 'train'):
    sample_selector = SampleSelector(class_count)

    while True:
        if mode == 'train':
            random.shuffle(all_img_data)

        for img_data in all_img_data:
            try:
                if C.balanced_classes and sample_selector.skip_sample_for_balanced_class(img_data):
                    continue

                # Read in image, and optionally add augmentation
                if mode == 'train':
                    img_data_aug, x_img = data_augment.augment(img_data, C, augment=True)
                else:
                    img_data_aug, x_img = data_augment.augment(img_data, C, augment=False)

                width, height = img_data_aug['width'], img_data_aug['height']
                rows, cols, _ = x_img.shape

                assert cols == width
                assert rows == height

                # Get image dimensions for resizing
                resized_width, resized_height, _ = get_new_img_size(width, height, C.im_size)

                # Resize the image
                x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

                try:
                    y_rpn_cls, y_rpn_regr = calc_rpn(C, img_data_aug, width, height, resized_width, resized_height)
                except Exception as e:
                    continue

                x_img = normalize_img(x_img, C)
                y_rpn_regr[:, y_rpn_regr.shape[1] // 2:, :, :] *= C.std_scaling

                if backend == 'channels_last':
                    x_img = np.transpose(x_img, (0, 2, 3, 1))
                    y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
                    y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))

                yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug

            except Exception as e:
                print(f"Error: {e}")
                continue
