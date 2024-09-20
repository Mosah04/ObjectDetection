import cv2
import numpy as np
import copy
from typing import Tuple, Dict


def augment(img_data: Dict, config, augment: bool = True) -> Tuple[Dict, np.ndarray]:
    """
    Augments the image data by applying horizontal/vertical flips and rotations based on config settings.

    :param img_data: Dictionary containing image metadata, such as bounding boxes and image dimensions.
    :param config: Configuration object with augmentation parameters.
    :param augment: Boolean flag to apply augmentations.
    :return: Tuple containing the augmented image data and the augmented image.
    """
    assert 'filepath' in img_data
    assert 'bboxes' in img_data
    assert 'width' in img_data
    assert 'height' in img_data

    # Deep copy to avoid modifying the original data
    img_data_aug = copy.deepcopy(img_data)
    img = cv2.imread(img_data_aug['filepath'])

    if augment:
        rows, cols = img.shape[:2]

        # Apply horizontal flip
        if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
            img = np.fliplr(img)
            for bbox in img_data_aug['bboxes']:
                bbox['x1'], bbox['x2'] = cols - bbox['x2'], cols - bbox['x1']

        # Apply vertical flip
        if config.use_vertical_flips and np.random.randint(0, 2) == 0:
            img = np.flipud(img)
            for bbox in img_data_aug['bboxes']:
                bbox['y1'], bbox['y2'] = rows - bbox['y2'], rows - bbox['y1']

        # Apply random 90-degree rotations
        if config.rot_90:
            angle = np.random.choice([0, 90, 180, 270])
            if angle == 90:
                img = np.transpose(img, (1, 0, 2))
                img = np.fliplr(img)
            elif angle == 180:
                img = np.flipud(np.fliplr(img))
            elif angle == 270:
                img = np.transpose(img, (1, 0, 2))
                img = np.flipud(img)

            # Adjust bounding boxes accordingly for rotations
            for bbox in img_data_aug['bboxes']:
                x1, x2, y1, y2 = bbox['x1'], bbox['x2'], bbox['y1'], bbox['y2']
                if angle == 90:
                    bbox['x1'], bbox['x2'], bbox['y1'], bbox['y2'] = rows - y2, rows - y1, x1, x2
                elif angle == 180:
                    bbox['x1'], bbox['x2'], bbox['y1'], bbox['y2'] = cols - x2, cols - x1, rows - y2, rows - y1
                elif angle == 270:
                    bbox['x1'], bbox['x2'], bbox['y1'], bbox['y2'] = y1, y2, cols - x2, cols - x1

    # Update width and height in augmented data
    img_data_aug['width'] = img.shape[1]
    img_data_aug['height'] = img.shape[0]

    return img_data_aug, img
