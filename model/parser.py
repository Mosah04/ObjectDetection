import cv2
import numpy as np
import random
import pprint

def get_data(input_path):
    """
    Parse annotation data from the input file and return image data, class counts, and class mappings.
    
    Args:
        input_path (str): Path to the annotation file.

    Returns:
        all_data (list): List of dictionaries containing image metadata and bounding boxes.
        classes_count (dict): Dictionary mapping class names to their respective counts.
        class_mapping (dict): Dictionary mapping class names to numerical class labels.
    """
    all_imgs = {}
    classes_count = {}
    class_mapping = {}

    # Open the annotation file
    with open(input_path, 'r') as f:
        print('Parsing annotation files')

        for line in f:
            line_split = line.strip().split(',')
            if len(line_split) != 6:
                print(f"Skipping invalid line: {line}")
                continue

            filename, x1, y1, x2, y2, class_name = line_split

            # Update class counts
            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            # Map class names to an integer index
            if class_name not in class_mapping:
                class_mapping[class_name] = len(class_mapping)

            # Process each unique image
            if filename not in all_imgs:
                try:
                    img = cv2.imread(filename)
                    if img is None:
                        print(f"Error reading image {filename}. Skipping...")
                        continue
                except Exception as e:
                    print(f"Exception occurred while reading {filename}: {e}")
                    continue

                (rows, cols) = img.shape[:2]
                all_imgs[filename] = {
                    'filepath': filename,
                    'width': cols,
                    'height': rows,
                    'bboxes': [],
                    'imageset': 'trainval' if np.random.randint(0, 6) > 0 else 'test'
                }

            # Append bounding box to the image data
            all_imgs[filename]['bboxes'].append({
                'class': class_name,
                'x1': int(x1),
                'x2': int(x2),
                'y1': int(y1),
                'y2': int(y2)
            })

    # Convert the dictionary of images to a list
    all_data = list(all_imgs.values())

    # Add a background class if needed
    if 'bg' not in classes_count:
        classes_count['bg'] = 0
    if 'bg' not in class_mapping:
        class_mapping['bg'] = len(class_mapping)

    # Shuffle the data to ensure randomness in training
    random.shuffle(all_data)

    # Output the summary of training images per class
    print('Training images per class ({} classes):'.format(len(classes_count)))
    pprint.pprint(classes_count)

    return all_data, classes_count, class_mapping
