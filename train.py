import os
import sys
import time
import argparse
from datetime import datetime

import numpy as np
import pickle

from model import faster_rcnn
from model import config, data_generators
from model.parser import get_data
import model.roi_helpers as roi_helpers

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Progbar

sys.setrecursionlimit(10000)

# Argument parser
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='Object Detection')
parser.add_argument("-p", "--path", default=None, help="path to annotation file")
parser.add_argument("--save_dir", default="./save", help="path to save directory")
parser.add_argument('--n_epochs', default=10, type=int, metavar='N', help='number of epochs')
parser.add_argument('--n_iters', default=10, type=int, metavar='N', help='number of iterations')
parser.add_argument('--horizontal_flips', action='store_true', help='augment with horizontal flips (Default: False)')
parser.add_argument('--vertical_flips', action='store_true', help='augment with vertical flips (Default: False)')
parser.add_argument('--rot_90', action='store_true', help='augment with 90 degree rotations (Default: False)')

def main():
    args = parser.parse_args()
    time_stamp = "{0:%Y%m%d-%H%M%S}".format(datetime.now())
    save_name = os.path.join(args.save_dir, f"train_{time_stamp}")

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    if args.path is None:
        raise OSError("Path to annotation file is required.")

    # Load configuration
    C = config.Config()
    C.config_filename = save_name + "_config.pickle"
    C.model_path = save_name + "_model.h5"
    C.use_horizontal_flips = args.horizontal_flips
    C.use_vertical_flips = args.vertical_flips
    C.rot_90 = args.rot_90

    all_imgs, classes_count, class_mapping = get_data(args.path)
    C.class_mapping = class_mapping

    with open(C.config_filename, 'wb') as config_f:
        pickle.dump(C, config_f)
        print(f"Path to config file: {C.config_filename}")

    train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
    val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

    # Data generators
    data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, K.image_data_format(), mode='train')
    data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, K.image_data_format(), mode='val')

    model_rpn, model_classifier, model_all = faster_rcnn.get_model(C, classes_count)

    losses = np.zeros((args.n_iters, 5))
    rpn_accuracy_rpn_monitor, rpn_accuracy_for_epoch = [], []
    best_loss = np.Inf

    with open('out.csv', 'w') as f:
        f.write('Accuracy,RPN classifier,RPN regression,Detector classifier,Detector regression,Total\n')

    iter_num = 0
    t0 = start_time = time.time()

    print(model_all.summary())
    try:
        for epoch_num in range(args.n_epochs):
            progbar = Progbar(args.n_iters)
            print(f'Epoch {epoch_num + 1}/{args.n_epochs}')

            while True:
                try:
                    if len(rpn_accuracy_rpn_monitor) == args.n_iters and C.verbose:
                        mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
                        rpn_accuracy_rpn_monitor = []
                        print(f'Average overlapping bounding boxes from RPN = {mean_overlapping_bboxes}')
                        if mean_overlapping_bboxes == 0:
                            print('RPN is not producing overlapping bounding boxes. Check RPN settings or keep training.')

                    X, Y, img_data = next(data_gen_train)
                    loss_rpn = model_rpn.train_on_batch(X, Y)
                    P_rpn = model_rpn.predict_on_batch(X)

                    R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_data_format(), use_regr=True, overlap_thresh=0.7, max_boxes=300)

                    X2, Y1, Y2 = roi_helpers.calc_iou(R, img_data, C, class_mapping)

                    neg_samples = np.where(Y1[0, :, -1] == 1)
                    pos_samples = np.where(Y1[0, :, -1] == 0)
                    neg_samples = neg_samples[0] if len(neg_samples) > 0 else []
                    pos_samples = pos_samples[0] if len(pos_samples) > 0 else []

                    rpn_accuracy_rpn_monitor.append(len(pos_samples))
                    rpn_accuracy_for_epoch.append(len(pos_samples))

                    selected_pos_samples = np.random.choice(pos_samples, C.num_rois // 2, replace=False).tolist() if len(pos_samples) >= C.num_rois // 2 else pos_samples.tolist()
                    selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist() if len(neg_samples) >= (C.num_rois - len(selected_pos_samples)) else neg_samples.tolist()

                    sel_samples = selected_pos_samples + selected_neg_samples

                    loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

                    if iter_num == args.n_iters:
                        loss_rpn_cls = np.mean(losses[:, 0])
                        loss_rpn_regr = np.mean(losses[:, 1])
                        loss_class_cls = np.mean(losses[:, 2])
                        loss_class_regr = np.mean(losses[:, 3])
                        class_acc = np.mean(losses[:, 4])

                        mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                        rpn_accuracy_for_epoch = []

                        print(f'Mean number of bounding boxes from RPN: {mean_overlapping_bboxes}')
                        print(f'Classifier accuracy: {class_acc}')
                        print(f'Loss RPN classifier: {loss_rpn_cls}')
                        print(f'Loss RPN regression: {loss_rpn_regr}')
                        print(f'Loss Detector classifier: {loss_class_cls}')
                        print(f'Loss Detector regression: {loss_class_regr}')
                        print(f'Elapsed time: {time.time() - start_time}[s]')

                        with open('out.csv', 'a') as f:
                            f.write(f'{class_acc},{loss_rpn_cls},{loss_rpn_regr},{loss_class_cls},{loss_class_regr},{loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr}\n')

                        curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                        iter_num = 0
                        start_time = time.time()

                        if curr_loss < best_loss:
                            print(f'Total loss decreased from {best_loss} to {curr_loss}, saving weights.')
                            best_loss = curr_loss
                            model_all.save_weights(C.model_path)

                        break

                    losses[iter_num, 0] = loss_rpn[1]
                    losses[iter_num, 1] = loss_rpn[2]
                    losses[iter_num, 2] = loss_class[1]
                    losses[iter_num, 3] = loss_class[2]
                    losses[iter_num, 4] = loss_class[3]
                    iter_num += 1

                    progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                                              ('detector_cls', np.mean(losses[:iter_num, 2])), ('detector_regr', np.mean(losses[:iter_num, 3]))])

                except Exception as e:
                    print(f'Exception: {e}')
                    continue

    except KeyboardInterrupt:
        t1 = time.time()
        print(f'\nIt took {t1 - t0:.2f}s')
        sys.exit('Keyboard Interrupt')

    print("Training is done")
    print(f'Path to config file: {C.config_filename}')

if __name__ == '__main__':
    main()
