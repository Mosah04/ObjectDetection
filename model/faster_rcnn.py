import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from model import losses as losses
from model import resnet as nn

def get_model(C, classes_count):
    img_input = Input(shape=(None, None, 3))
    roi_input = Input(shape=(C.num_rois, 4))

    # Define the base network (ResNet here)
    shared_layers = nn.nn_base(img_input, trainable=True)

    # Define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn = nn.rpn(shared_layers, num_anchors)

    # Define the classifier, built on the base layers
    classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)

    # Define the models: RPN, classifier, and combined model
    model_rpn = Model(img_input, rpn[:2])
    model_classifier = Model([img_input, roi_input], classifier)
    model_all = Model([img_input, roi_input], rpn[:2] + classifier)

    # Compile models with appropriate losses and optimizers
    model_rpn.compile(optimizer=Adam(learning_rate=1e-4), 
                      loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])

    model_classifier.compile(optimizer=Adam(learning_rate=1e-4), 
                             loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count)-1)], 
                             metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})

    model_all.compile(optimizer='sgd', loss='mae')

    return model_rpn, model_classifier, model_all
