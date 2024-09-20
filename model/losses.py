import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy

# Check image data format
if K.image_data_format() == 'channels_last':
    import tensorflow as tf

# Constants
lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0
lambda_cls_regr = 1.0
lambda_cls_class = 1.0
epsilon = 1e-4

# RPN loss function for bounding box regression
def rpn_loss_regr(num_anchors):
    def rpn_loss_regr_fixed_num(y_true, y_pred):
        if K.image_data_format() == 'channels_first':
            x = y_true[:, 4 * num_anchors:, :, :] - y_pred
            x_abs = K.abs(x)
            x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
            return lambda_rpn_regr * K.sum(
                y_true[:, :4 * num_anchors, :, :] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :4 * num_anchors, :, :])
        else:
            x = y_true[:, :, :, 4 * num_anchors:] - y_pred
            x_abs = K.abs(x)
            x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
            return lambda_rpn_regr * K.sum(
                y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :, :4 * num_anchors])

    return rpn_loss_regr_fixed_num

# RPN loss function for classification
def rpn_loss_cls(num_anchors):
    def rpn_loss_cls_fixed_num(y_true, y_pred):
        if K.image_data_format() == 'channels_last':
            return lambda_rpn_class * K.sum(
                y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_true[:, :, :, num_anchors:], y_pred)) / K.sum(epsilon + y_true[:, :, :, :num_anchors])
        else:
            return lambda_rpn_class * K.sum(
                y_true[:, :num_anchors, :, :] * K.binary_crossentropy(y_true[:, num_anchors:, :, :], y_pred)) / K.sum(epsilon + y_true[:, :num_anchors, :, :])

    return rpn_loss_cls_fixed_num

# Loss function for bounding box regression for classifier
def class_loss_regr(num_classes):
    def class_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:, :, 4 * num_classes:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
        return lambda_cls_regr * K.sum(
            y_true[:, :, :4 * num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4 * num_classes])
    return class_loss_regr_fixed_num

# Loss function for classification
def class_loss_cls(y_true, y_pred):
    return lambda_cls_class * K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))
