from keras.layers import Layer
import keras.backend as K
import tensorflow as tf


class RoiPoolingConv(Layer):
    '''ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used.
    
    # Input shape
        List of two 4D tensors [X_img, X_roi] with shape:
        - X_img: `(1, channels, rows, cols)` if `channels_first`, or
          `(1, rows, cols, channels)` if `channels_last`.
        - X_roi: `(1, num_rois, 4)` list of rois, with ordering (x, y, w, h).
    
    # Output shape
        5D tensor with shape:
        `(1, num_rois, pool_size, pool_size, channels)` (or channels_last).
    '''
    def __init__(self, pool_size: int, num_rois: int, **kwargs):
        self.pool_size = pool_size
        self.num_rois = num_rois
        self.data_format = K.image_data_format()
        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        # Get the number of channels based on the image data format
        if self.data_format == 'channels_first':
            self.nb_channels = input_shape[0][1]  # Channels after batch dimension
        else:
            self.nb_channels = input_shape[0][3]  # Channels at the last dimension

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            return None, self.num_rois, self.nb_channels, self.pool_size, self.pool_size
        else:
            return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, inputs):
        assert len(inputs) == 2, 'Expected inputs: [X_img, X_roi]'

        img = inputs[0]  # The feature map of the image
        rois = inputs[1]  # The ROIs

        input_shape = K.shape(img)
        outputs = []

        for roi_idx in range(self.num_rois):
            # Get ROI coordinates (x, y, w, h)
            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            if self.data_format == 'channels_first':
                # Channels-first format: (batch, channels, rows, cols)
                x = K.cast(x, 'int32')
                y = K.cast(y, 'int32')
                w = K.cast(w, 'int32')
                h = K.cast(h, 'int32')

                x_crop = img[:, :, y:y+h, x:x+w]  # Crop the ROI from the image
                resized_crop = tf.image.resize(x_crop, (self.pool_size, self.pool_size))
                outputs.append(resized_crop)

            else:
                # Channels-last format: (batch, rows, cols, channels)
                x = K.cast(x, 'int32')
                y = K.cast(y, 'int32')
                w = K.cast(w, 'int32')
                h = K.cast(h, 'int32')

                x_crop = img[:, y:y+h, x:x+w, :]  # Crop the ROI from the image
                resized_crop = tf.image.resize(x_crop, (self.pool_size, self.pool_size))
                outputs.append(resized_crop)

        # Concatenate all the pooled regions
        final_output = K.stack(outputs, axis=0)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        if self.data_format == 'channels_first':
            # Permute dimensions for channels_first: (batch, num_rois, channels, pool_size, pool_size)
            final_output = K.permute_dimensions(final_output, (0, 1, 4, 2, 3))
        else:
            # Channels_last format (no change needed): (batch, num_rois, pool_size, pool_size, channels)
            final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output
