from keras.engine import Layer, InputSpec
from keras import initializers, regularizers
from keras import backend as K


class FixedBatchNormalization(Layer):

    def __init__(self, epsilon: float = 1e-3, axis: int = -1,
                 weights=None, beta_init: str = 'zeros', gamma_init: str = 'ones',
                 gamma_regularizer=None, beta_regularizer=None, **kwargs):
        self.supports_masking = True
        self.beta_initializer = initializers.get(beta_init)
        self.gamma_initializer = initializers.get(gamma_init)
        self.epsilon = epsilon
        self.axis = axis
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.initial_weights = weights
        super(FixedBatchNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        # Ensure the layer has an input shape to work with
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)

        # Adding weights for gamma and beta (not trainable)
        self.gamma = self.add_weight(name='{}_gamma'.format(self.name),
                                     shape=shape,
                                     initializer=self.gamma_initializer,
                                     regularizer=self.gamma_regularizer,
                                     trainable=False)

        self.beta = self.add_weight(name='{}_beta'.format(self.name),
                                    shape=shape,
                                    initializer=self.beta_initializer,
                                    regularizer=self.beta_regularizer,
                                    trainable=False)

        # Running mean and std, used for inference, are not trainable
        self.running_mean = self.add_weight(name='{}_running_mean'.format(self.name),
                                            shape=shape,
                                            initializer='zeros',
                                            trainable=False)

        self.running_std = self.add_weight(name='{}_running_std'.format(self.name),
                                           shape=shape,
                                           initializer='ones',
                                           trainable=False)

        # If initial weights are provided, set them
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.built = True

    def call(self, inputs, mask=None):
        # Assert that the layer has been built
        assert self.built, 'Layer must be built before being called'

        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        if sorted(reduction_axes) == list(range(K.ndim(inputs))[:-1]):
            # If the axis reduction is standard, no need for broadcasting
            output = K.batch_normalization(
                inputs, self.running_mean, self.running_std,
                self.beta, self.gamma,
                epsilon=self.epsilon)
        else:
            # Broadcasting is needed in this case
            broadcast_mean = K.reshape(self.running_mean, broadcast_shape)
            broadcast_std = K.reshape(self.running_std, broadcast_shape)
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            output = K.batch_normalization(
                inputs, broadcast_mean, broadcast_std,
                broadcast_beta, broadcast_gamma,
                epsilon=self.epsilon, axis=0)

        return output

    def get_config(self):
        config = {
            'epsilon': self.epsilon,
            'axis': self.axis,
            'gamma_regularizer': self.gamma_regularizer.get_config() if self.gamma_regularizer else None,
            'beta_regularizer': self.beta_regularizer.get_config() if self.beta_regularizer else None,
        }
        base_config = super(FixedBatchNormalization, self).get_config()
        return {**base_config, **config}
