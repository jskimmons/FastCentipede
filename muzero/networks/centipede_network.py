import math

import numpy as np
from tensorflow_core.python.keras import regularizers
from tensorflow_core.python.keras.layers.core import Dense
from tensorflow_core.python.keras.models import Sequential, model_from_json

from game.game import Action
from networks.network import BaseNetwork


class CentipedeNetwork(BaseNetwork):

    def __init__(self,
                 state_size: int,  # TODO: whatami, cur 4
                 action_size: int,
                 representation_size: int,  # TODO: whatami, cur 4
                 max_value: int,
                 hidden_neurons: int = 64,
                 weight_decay: float = 1e-4,
                 representation_activation: str = 'tanh',
                 directory: str = None):
        self.state_size = state_size
        self.action_size = action_size
        self.value_support_size = math.ceil(math.sqrt(max_value)) + 1

        if directory is not None:
            print("Loading network from " + directory)
            representation_network = self.load_model(directory + "/representation")
            value_network = self.load_model(directory + "/value")
            policy_network = self.load_model(directory + "/policy")
            dynamic_network = self.load_model(directory + "/dynamic")
            reward_network = self.load_model(directory + "/reward")
        else:
            print("Creating new network")
            regularizer = regularizers.l2(weight_decay)

            # TODO: determine and set input sizes so model can be saved
            in1 = (None, 4)
            in2 = (None, 6)

            # TODO: change me
            representation_network = Sequential([
                Dense(hidden_neurons, activation='relu', kernel_regularizer=regularizer),
                Dense(representation_size, activation=representation_activation, kernel_regularizer=regularizer)
            ])
            value_network = Sequential([
                Dense(hidden_neurons, activation='relu', kernel_regularizer=regularizer),
                Dense(self.value_support_size, kernel_regularizer=regularizer)
            ])
            policy_network = Sequential([
                Dense(hidden_neurons, activation='relu', kernel_regularizer=regularizer),
                Dense(action_size, kernel_regularizer=regularizer)
            ])

            # TODO: change me
            dynamic_network = Sequential([
                Dense(hidden_neurons, activation='relu', kernel_regularizer=regularizer),
                Dense(representation_size, activation=representation_activation, kernel_regularizer=regularizer)
            ])
            reward_network = Sequential([
                Dense(16, activation='relu', kernel_regularizer=regularizer),
                Dense(1, kernel_regularizer=regularizer)
            ])

        super().__init__(representation_network, value_network, policy_network, dynamic_network, reward_network)

    def _value_transform(self, value_support: np.array) -> float:
        """
        The value is obtained by first computing the expected value from the discrete support.
        Second, the inverse transform is then apply (the square function).
        """

        value = self._softmax(value_support)
        value = np.dot(value, range(self.value_support_size))
        value = np.asscalar(value) ** 2
        return value

    def _reward_transform(self, reward: np.array) -> float:
        return np.asscalar(reward)

    def _conditioned_hidden_state(self, hidden_state: np.array, action: Action) -> np.array:
        conditioned_hidden = np.concatenate((hidden_state, np.eye(self.action_size)[action.index]))
        return np.expand_dims(conditioned_hidden, axis=0)

    def _softmax(self, values):
        """Compute softmax using numerical stability tricks."""
        values_exp = np.exp(values - np.max(values))
        return values_exp / np.sum(values_exp)




##### JOE CODE HERE ####
import math

import numpy as np
from tensorflow_core.python.keras import regularizers
from tensorflow_core.python.keras import layers
from tensorflow_core.python.keras.models import Sequential

def residual_block(y, nb_channels, _strides=(2, 2), _project_shortcut=False):
    shortcut = y

    # down-sampling is performed with a stride of 2
    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)

    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)
    y = layers.BatchNormalization()(y)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = layers.Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    y = layers.add([shortcut, y])
    y = layers.LeakyReLU()(y)

    return y

# 210 160 3
# 42 32 3
# input 96x96x128, 1 conv w stride 2, 128 output planes
representation_network = Sequential([
    Conv2D(3, (3, 3), strides=(2, 2), padding='same', input_shape=(96, 96, 3)),
    residual_block(),
    residual_block(),
    Conv2D(),
    residual_block(),
    residual_block(),
    residual_block(),

])

class Residual_CNN(Gen_Model):
    def __init__(self, reg_const, learning_rate, input_dim, output_dim, hidden_layers):
        Gen_Model.__init__(self, reg_const, learning_rate, input_dim, output_dim)
        self.hidden_layers = hidden_layers
        self.num_layers = len(hidden_layers)
        self.model = self._build_model()

    def residual_layer(self, input_block, filters, kernel_size):

        x = self.conv_layer(input_block, filters, kernel_size)

        x = Conv2D(
            filters=filters
            , kernel_size=kernel_size
            , data_format="channels_first"
            , padding='same'
            , use_bias=False
            , activation='linear'
            , kernel_regularizer=regularizers.l2(self.reg_const)
        )(x)

        x = BatchNormalization(axis=1)(x)

        x = add([input_block, x])

        x = LeakyReLU()(x)

        return (x)

    def conv_layer(self, x, filters, kernel_size):

        x = Conv2D(
            filters=filters
            , kernel_size=kernel_size
            , data_format="channels_first"
            , padding='same'
            , use_bias=False
            , activation='linear'
            , kernel_regularizer=regularizers.l2(self.reg_const)
        )(x)

        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)

        return (x)

    def _build_model(self):

        main_input = Input(shape=self.input_dim, name='main_input')

        x = self.conv_layer(main_input, self.hidden_layers[0]['filters'], self.hidden_layers[0]['kernel_size'])

        if len(self.hidden_layers) > 1:
            for h in self.hidden_layers[1:]:
                x = self.residual_layer(x, h['filters'], h['kernel_size'])

        model = Model(inputs=[main_input], outputs=self.hidden_layers[-1])
        model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': softmax_cross_entropy_with_logits},
                      optimizer=SGD(lr=self.learning_rate, momentum=config.MOMENTUM),
                      loss_weights={'value_head': 0.5, 'policy_head': 0.5}
                      )

        return model