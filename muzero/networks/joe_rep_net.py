from tensorflow_core.python.keras.models import Model
from tensorflow_core.python.keras.optimizers import Nadam
from tensorflow_core.python.keras.layers import AveragePooling2D, BatchNormalization, Conv2D, Input, merge
from tensorflow_core.python.keras.layers.core import Activation, Layer
from tensorflow_core.python.keras.utils.visualize_util import plot

'''
Keras Customizable Residual Unit
This is a simplified implementation of the basic (no bottlenecks) full pre-activation residual unit from He, K., Zhang, 
X., Ren, S., Sun, J., "Identity Mappings in Deep Residual Networks" (http://arxiv.org/abs/1603.05027v2).
'''


def conv_block(feat_maps_out, prev):
    prev = BatchNormalization(axis=1, mode=2)(prev)  # Specifying the axis and mode allows for later merging
    prev = Activation('relu')(prev)
    prev = Conv2D(feat_maps_out, 3, 3, border_mode='same')(prev)
    prev = BatchNormalization(axis=1, mode=2)(prev)  # Specifying the axis and mode allows for later merging
    prev = Activation('relu')(prev)
    prev = Conv2D(feat_maps_out, 3, 3, border_mode='same')(prev)
    return prev


def skip_block(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = Conv2D(feat_maps_out, 1, 1, border_mode='same')(prev)
    return prev 


def Residual(feat_maps_in, feat_maps_out, prev_layer):
    '''
    A customizable residual unit with convolutional and shortcut blocks
    Args:
      feat_maps_in: number of channels/filters coming in, from input or previous layer
      feat_maps_out: how many output channels/filters this block will produce
      prev_layer: the previous layer
    '''

    skip = skip_block(feat_maps_in, feat_maps_out, prev_layer)
    conv = conv_block(feat_maps_out, prev_layer)

    print('Residual block mapping '+str(feat_maps_in)+' channels to '+str(feat_maps_out)+' channels built')
    return merge([skip, conv], mode='sum')  # the residual connection


if __name__ == "__main__":
    img_rows = 42
    img_cols = 32

    inp = Input((3, img_rows, img_cols))
    conv1 = Conv2D(filters=3, kernel_size=(3,3), strides=(2,2), padding='same',
                         activation='relu', border_mode='same')(inp)
    r1 = Residual(3, 3, conv1)
    r2 = Residual(3, 3, r1)
    conv2 = Conv2D(filters=6, kernel_size=(3, 3), strides=(2, 2), padding='same',
                         activation='relu', border_mode='same')(r2)

    r3 = Residual(6, 6, conv2)
    r4 = Residual(6, 6, r3)

    avg1 = AveragePooling2D(pool_size=(2,2))(r4)

    r5 = Residual(6, 6, avg1)
    r6 = Residual(6, 6, r5)
    r7 = Residual(6, 6, r6)

    avg2 = AveragePooling2D(pool_size=(2, 2))(r7)

    model = Model(input=inp, output=avg2)
    model.compile(optimizer=Nadam(lr=1e-5), loss='mean_squared_error')

    plot(model, to_file='./toy_model.png', show_shapes=True)