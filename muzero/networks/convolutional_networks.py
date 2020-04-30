from tensorflow_core.python.keras import Model
from tensorflow_core.python.keras.layers import AveragePooling2D, Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU
from tensorflow_core.python.keras.layers.core import Activation


'''
Residual unit modified and updated to account for being keras 2.0 from https://github.com/relh/keras-residual-unit
'''


def conv_block(feat_maps_out, prev):
    prev = BatchNormalization(axis=-1)(prev)  # Specifying the axis and mode allows for later merging
    prev = Activation('tanh')(prev)
    prev = Conv2D(filters=feat_maps_out, kernel_size=3, padding='same')(prev)
    prev = BatchNormalization(axis=-1)(prev)  # Specifying the axis and mode allows for later merging
    prev = Activation('tanh')(prev)
    prev = Conv2D(filters=feat_maps_out, kernel_size=3, padding='same')(prev)
    return prev


def skip_block(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = Conv2D(filters=feat_maps_out, kernel_size=1, padding='same')(prev)
    return prev


def residual(feat_maps_in, feat_maps_out, prev_layer):
    '''
    A customizable residual unit with convolutional and shortcut blocks
    Args:
      feat_maps_in: number of channels/filters coming in, from input or previous layer
      feat_maps_out: how many output channels/filters this block will produce
      prev_layer: the previous layer
    '''

    skip = skip_block(feat_maps_in, feat_maps_out, prev_layer)
    conv = conv_block(feat_maps_out, prev_layer)

    merged = Add()([skip, conv])
    return merged


def build_dynamic_network(shape, filter_size=6, conv_strides=1):
    input = Input(shape)
    c1 = Conv2D(filters=filter_size, kernel_size=3, strides=conv_strides, padding='same', activation='tanh',
                input_shape=shape)(input)
    r1 = residual(filter_size, filter_size, c1)

    #### ADDED #####
    r2 = residual(filter_size, filter_size, r1)

    c2 = Conv2D(filters=filter_size, kernel_size=3, strides=conv_strides, padding='same', activation='tanh',
                input_shape=shape)(r2)

    r3 = residual(filter_size, filter_size, c2)
    r4 = residual(filter_size, filter_size, r3)
    r5 = residual(filter_size, filter_size, r4)
    #### ADDED #####

    model = Model(inputs=input, outputs=r5)
    return model


def build_reward_network(shape, filter_size1=3, filter_size2=1):
    input = Input(shape)
    c1 = Conv2D(filters=filter_size1, kernel_size=3, strides=2, padding='same', activation='relu',
                input_shape=shape)(input)
    a1 = AveragePooling2D(strides=2)(c1)
    c2 = Conv2D(filters=filter_size2, kernel_size=3, strides=1, padding='same', activation='relu',
                input_shape=shape)(a1)
    a2 = AveragePooling2D(strides=2)(c2)
    f1 = Flatten()(a2)

    model = Model(inputs=input, outputs=f1)
    return model


def build_policy_network(shape, action_size, regularizer):
    policy_input = Input(shape)
    c1 = Conv2D(filters=1, kernel_size=1, padding='same', activation='linear',
                kernel_regularizer=regularizer)(policy_input)
    b1 = BatchNormalization(axis=-1)(c1)
    l1 = LeakyReLU()(b1)
    f1 = Flatten()(l1)
    d1 = Dense(18, use_bias=True, activation='linear', kernel_regularizer=regularizer)(f1)
    d2 = Dense(256, use_bias=True, activation='linear', kernel_regularizer=regularizer)(d1)
    d3 = Dense(action_size, use_bias=True, kernel_regularizer=regularizer)(d2)
    #d3 = Dense(action_size, use_bias=True, activation='relu', kernel_regularizer=regularizer)(d2)
    #b1 = BatchNormalization(axis=-1)(d3)
    policy_model = Model(inputs=policy_input, outputs=d3)
    return policy_model


def build_value_network(shape, value_support_size):
    value_input = Input(shape)
    c1 = Conv2D(filters=1, kernel_size=1, padding='same', activation='linear')(value_input)
    b1 = BatchNormalization(axis=-1)(c1)
    l1 = LeakyReLU()(b1)
    f1 = Flatten()(l1)
    d2 = Dense(256, use_bias=True, activation='linear')(f1)
    d3 = Dense(20, use_bias=True, activation='linear')(d2)
    l2 = LeakyReLU()(d3)
    d4 = Dense(value_support_size, use_bias=True, activation='linear')(l2)
    value_model = Model(inputs=value_input, outputs=d4)
    return value_model


def build_representation_network(input_shape, filter_size1=3, filter_size2=6, conv_strides=1, avg_pool_strides=2):

    input = Input(input_shape)
    c1 = Conv2D(filters=filter_size1, kernel_size=3, strides=conv_strides, padding='same', activation='tanh',
                input_shape=input_shape)(input)

    r1 = residual(filter_size1, filter_size1, c1)
    r2 = residual(filter_size1, filter_size1, r1)

    c2 = Conv2D(filters=filter_size2, kernel_size=3, strides=conv_strides, padding='same', activation='tanh',
                input_shape=(input_shape[0]/conv_strides, input_shape[0]/conv_strides, 3))(r2)

    r3 = residual(filter_size2, filter_size2, c2)
    r4 = residual(filter_size2, filter_size2, r3)
    r5 = residual(filter_size2, filter_size2, r4)

    a1 = AveragePooling2D(strides=avg_pool_strides)(r5)

    r6 = residual(filter_size2, filter_size2, a1)
    r7 = residual(filter_size2, filter_size2, r6)
    r8 = residual(filter_size2, filter_size2, r7)

    a2 = AveragePooling2D(strides=avg_pool_strides)(r8)

    model = Model(inputs=input, outputs=a2)
    return model
