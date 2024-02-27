# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K

from keras.optimizers import RMSprop, SGD, Adam
from keras.layers import Input, Dense, Activation, Dropout, ZeroPadding2D, \
    Reshape, Flatten, Embedding, \
    Conv2D, Conv1D,GlobalMaxPooling2D, GlobalMaxPooling1D, \
    MaxPooling2D, MaxPooling1D, AveragePooling2D, GlobalAveragePooling1D,\
    LocallyConnected1D, AveragePooling1D, UpSampling2D,\
    BatchNormalization, Lambda, Layer, Conv2DTranspose, \
    LSTM, GRU, TimeDistributed, SimpleRNN, ConvLSTM2D, \
    Permute, RepeatVector, Cropping1D, Cropping2D, Add, \
    SeparableConv2D, LocallyConnected2D, Multiply, Concatenate, \
    SeparableConv1D,  GlobalAveragePooling2D, \
    ZeroPadding1D

from keras.models import Model, Sequential
from keras.utils import plot_model,np_utils
from keras.layers.wrappers import Bidirectional
from tensorflow.python.keras.layers import MultiHeadAttention,LayerNormalization
# import tensorflow_addons as tfa
# from tensorflow_addons.activations import gelu


random_state_tf=2100

m_dl = "mtry7"

opt_name = "adam"
lr_init = 0.1#0.001
beta_1 = 0.8#0.8,0.0001
beta_2 = 0.999#0.999,0.01
sgd_decay = 0.01#0.1/0.01/0.001
num_heads = 2
head_dim = 8
num_blocks = 2
num_classes = 2
k_init = 'he_uniform'
show_model = True

save_dir = "./data/mtry7/"
conf_dl = "-" + m_dl
ft_num = 516#516

global model_showed
model_showed = False
def create_dlmodel(m_name=None):
    np.random.seed(random_state_tf)
    #        set_random_seed(random_state_tf)
    # tf.set_random_seed(random_state_tf)

    # ==========================================
    if m_name is None:
        m_name = m_dl

    if opt_name == "adam":
        optimizer = Adam(lr=lr_init,beta_1=beta_1,beta_2=beta_2)  # 0.0002/0.5 0.001/0.75/0.9
    elif opt_name == "adadelta":
        optimizer = keras.optimizers.Adadelta()
    # else:
    #     optimizer = SGD(lr=lr_init, momentum=0.9, decay=sgd_decay, nesterov=True)
    elif opt_name == "sgd":
        optimizer = SGD(lr=lr_init, momentum=0.9, decay=sgd_decay, nesterov=True)
    else:
        raise ValueError("Unknown optimizer name: {}".format(opt_name))

    losses_used = "categorical"
    metrics = ['accuracy']
    if m_name == "mtry1":
        units = 120
        drop = 0.5

        x_in = Input(shape=(ft_num, 1))

        x = LocallyConnected1D(15, 32, padding="valid", strides=8)(x_in)#15,32,8
        x = Reshape((-1, 1))(x)
        x = LocallyConnected1D(9, 15, padding="valid", strides=3)(x)#9,15,3
        x = Flatten()(x)
        x0 = x
        x01 = Dense(units, activation="tanh")(x0)
        x02 = Dense(units,activation="sigmoid")(x0)
        xa = Dropout(drop)(x01)
        xb = Dropout(drop)(x02)
        x = Add()([xa, xb])
        prediction = Dense(num_classes,
                           kernel_initializer=k_init,
                           activation="softmax")(x)
        model = Model(x_in, prediction)
        # ==========================================
    #===========================================
    if m_name == "mtry2":
        units = 120
        drop = 0.5
        x_in = Input(shape=(ft_num,1))
        x = LSTM(128, activation='tanh', return_sequences=True)(x_in)
        x = Dropout(rate=drop)(x)
        x = LSTM(64, activation='sigmoid')(x)#sigmoid
        x = Dropout(rate=drop)(x)
        x = Dense(units)(x)

        prediction = Dense(num_classes,
                           kernel_initializer=k_init,
                           activation="softmax")(x)
        model = Model(x_in, prediction)

        #===========================================

    # ==========================================
    if m_name == "mtry3":
        units = 96
        drop = 0.5

        x_in = Input(shape=(ft_num, 1))

        x = Dense(256,activation="relu")(x_in)
        x = Dense(128,activation="relu")(x)

        x = Flatten()(x)
        x0 = x
        x01 = Dense(units, activation="tanh")(x0)
        x02 = Dense(units, activation="sigmoid")(x0)
        xa = Dropout(drop)(x01)
        xb = Dropout(drop)(x02)
        x = Add()([xa, xb])
        prediction = Dense(num_classes,
                           kernel_initializer=k_init,
                           activation="softmax")(x)
        model = Model(x_in, prediction)
        # ==========================================
    if m_name == "mtry4":
        units = 120
        drop=0.5

        x_in = Input(shape=(ft_num, 1))

        x = LocallyConnected1D(15, 32, padding="valid", strides=8)(x_in)
        x = Reshape((-1, 1))(x)
        x = LocallyConnected1D(9, 15, padding="valid", strides=3)(x)
        x = Reshape((-1, 1))(x)
        x = LocallyConnected1D(6, 15, padding="valid", strides=6)(x)
        x = Flatten()(x)
        x0 = x
        x01 = Dense(units, activation="tanh")(x0)
        x02 = Dense(units,activation="sigmoid")(x0)
        xa = Dropout(drop)(x01)
        xb = Dropout(drop)(x02)
        x = Add()([xa, xb])
        prediction = Dense(num_classes,
                           kernel_initializer=k_init,
                           activation="softmax")(x)
        model = Model(x_in, prediction)
        #==================================================================
    if m_name == "mtry5":
        units = 120
        drop=0.5

        x_in = Input(shape=(ft_num, 1))

        x = LocallyConnected1D(15, 32, padding="valid", strides=8)(x_in)

        x = Flatten()(x)
        x0 = x
        x01 = Dense(units, activation="tanh")(x0)
        x02 = Dense(units,activation="sigmoid")(x0)
        xa = Dropout(drop)(x01)
        xb = Dropout(drop)(x02)
        x = Add()([xa, xb])
        prediction = Dense(num_classes,
                           kernel_initializer=k_init,
                           activation="softmax")(x)
        model = Model(x_in, prediction)
        #=================================================

        #=======================================

    def residual_block(x, filters, kernel_size=3, activation='relu'):

        # x = LayerNormalization(epsilon=1e-6)(x)
        # x = Conv1D(filters, kernel_size, padding='same')(x)
        x = Dense(filters)(x)
        x = Dropout(rate=0.5)(x)
        x = Dense(filters)(x)
        x = Activation(activation)(x)
        return x

    if m_name == "mtry7":
        units = 120
        drop = 0.5

        x_in = Input(shape=(ft_num, 1))
        # x = LocallyConnected1D(15, 16, padding="valid", strides=3)(x_in)  # 9,15,3
        # x = Reshape((-1, 1))(x)
        # x = LocallyConnected1D(9, 32, padding="valid", strides=8)(x)  # 15,32,8

        x = LocallyConnected1D(15, 3, padding="valid", strides=3)(x_in)  # 15,32,8
        x = Reshape((-1, 1))(x)
        x = LocallyConnected1D(9, 2, padding="valid", strides=2)(x)  # 9,15,3

        x = MultiHeadAttention(num_heads=num_heads, key_dim=head_dim)(query=x, value=x, key=x)

        # for _ in range(num_blocks):
        #     x = MultiHeadAttention(num_heads=num_heads, key_dim=head_dim)(query=x, value=x, key=x)
        #     x = Dropout(rate=0.5)(x)
        #     x = residual_block(x, 32)
        x = Flatten()(x)
        # x = GlobalAveragePooling1D()(x)
        x0 = x
        x01 = Dense(units, activation="tanh",kernel_regularizer=tf.keras.regularizers.l2(0.01))(x0)#

        x02 = Dense(units, activation="sigmoid",kernel_regularizer=tf.keras.regularizers.l2(0.01))(x0)#
        xa = Dropout(drop)(x01)
        xb = Dropout(drop)(x02)
        x = Add()([xa, xb])

        prediction = Dense(num_classes,
                           kernel_initializer=k_init,
                           activation="softmax")(x)
        model = Model(inputs=x_in, outputs=prediction)



    global model_showed
    if show_model and not model_showed:
        model.summary()
        plot_model(model, to_file=save_dir + "/model-%s.png" % conf_dl,
                    show_layer_names=True, show_shapes=True)
        model_showed = True

    if losses_used == "sparse":
        model.compile(optimizer=optimizer,
                      loss=keras.losses.sparse_categorical_crossentropy,

                      metrics=metrics)
    if losses_used == "categorical":
        model.compile(optimizer=optimizer,
                          #                          loss=xcustom_loss_batch,
                      loss='categorical_crossentropy',
                      metrics=metrics)
    return model
