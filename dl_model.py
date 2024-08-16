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
from tensorflow.python.keras.layers import MultiHeadAttention, LayerNormalization, LeakyReLU
from resnet import ResNet50
# import tensorflow_addons as tfa
# from tensorflow_addons.activations import gelu


random_state_tf=2100

m_dl = "mtry6"
# 调试
opt_name = "adam"
sgd_decay = 0.005#0.1/0.01/0.001
lr_init =0.001#0.001.
beta_1 = 0.8#0.8,0.0001
beta_2 = 0.99#0.999,0.01
num_heads = 4#8
head_dim = 32
num_blocks = 2
num_classes = 2
k_init = 'he_uniform'#
show_model = True

save_dir = "./dataset/"
conf_dl = "-" + m_dl
ft_num = 518
img_rows, img_cols = 32, 32

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

        x = LocallyConnected1D(16, 32, padding="valid", strides=3)(x_in)#15,32,8
        x = Reshape((-1, 1))(x)
        x = LocallyConnected1D(32, 64, padding="valid", strides=5)(x)#9,15,3
        x = Flatten()(x)
        x0 = x
        x01 = Dense(units, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(0.1))(x0)
        x02 = Dense(units,activation="tanh",kernel_regularizer=tf.keras.regularizers.l2(0.1))(x0)
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
        x = LSTM(128, activation='relu', return_sequences=True)(x_in)
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

        x = LocallyConnected1D(15, 64, padding="valid", strides=10)(x_in)#15,32,8
        x = Reshape((-1, 1))(x)
        x = LocallyConnected1D(9, 32, padding="valid", strides=8)(x)#9,15,3
        x = Reshape((-1, 1))(x)
        x = LocallyConnected1D(6, 16, padding="valid", strides=5)(x)#6,15,6
        x = Flatten()(x)
        x0 = x
        x01 = Dense(units, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(0.1))(x0)
        x02 = Dense(units,activation="tanh",kernel_regularizer=tf.keras.regularizers.l2(0.1))(x0)
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

        x = LocallyConnected1D(16, 32, padding="valid", strides=5)(x_in)

        x = Flatten()(x)
        x0 = x
        x01 = Dense(units, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(0.1))(x0)
        x02 = Dense(units,activation="tanh",kernel_regularizer=tf.keras.regularizers.l2(0.1))(x0)
        xa = Dropout(drop)(x01)
        xb = Dropout(drop)(x02)
        x = Add()([xa, xb])
        prediction = Dense(num_classes,
                           kernel_initializer=k_init,
                           activation="softmax")(x)
        model = Model(x_in, prediction)
        #=================================================

    if m_name == "mtry6":
        units = 120
        drop = 0.5

        x_in = Input(shape=(ft_num, 1))
        x = LocallyConnected1D(16, 32, padding="valid", strides=3)(x_in)  # 9,15,3
        x = Reshape((-1, 1))(x)
        x = LocallyConnected1D(32, 64, padding="valid", strides=5)(x)  # 15,32,8


        x0 = x
        x01 = Dense(units, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.1))(x0)
        x02 = Dense(units, activation="tanh", kernel_regularizer=tf.keras.regularizers.l2(0.1))(x0)
        xa = Dropout(drop)(x01)
        xb = Dropout(drop)(x02)
        x = Add()([xa, xb])
        x = MultiHeadAttention(num_heads=num_heads, key_dim=head_dim)(query=x, value=x, key=x)
        x = Flatten()(x)
        # x = Dense(32, activation='sigmoid')(x)
        # x = Dropout(drop)(x)
        prediction = Dense(num_classes,
                            kernel_initializer=k_init,
                            activation="softmax")(x)
        model = Model(inputs=x_in, outputs=prediction)
        #===========================================================

    if m_name == "resnet":

        x_in = Input(shape=(ft_num, 1))

        x = ResNet50(weights=None, include_top=False)(x_in)
        x = GlobalAveragePooling1D()(x)
        x = Dense(1024, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.1))(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=x_in,outputs=predictions)

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