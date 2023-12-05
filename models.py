#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
* File       : models.py
* Created    : 2023-09-19 10:06:56
* Author     : M0nk3y
* Version    : 1.0
'''

from keras.layers import Conv1D, AvgPool1D, BatchNormalization, Dense, Flatten, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy

# For ASCAD N0
def simple_ascad_n0(input_size, learning_rate, classes):
    # 定义输入shape
    inputs = Input(shape=(input_size, 1))

    # 第一个卷积层 归一化层去掉, 保留池化层
    x = AvgPool1D(2, 2, name="block1_pool1")(inputs)
    x = Flatten(name="flatten_1")(x)

    # 三次全连接
    x = Dense(10, activation='relu', name='fc1')(x)
    x = Dense(10, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='fc3')(x)

    model = Model(inputs, x, name="simple_ascad_n0")
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['acc'])
    return model


def simple_ascad_n50(input_size, learning_rate, classes):
    inputs = Input(shape=(input_size, 1))

    x = AvgPool1D(2, 2, name='block1_pool')(inputs)
    
    x = Conv1D(64, 25, kernel_initializer='he_uniform', activation='selu', padding='same', name='block2_conv')(x)
    x = BatchNormalization()(x)
    x = AvgPool1D(25, 25, name='block2_pool')(x)

    x = Conv1D(128, 3, kernel_initializer='he_uniform', activation='selu', padding='same', name='block3_conv')(x)
    x = BatchNormalization()(x)
    x = AvgPool1D(4, 4, name='block3_pool')(x)

    # flatten
    x = Flatten()(x)

    x = Dense(20, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)
    x = Dense(20, kernel_initializer='he_uniform', activation='selu', name='fc2')(x)
    x = Dense(20, kernel_initializer='he_uniform', activation='selu', name='fc3')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(inputs, x, name='simple_ascad_n50')
    model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])
    return model

def simple_ascad_n100(input_size, learning_rate, classes):
    inputs = Input(shape=(input_size, 1))

    x = AvgPool1D(2, 2, name='block1_pool')(inputs)
    
    x = Conv1D(64, 50, kernel_initializer='he_uniform', activation='selu', padding='same', name='block2_conv')(x)
    x = BatchNormalization()(x)
    x = AvgPool1D(50, 50, name='block2_pool')(x)

    x = Conv1D(128, 3, kernel_initializer='he_uniform', activation='selu', padding='same', name='block3_conv')(x)
    x = BatchNormalization()(x)
    x = AvgPool1D(2, 2, name='block3_pool')(x)

    # flatten
    x = Flatten()(x)

    x = Dense(20, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)
    x = Dense(20, kernel_initializer='he_uniform', activation='selu', name='fc2')(x)
    x = Dense(20, kernel_initializer='he_uniform', activation='selu', name='fc3')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(inputs, x, name='simple_ascad_n100')
    model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])
    return model