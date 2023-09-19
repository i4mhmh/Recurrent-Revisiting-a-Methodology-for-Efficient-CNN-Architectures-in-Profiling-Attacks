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


    