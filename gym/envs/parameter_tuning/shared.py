"""
This defines code reused in parameter tuning environments
"""

from __future__ import print_function
import gym
import random
from gym import spaces
import numpy as np
from keras.datasets import cifar10, mnist, cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.regularizers import WeightRegularizer
from keras import backend as K


def get_model(par, nb_classes):
    reg = WeightRegularizer()

    # a hack to make regularization variable
    reg.l1 = K.variable(0.0)
    reg.l2 = K.variable(0.0)

    # input square image dimensions
    img_rows, img_cols = X.shape[-1], X.shape[-1]
    img_channels = X.shape[1]
    # save number of classes and instances
    self.nb_classes = nb_classes
    self.nb_inst = len(X)

    # convert class vectors to binary class matrices
    Y = np_utils.to_categorical(Y, nb_classes)
    Yv = np_utils.to_categorical(Yv, nb_classes)

    # here definition of the model happens
    model = Sequential()

    # double true for icnreased probability of conv layers
    if random.choice([True, True, False]):

        # Choose convolution #1
        self.convAsz = random.choice([32, 64, 128])

        model.add(Convolution2D(self.convAsz, 3, 3, border_mode='same',
                                input_shape=(img_channels, img_rows, img_cols),
                                W_regularizer=reg,
                                b_regularizer=reg))
        model.add(Activation('relu'))

        model.add(Convolution2D(self.convAsz, 3, 3,
                                W_regularizer=reg,
                                b_regularizer=reg))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Choose convolution size B (if needed)
        self.convBsz = random.choice([0, 32, 64])

        if self.convBsz > 0:
            model.add(Convolution2D(self.convBsz, 3, 3, border_mode='same',
                                    W_regularizer=reg,
                                    b_regularizer=reg))
            model.add(Activation('relu'))

            model.add(Convolution2D(self.convBsz, 3, 3,
                                    W_regularizer=reg,
                                    b_regularizer=reg))
            model.add(Activation('relu'))

            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

        model.add(Flatten())

    else:
        model.add(Flatten(input_shape=(img_channels, img_rows, img_cols)))
        self.convAsz = 0
        self.convBsz = 0

    # choose fully connected layer size
    self.densesz = random.choice([256, 512, 762])

    model.add(Dense(self.densesz,
                    W_regularizer=reg,
                    b_regularizer=reg))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes,
                    W_regularizer=reg,
                    b_regularizer=reg))
    model.add(Activation('softmax'))

    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    X = X.astype('float32')
    Xv = Xv.astype('float32')
    X /= 255
    Xv /= 255

    self.data = (X, Y, Xv, Yv)
    self.model = model
    self.sgd = sgd

    # initial accuracy values
    self.best_val = 0.0
    self.previous_acc = 0.0

    self.reg = reg
    self.epoch_idx = 0