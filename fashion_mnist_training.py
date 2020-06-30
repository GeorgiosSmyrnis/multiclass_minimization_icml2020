#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd

from keras import layers, callbacks
from keras.utils import to_categorical
from keras.datasets import fashion_mnist
from keras.models import Model

if __name__ == '__main__':
    """
    Train a model for the Fashion-MNIST dataset, using the architecture defined
    in the paper.
    """
    if(len(sys.argv) != 2):
        print('Usage: python mnist_training.py <filename>')

    filename = sys.argv[1]

    (train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

    # Retrieve dataset
    train_data = train_data.reshape(train_data.shape[0],28, 28, 1).astype('float32')/255
    test_data = test_data.reshape(test_data.shape[0], 28, 28, 1).astype('float32')/255
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # Permute data
    np.random.seed()
    idx = np.arange(train_data.shape[0])
    np.random.shuffle(idx)
    train_data = train_data[idx, :, :, :]
    train_labels = train_labels[idx]

    # Define model for training
    input = layers.Input(shape=(28,28,1))
    conv1 = layers.Conv2D(32, (5,5), activation='relu')(input)
    pool1 = layers.MaxPooling2D(3)(conv1)
    conv2 = layers.Conv2D(32, (5,5), activation='relu')(pool1)
    pool2 = layers.MaxPooling2D(3)(conv2)
    dense_in = layers.Flatten()(pool2)
    dense1 = layers.Dense(1000, activation='relu')(dense_in)
    dense2 = layers.Dense(10, activation='softmax')(dense1)

    model = Model(inputs = input, outputs = dense2)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    callback_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]

    # Train and save model
    model.fit(
        train_data, train_labels, epochs = 50, verbose=1,
        validation_split=0.2,
        batch_size=128, callbacks=callback_list
    )

    model.save(filename)
