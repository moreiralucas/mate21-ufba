# ---------------------------------------------------------------------------------------------------------- #
# Author: maups                                                                                              #
# ---------------------------------------------------------------------------------------------------------- #
import tensorflow as tf
import numpy as np
import random
import datetime
import time
import sys
import os

from data import Dataset
from parameters import Parameters
from network import Net


# X_train, y_train, classes_train = d.load_multiclass_dataset(p.TRAIN_FOLDER, p.IMAGE_HEIGHT, p.IMAGE_WIDTH, p.NUM_CHANNELS)

# X_test, X_labels = d.load_images(p.TEST_FOLDER, p.IMAGE_HEIGHT, p.IMAGE_WIDTH, p.NUM_CHANNELS)

# X_train, y_train = d.shuffle(X_train, y_train, seed=42)
# X_train, y_train, X_val, y_val = d.split(X_train, y_train, p.SPLIT_RATE)

# d.set_scales(np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]))
# d.set_angles(np.array([-10, 0, 10]))

# print("Before augmentation: X: {}, Y: {}".format(len(X_train), len(y_train)))
# X_train, y_train = d.augmentation(X_train, y_train) # aumentation no treino
# X_val, y_val = d.augmentation(X_val, y_val) # aumentation no validation
# print("After augmentation: X: {}, Y: {}".format(len(X_train), len(y_train)))

def main():
    p = Parameters()
    d = Dataset()
    d.set_scales(np.linspace(0.7, 1.3, 5))
    d.set_angles(np.linspace(-10, 10, 3))

    train, classes_train = d.load_multiclass_dataset(p.TRAIN_FOLDER, p.IMAGE_HEIGHT, p.IMAGE_WIDTH, p.NUM_CHANNELS)
    train = d.shuffle(train[0], train[1], seed=42)

    train, val = d.split(train[0], train[1], p.SPLIT_RATE)
    train = d.augmentation(train)
    
    n = Net(train, val, p, size_class_train=10)
    n.treino()
    n.prediction(classes_train)


if __name__ == "__main__":
    main()