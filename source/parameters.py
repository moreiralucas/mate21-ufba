import tensorflow as tf
import numpy as np
import os

class Parameters:
    def __init__(self):
        # Image
        self.IMAGE_HEIGHT = 77  # height of the image / rows
        self.IMAGE_WIDTH = 71   # width of the image / cols
        self.NUM_CHANNELS = 1   # number of channels of the image
        # Database
        self.TRAIN_FOLDER = os.path.join('data_part1','train') # folder with training images
        self.TEST_FOLDER = os.path.join('data_part1', 'test')   # folder with testing images
        self.SPLIT_RATE = 0.80        # split rate for training and validation sets
        # Training loop
        self.LOG_DIR = 'modelos'
        self.NUM_EPOCHS_FULL = 1001
        self.S_LEARNING_RATE_FULL = 0.001
        self.F_LEARNING_RATE_FULL = 0.001
        self.BATCH_SIZE = 64
        self.TOLERANCE = 3
        # Others parameters
        # here ...
