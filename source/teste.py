import tensorflow as tf
import numpy as np
import random
import cv2

# import sys
# import os
# imagePath = 'primos.jpg' # folder with training images
#imagePath = '../data_part1/train/3/10.png' # folder with training images

#image = cv2.imread(imagePath)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


#cols = image.shape[1]
#rows = image.shape[0]
#print("cols: {}".format(cols))
#print("rows: {}".format(rows))
# angle_degrees = 20
# scale = 1.5

# -------------
# -------------

#for scale in np.linspace(0.7, 1.5, num=3):
#    for angle in np.linspace(-40, 40, num=3):
#        print (angle)
#        print (scale)
#        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
#        out = cv2.warpAffine(image, M, (cols, rows))
#        cv2.imshow('image',out)
#        cv2.waitKey(0)

# cv2.destroyAllWindows()

from data import Dataset
from parameters import Parameters
from random import randint

TRAIN_FOLDER = '../data_part1/train' # folder with training images
TEST_FOLDER = '../data_part1/test'   # folder with testing images
SPLIT_RATE = 0.90        # split rate for training and validation sets

p = Parameters()
d = Dataset()

X_train, y_train, classes_train = d.load_multiclass_dataset(TRAIN_FOLDER, p.IMAGE_HEIGHT, p.IMAGE_WIDTH, p.NUM_CHANNELS)
X_train = X_train/255.0

X_train, y_train = d.shuffle(X_train, y_train, seed=42)
X_train, y_train, X_val, y_val = d.split(X_train, y_train, SPLIT_RATE)

print('shapes: {}'.format(X_train[0].shape))
print('dtype: {}'.format(X_train[0].dtype))
print('-------- d.augmentation ----------------')
print ('len(X_train): {}, len(y_train): {}'.format(len(X_train), len(y_train)))

d.set_angles()
d.set_scales()
X_train, y_train = d.augmentation(X_train, y_train)

print('-------- acabou o augmentation ----------------')
print('shapes: {}'.format(X_train[0].shape))
print('dtype: {}'.format(X_train[0].dtype))
print ('len(X_train): {}, len(y_train): {}'.format(len(X_train), len(y_train)))

cont = 0
while True:
    idx = randint(0, len(X_train) - 1)
    print('----++++----++++----++++----++++----++++----++++')
    print("label: {}".format(y_train[idx]))
    print("idx: {}".format(idx))
    print(np.max(X_train[idx]))
    print('----++++----++++----++++----++++----++++----++++')
    cv2.imshow('image', X_train[idx])
    cv2.waitKey(0)
    if cont > 20:
        break
    cont += 1
cv2.destroyAllWindows()
