from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from PIL import Image
import os
import numpy as np

batch_size = 128
num_classes = 10
epochs = 4
full_path = os.path.join('data_tmp', 'teste')

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def treino():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
    model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

def teste():
    model = load_model('my_model.h5')
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def predict(imagens, img_paths):
    
    model = load_model('my_model.h5')
    result = model.predict(imagens)

    for i, p in enumerate(result):
        print(img_paths[i] + ' foi identificada como', end=' ')
        maior = 0.
        indice = 0
        for j, m in enumerate(p):
            # print('j', j)
            # print('m', m)
            if m > maior:
                maior = m
                indice = j
        print(str(indice) + ' com precisao de ' + str(maior))
        print('--------------------------------')

def load_image():
    images_path = sorted(img for img in os.listdir(full_path) if img.endswith('.png'))
    imgs_input = np.empty([len(images_path), img_rows, img_cols, 1], dtype=np.uint8)
    for j, p in enumerate(images_path):
        img = Image.open(os.path.join(full_path, p))
        img = np.asarray(img).reshape(img_rows, img_cols, 1)
        imgs_input[j] = img / 255.

    return images_path, imgs_input

print("Inicia treino")
treino()

print("Inicia teste")
teste()


img_paths, imgs = load_image()

print("predição")
predict(imgs, img_paths)