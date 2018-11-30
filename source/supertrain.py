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

AUGMENTATION = False
d = Dataset()

def normaliza255(input):
    novo = input[0] / 255.0
    return novo, input[1]

def load_images(p):
    # Carrega as imagens do treino com suas respectivas labels
    x_train_orig, y_train_orig, classes_train = d.load_multiclass_dataset(p.TRAIN_FOLDER, p.IMAGE_HEIGHT, p.IMAGE_WIDTH, p.NUM_CHANNELS)
    x_train_fake, y_train_fake, classes_train = d.load_multiclass_dataset(p.FAKE_TRAIN_FOLDER, p.IMAGE_HEIGHT, p.IMAGE_WIDTH, p.NUM_CHANNELS)
    
    # Carrega a nova base
    #x_new_train, y_new_train, classes_train = d.load_multiclass_dataset(p.NEW_TRAIN_FOLDER, p.IMAGE_HEIGHT, p.IMAGE_WIDTH, p.NUM_CHANNELS)

    # Embaralhas as imagens com seus respectivos labels
    x_train_orig, x_train_orig = d.shuffle(x_train_orig, y_train_orig, seed=42)
    train_fake = d.shuffle(train_fake[0], train_fake[1], seed=42)
    
    print("Train: ", end=' ')
    print(len(train_orig))
    
    train_orig, val = d.split(train_orig[0], train_orig[1], p.SPLIT_RATE)

    if AUGMENTATION:
        train_orig = d.augmentation(train_orig)
        #train_orig.extend(d.augmentation(train_fake))
        #train_orig.extend(d.augmentation(new_train))

        val = d.augmentation(val)
    else:
        train_orig = normaliza255(train_orig)
        #train_orig.extend(normaliza255(train_fake))
        #train_orig.extend(normaliza255(new_train))

        val = normaliza255(val)
    
    print("Train_modificado: ", end=' ')
    print(len(train_orig))

    return train_orig, val

def main():
    # Inicializa e configura parâmetros
    p = Parameters()
    d.set_scales(np.linspace(0.7, 1.3, 5))
    d.set_angles(np.linspace(-10, 10, 3))
    AUGMENTATION = False

    # A função augmentation realiza a normalização da imagem (Divide por 255)
    # train = d.augmentation(train)
    # val = d.augmentation(val)

    # Inicializa a rede
    n = Net(p)
    # Inicia treino
    
    # Carrega imagens de teste para o prediction
    train, val = load_images(p)

    print("Carregou as imagens")
    #n.treino(train, val)
    print ("Iniciou o prediction.")

    # Realiza a predição das imagens de teste
    #n.prediction2(test)
    print ("Finish him!")

if __name__ == "__main__":
    main()