import tensorflow as tf
import numpy as np
import random
import datetime
import time
import sys
import os

from data_PIL import Dataset
#from data import Dataset
from parameters import Parameters
from network import Net

def main():
    # Inicializa e configura parâmetros
    p = Parameters()
    d = Dataset()
    # d.set_scales(np.array([1.]))
    # d.set_scales(np.linspace(0.7, 1.3, 5))
    # d.set_angles(np.linspace(-10, 10, 3))

    # Carrega as imagens do treino com suas respectivas labels
    train, classes_train = d.load_multiclass_dataset(p.TRAIN_FOLDER, p.IMAGE_HEIGHT, p.IMAGE_WIDTH, p.NUM_CHANNELS)
    
    # Embaralhas as imagens com seus respectivos labels
    train = d.shuffle(train[0], train[1], seed=42)

    # Divide a base de treino em duas, de acordo com o SPLIT_RATE
    train, val = d.split(train[0], train[1], p.SPLIT_RATE)

    print("len(train): {}".format(len(train[0])))
    print("len(train): {}".format(len(val[0])))
    # A função augmentation realiza a normalização da imagem (Divide por 255)
    # train = d.augmentation(train)
    # val = d.augmentation(val)

    # Inicializa a rede
    n = Net(p, size_class_train=10)
    # Inicia treino
    n.treino(train, val)


if __name__ == "__main__":
    main()