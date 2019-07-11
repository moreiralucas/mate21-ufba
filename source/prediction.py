import tensorflow as tf
import numpy as np

from data_PIL import Dataset
from parameters import Parameters
from network import Net

def main():
    # Inicializa e configura parâmetros
    p = Parameters()
    d = Dataset()

    # Inicializa a rede
    n = Net(p, size_class_train=10)

    # Carrega imagens de teste para o prediction
    test = d.load_images(p.TEST_FOLDER, p.IMAGE_HEIGHT, p.IMAGE_WIDTH, p.NUM_CHANNELS)

    print ("Iniciou o prediction.")
    # Realiza a predição das imagens de teste
    n.prediction2(test)
    print ("Finish him!")

if __name__ == "__main__":
    main()