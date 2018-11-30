import random
import numpy as np
import os
import math
from random import shuffle
from cv2 import cv2

#w = np.random.uniform(-1, 1, [77*71]) # vetor de pesos
#b = np.zeros([10]) # vetor de bias

def read_data(path, is_train=True):
	entrada = os.listdir(path)
	path += '/'
	
	images = []
	labels = []
	for x in entrada:
		nome_das_imagens = os.listdir(path + x)
		threshold = int(len(nome_das_imagens) * 0.7) # define threshold
		shuffle(nome_das_imagens) # embaralha a ordem

		if is_train:
			for l in range (0, threshold):
				images += [cv2.imread(path + x + '/' + nome_das_imagens[l], cv2.IMREAD_GRAYSCALE).reshape([77*71])/255.0]
				labels.append(int(x))
		else:
			for l in range (threshold, len(nome_das_imagens)):
				images += [cv2.imread(path + x + '/' + nome_das_imagens[l], cv2.IMREAD_GRAYSCALE).reshape([77*71])/255.0]
				labels.append(int(x))

	return np.array(images), np.array(labels)

def sigmoide(z):
	return 1.0/(1.0 + np.exp(-z))

# b0 biases (um valor real)
# w0 vetor de pesos
# x são inputs
# y são outputs
def gradient_descent_step(b0, w0, x, y, learning_rate):
    b_grad = 0
    w_grad = np.zeros(len(w0))
    loss = 0
    N = len(x)
    # compute gradients
    for i in range(N): # x[i] -> y[i]
        #w_reshapi = np.reshape(w0, (1, 5467))
        #x1 = np.reshape(x[i], (1, 5467))
        yy = np.dot(w0, x[i]) + b0
        loss += 1/N * (yy - y[i])**2

        b_grad += -(2.0/N) * (yy - y[i])
        w_grad += -(2.0/N) * x[i] * (yy - y[i])

    # update parameters
    b1 = b0 - (learning_rate * b_grad)
    w1 = w0 - (learning_rate * w_grad)
    return b1, w1, loss

def gradient_descent_runner(input, output_esperado, starting_b, learning_rate, num_iterations):
    b = starting_b
    w = np.random.uniform(-1, 1, len(input[0]))
    loss = 0
    for i in range(num_iterations):
        print("Iteração: {}".format(i))
        b, w, loss = gradient_descent_step(b, w, input, output_esperado,learning_rate)
    return b, w, loss

def validate(w, b, image):
    return round(np.dot(w, image) + b)

def main():
    random.seed(1)
    np.random.seed(1)

    learning_rate = 0.1
    initial_b = 0 # initial y-intercept guess
    num_iterations = 10 #alterar para 1000

    path = "../data_part1/train"
    imgs_train, l_train = read_data(path, True)
    imgs_validation, l_validation = read_data(path, False)

    print ("Running...")
    [b, w, loss] = gradient_descent_runner(imgs_train, l_train, initial_b, learning_rate, num_iterations)
    print ("After {0} iterarions b = {1}, w = {2}, error = {3}".format(num_iterations, b, w, loss))
    
    # print("len(l_validation): {}".format(len(l_validation)))
    # print("len(imgs_validation): {}".format(len(imgs_validation)))
    cont = 0
    for label, img in zip(l_validation, imgs_validation):
        # label == resultado da função
        if int(label) == int(validate(w, b, img)):
            print(label)
            cont = cont + 1
        #print("Label {} has detected: {}".format(img[0], int(validate(w, b, img[1]))))
    #print("Total de acertos: {}".format(cont))

if __name__ == '__main__':
    main()