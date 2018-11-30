import numpy as np
import random
from cv2 import cv2
import os
import math
from random import shuffle

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
			# cv.imread(path, cv2.IMREAD_GRAYSCALE)
			for l in range (0, threshold):
				images += [cv2.imread(path + x + '/' + nome_das_imagens[l], cv2.IMREAD_GRAYSCALE).reshape([77*71])/255.0]
				labels.append(int(x))
		else:
			for l in range (threshold, len(nome_das_imagens)):
				images += [cv2.imread(path + x + '/' + nome_das_imagens[l], cv2.IMREAD_GRAYSCALE).reshape([77*71])/255.0]
				labels.append(int(x))

	return np.array(images), np.array(labels)

def sigmoid(z):
	return 1.0/(1.0 + np.exp(-z))

# Função para retornar as derivadas da função Sigmóide
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

# def avaliacao(input):
# 	return np.vectorize(sigmoide)(np.dot(input, w) + b)

def backprop(input, output_esperado):
	for image in (input):
		o = avaliacao(image)
		d = o*(1-o)*(o-output_esperado) # retorna um nparray
	return d

# b0 biases (um valor real)
# w0 vetor de pesos
# x são inputs
# y são outputs
def gradient_descent_step(b0, w0, x, y, learning_rate):
    b_grad = np.zeros(b0.shape)
    w_grad = np.zeros(w0.shape)

    loss = 0
    N = len(x)
    # compute gradients
    for i in range(N): # x[i] -> y[i]
        #w_reshapi = np.reshape(w0, (1, 5467))
        #x1 = np.reshape(x[i], (1, 5467))
        yy = np.dot(w0, x[i]) + b0
        yy = sigmoid(yy)
        loss += 1/(2*N) * (yy - y[i])**2

        delta = yy*(1 - yy)*(yy - y[i])

        x[i] = np.reshape(x[i], (len(x[i], 1)))
        delta = np.reshape(delta, (1, len(delta)))
        w_grad += np.dot(x[i], delta)

        x[i] = np.reshape(x[i], (len(x[i])))
        delta = np.reshape(delta, len(delta[0]))
        #for j in range(5467):
        #	w_grad[j] += x[i][j]*delta
        b_grad += delta
    
    # update parameters
    b_grad /= N
    w_grad /= N
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

def main():
    random.seed(1)
    np.random.seed(1)
    learning_rate = 0.1
    b = np.zeros(10)
    w = np.random.uniform(-1, 1, (5467, 10))
    num_iterations = 10 #alterar para 1000

    path = "../data_part1/train"
    imgs_train, l_train = read_data(path, True)
    imgs_validation, l_validation = read_data(path, False)

    print ("Starting gradient descent at b = {}, w = 0, error = 0".format(b))
    
    print ("Running...")
    b, w, loss = gradient_descent_runner(imgs_train, l_train, b, learning_rate, num_iterations)
    print ("After {0} iterarions b = {1}, w = {2}, error = {3}".format(num_iterations, b, w, loss))
    
    # print("Result: {}".format(validate(w, b, imgs_validation[0])))

    out_esp = [0,0,1,0,0,0,0,0,0, 0]
    test = backprop(imgs_train[0], out_esp)
    print(test)


if __name__ == '__main__':
    main()