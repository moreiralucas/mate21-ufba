from random import shuffle
import numpy as np
import os
from cv2 import cv2
import math

w = np.zeros([77*71, 10]) # vetor de pesos
b = np.zeros([10]) # vetor de bias

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
				images += [cv2.imread(path + x + '/' + nome_das_imagens[l], cv2.IMREAD_GRAYSCALE).reshape([77*71])]
				labels.append(int(x))
			pass
		else:
			for l in range (threshold, len(nome_das_imagens)):
				images += [cv2.imread(path + x + '/' + nome_das_imagens[l], cv2.IMREAD_GRAYSCALE).reshape([77*71])]
				labels.append(int(x))

	return np.array(images), np.array(labels)

def sigmoide(z):
	return 1.0/(1.0 + np.exp(-z))

# Função para retornar as derivadas da função Sigmóide
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def gradient_descent_step(b0, w0, batch, learning_rate):
	b_grad = 0
	w_grad = 0
	N = len(batch)
	# compute gradients
	for i in range(N):
		x = batch[i, 0]
		y = batch[i, 1]
		b_grad += (2.0/N) * (w0*x + b0 - y)
		w_grad += (2.0/N) * x * (w0*x + b0 - y)

	# update parameters
	b1 = b0 - (learning_rate * b_grad)
	w1 = w0 - (learning_rate * w_grad)
	return b1, w1

def avaliacao(input):
	return np.vectorize(sigmoide)(np.dot(input, w) + b)

def backprop(input, output_esperado):
	for image in (input):
		o = avaliacao(image)
		d = o*(1-o)*(o-output_esperado) # retorna um nparray
	return d


if __name__ == '__main__':
	path = "../data_part1/train"
	imgs_train, l_train = read_data(path, True)
	imgs_validation, l_validation = read_data(path, False)

	print (len(imgs_train))
	print (len(imgs_validation))
	
	# cv2.imshow('image', imgs_train[0])
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	# imgs_train = np.reshape(imgs_train, [imgs_train.shape[0], imgs_train.shape[1], imgs_train.shape[2], 1])
	
	# imgs_train = np.expand_dims(imgs_train, 3)
	# imgs_validation = np.expand_dims(imgs_validation, 3)

	# l_train = np.expand_dims(l_train, 1)
	# l_validation = np.expand_dims(l_validation, 1)
	#print (imgs_train.shape)
	#print (imgs_validation.shape)
	# print (l_validation.shape)
	# print (l_train.shape)
	#print(avaliacao(imgs_train[0]))

	out_esp = [0,0,1,0,0,0,0,0,0, 0]
	test = backprop(imgs_train[0], out_esp)
	print(test)