#!/usr/bin/python
# import numpy as np
from numpy import *
import random

def compute_error_for_line_given_points(b, m, points):
	totalError = 0
	for i in range(0, len(points)):
		x = points[i][0]
		y = points[i][1]
		totalError += (y - (m * x + b)) ** 2
	return totalError / float(len(points))

def step_gradient(b_current, m_current, points, leaningRate):
	b_gradient = 0
	m_gradient = 0
	N = float(len(points))
	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]
		# b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
		# m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
		b_gradient += -x * (y - (m_current * x + b_current))
		m_gradient += -(y - (m_current * x + b_current))
	b_gradient *= 2.0 / float(N)
	m_gradient *= 2.0 / float(N)
	new_b = b_current - (leaningRate * b_gradient)
	new_m = m_current - (leaningRate * m_gradient)
	return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
	b = starting_b
	m = starting_m
	for i in range(num_iterations):
		b, m = step_gradient(b, m, array(points), learning_rate)
	return [b, m]

def generate():
	return [[random.randint(0, 100), random.randint(0, 100)] for x in range(200)]
	l = []
	for i in range(0, 200):
		x = random.randint(0, 100)
		y = random.randint(0, 100)
		l += [[x,y]]
	return l

def run():
	random.seed(1)
	points = [[random.random() for i in range(500)] for j in range(500)]

	learning_rate = 0.0001
	initial_b = 0 # initial y-intercept guess
	initial_m = 0 # initial slope guess
	num_iterations = 1000
	print ("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b,
			initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
	print ("Running...")
	[b,m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
	print ("After {0} iterarions b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))

if __name__ == '__main__':
	run()