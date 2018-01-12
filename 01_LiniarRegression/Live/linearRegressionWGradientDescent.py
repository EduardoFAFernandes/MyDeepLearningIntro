from numpy import *

def compute_error_for_given_points(m, b, points):
	totalError = 0
	for point in points:
		totalError += (point[1] - (m * point[0] + b))**2
	return totalError / float(size(points))

def step_gradient(current_b, current_m, points, learning_rate):
	gradient_m = 0
	gradient_b = 0
	N = float(len(points))
	for point in points:
		x = point[0]
		y = point[1]
		gradient_m += -(2/N) * x *(y - ((current_m * x) + current_b))
		gradient_b += -(2/N) * (y - ((current_m * x) + current_b))
	
	new_b = current_b - (learning_rate * gradient_b)
	new_m = current_m - (learning_rate * gradient_m)

	return [ new_b, new_m]

def gradient_descent_runner(points, initial_b, initial_m, 
							learning_rate, num_iterations ):
	b = initial_b
	m = initial_m

	for i in range(num_iterations):
		b,m = step_gradient(b, m, array(points), learning_rate)

	return [b, m]

def run():
	points = genfromtxt('data.csv', delimiter = ',')
	
	#hyperparameters

	learning_rate = 0.0001	
	#y = mx + b
	initial_b = 0
	initial_m = 0
	num_iterations = 1000
	
	print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(
		initial_b, initial_m, 
		compute_error_for_given_points(initial_b, initial_m, points)))
	
	print("Running...")
	
	[b, m] = gradient_descent_runner(points, initial_b, initial_m,
									 learning_rate, num_iterations )
	print("After {0} iterations b = {1}, m = {2}, error = {3}".format(
		num_iterations, b, m, compute_error_for_given_points(b, m, points)))





if __name__ == '__main__':
	run()