import numpy as np
from NeuralNetwork import NeuralNetwork

nn = NeuralNetwork(layers = [3, 1], activations = ['sigmoid'])

#input data
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

#ouput data
y = np.array([[0, 0, 1, 1]]).T

nn.train(X, y, step_size = 25, epochs = 10000)

print("Output after training: ")
print(nn.fprop(X))

test = np.array([[1, 0, 0],
                 [1, 1, 0],
                 [0, 1, 0],
                 [0, 0, 1]])

print("Test result: ")
print(nn.fprop(test))
