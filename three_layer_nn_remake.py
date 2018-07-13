import numpy as np
from NeuralNetwork import NeuralNetwork


def convert_output(o):
    if (o >= .5):
        return 1
    else:
        return 0;

vout = np.vectorize(convert_output)

nn = NeuralNetwork(layers = [8, 12, 8], activations = ['relu', 'sigmoid'])

#input data
X = np.array([[0, 0, 1, 0, 1, 1, 0, 1],
              [0, 1, 1, 1, 1, 0, 0, 0],
              [1, 0, 1, 0, 0, 0, 1, 1],
              [1, 1, 1, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 1, 0, 0, 0, 1, 0],
              [0, 1, 1, 0, 1, 0, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1]])

#ouput data
y = np.array([[1, 1, 0, 1, 0, 0, 1, 0],
              [1, 0, 0, 0, 0, 1, 1, 1],
              [0, 1, 0, 1, 1, 1, 0, 0],
              [0, 0, 0, 0, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1],
              [1, 0, 0, 1, 1, 1, 0, 1],
              [1, 0, 0, 1, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]])

nn.train(X, y, step_size = 0.1, epochs = 10000)

print("Output after training: ")
print(vout(nn.fprop(X)))

test = np.array([[0, 0, 0, 1, 1, 0, 1, 1],
                 [0, 0, 0, 0, 0, 0, 1, 0],
                 [1, 1, 0, 0, 1, 0, 1, 0],
                 [1, 0, 0, 0, 1, 1, 0, 0]])

print("Test result: ")
print(vout(nn.fprop(test)))
