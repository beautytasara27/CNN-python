import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from FullyConnected import Dense
from TanhActivation import Tanh
from Loss import mse, mse_prime
from network import train, predict

#X = np.reshape([[-1, 1], [-1, 2], [3, 0], [-2, 1], [4, 1], [1, 4], [7, 0], [2, 3], [4, -2], [3, 0]], (10, 2, 1))
#Y = np.reshape([[2], [3], [9], [5], [17], [5], [49], [7], [14], [9]], (10, 1, 1))

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))
network = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]
epochs = 10000
lr = 0.1  #0.01 0.1 1
# train
train(network, mse, mse_prime, X, Y, epochs=epochs, learning_rate=lr)

# decision boundary plot
#generates 20 values between 0 and 1
X_Value = np.linspace(0, 1, 20)
Y_Value = np.linspace(0, 1, 20)

points = []

for x in X_Value:
    for y in Y_Value:
        prediction = predict(network, [[x], [y]]) #returns 1x1 aray with predcted value
        points.append([x, y, prediction[0,0]])

points = np.array(points)
print(points)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1],points[:, 2], c=points[:, 2], cmap="winter")
plt.show()
