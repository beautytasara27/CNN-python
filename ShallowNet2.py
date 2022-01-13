import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from FullyConnected import Dense
from ConvolutionLayer import Convolutional
from Reshape import Reshape
from TanhActivation import Tanh
from SigmoidActivation import Sigmoid
from Loss import categorical_cross_entropy, categorical_cross_entropy_prime, binary_cross_entropy_prime, binary_cross_entropy
from network import train, predict
from SoftmaxActivation import Softmax
from ReLUActivation import ReLU

def preprocess_data(x, y, limit):

    all_indices = np.where(y == 0)[0][:limit]
    for i in range(0,10):
        index = np.where(y == i)[0][:limit]
        all_indices = np.hstack((all_indices, index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = np_utils.to_categorical(y)
    y = y.reshape(len(y), 10, 1)
    return x, y

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 200)
x_test, y_test = preprocess_data(x_test, y_test, 200)

network = [
    Convolutional((1, 28, 28), 3, 5), #5 3x3 kernels
    Tanh(),
    #Convolutional((1, 26, 26), 3, 5), #28-3+1
   # Tanh(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 10),
    Softmax(),
]

train(
    network,
    categorical_cross_entropy,
    categorical_cross_entropy_prime,
    x_train,
    y_train,
    epochs=100,
    learning_rate=0.01
)


correct_predictions = 0
for x, y in zip(x_test, y_test):
    output = predict(network, x)

    if (np.argmax(output) == np.argmax(y)):
        correct_predictions += 1

    print(f"pred: {np.argmax(output)}, actual: {np.argmax(y)}")
print("Accuracy :",correct_predictions/len(x_test))