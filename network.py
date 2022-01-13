import matplotlib.pyplot as plt

def predict(network, input):
    output = input #output of one layer is the input to the next
    for layer in network:
        output = layer.forward(output)
    return output

def train(network, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True):
    Errors = []
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)

            # error
            error += loss(y, output)


           # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)
        Errors.append(error/len(x_train))

        if verbose:
            print(f"{e + 1}/{epochs}, error={error/len(x_train)}")
    # Loss after each epoch
    plt.plot(list(range(0,epochs)),Errors)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title("Loss Plot")
    plt.show()