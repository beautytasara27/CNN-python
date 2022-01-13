import numpy as np
from ActivationLayer import ActivationLayer

class ReLU(ActivationLayer):
    def __init__(self):
        def relu(x):
          #  print("x =", x)
        #    print("max", np.maximum(1e-7, x))
            return np.maximum(1e-7, x)

        def relu_prime(x):
          #  print("relu received", x)
           # print("relu prime", (x > 0).astype(int))
            return (x > 1e-7).astype(int)
        super().__init__(relu, relu_prime)