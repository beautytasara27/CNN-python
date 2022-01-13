import numpy as np
from BaseLayer import BaseLayer
from ActivationLayer import ActivationLayer
class Softmax(BaseLayer):
    def forward(self, input):
        #subtracting the max prevents overflow, limits the values to the range (0,1)
      #  print(("inputs", input))
        #subtracting the max value to prevent overflow, doesnt impact the output of the softmax
        exponentials = np.exp(input - np.max(input)) # exp removes neg values,but still preserving the meaning,
        #print("temp", tmp )
        self.output = exponentials / np.sum(exponentials) #normalization
        #print("softmax", self.output.sum())
        return self.output

    def backward(self, output_gradient, learning_rate):
        n = np.size(self.output)
     #   print("prime1",self.output * (output_gradient - (output_gradient * self.output).sum(axis=1)[:, None]))
     #   print("prime",np.dot((np.identity(n) - self.output.T) * self.output, output_gradient))
        return self.output * (output_gradient - (output_gradient * self.output).sum(axis=1)[:, None])
            #np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
