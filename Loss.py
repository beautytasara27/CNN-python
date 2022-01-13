import numpy as np
#error = 1/2(actual-predicted)^2
def mse(y_actual, y_pred):
    #print(y_actual, y_pred)
    #print(np.mean(np.power(y_actual - y_pred, 2)))
    return np.mean(np.power(y_actual - y_pred, 2))

def mse_prime(y_actual, y_pred):
   # print(np.size(y_actual))
    return 2 * (y_pred - y_actual) / np.size(y_actual)

def binary_cross_entropy(y_actual, y_pred):
   # print("mean", np.mean(-y_actual * np.log(y_pred) - (1 - y_actual) * np.log(1 - y_pred)))
    return (-y_actual * np.log(y_pred) - (1 - y_actual) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_actual, y_pred):
   # print("mean prime", ((1 - y_actual) / (1 - y_pred) - y_actual / y_pred) / np.size(y_actual))
    return ((1 - y_actual) / (1 - y_pred) - y_actual / y_pred) / np.size(y_actual)

def categorical_cross_entropy(y_actual, y_pred):
    y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) #to remove the log 0
    correct_confidences = np.sum(y_pred_clipped*y_actual, axis=1)
   # print("conf", correct_confidences)
    negative_log = -np.log(np.max(correct_confidences))
   # print("neg log",negative_log)
   # print("neg log mean", np.mean(negative_log))
    return negative_log

def categorical_cross_entropy_prime(y_actual, y_pred):
   # print("actual",y_actual)
   # print("ypred", y_pred)
  #  print("diffe",y_pred - y_actual)
    return y_pred - y_actual