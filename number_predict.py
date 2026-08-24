import numpy as np
from pathlib import Path

def relu(x):
    return np.maximum(0,x)

def predict(X, params):
    Z1 = np.dot(X, params['w1']) + params['b1']
    A1 = relu(Z1)
    Z2 = np.dot(A1, params['w2']) + params['b2']    
    x_exp = np.exp(Z2 -np.max(Z2, axis = -1, keepdims=True))
    A2 = x_exp / np.sum(x_exp, axis=-1, keepdims=True)
    return A2

params = np.load(r"C:\Users\talon\Downloads\weights.npy", allow_pickle=True).item()
image = np.load(Path(input("Enter MNIST image or n to stop: ")), allow_pickle = True)

while(True):
     image = image.reshape(1, -1).astype(np.float32)
     number = np.argmax(predict(image, params),axis=1)
     print('This image is the number:', number)
     image = input("Enter MNIST image or n to stop")