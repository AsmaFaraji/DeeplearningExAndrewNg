from __future__ import division
import h5py
import numpy as np



def load_data():
    with h5py.File('/Users/asma/^_^/L_layerDeepLearning/train_catvnoncat.h5', "r") as train_dataset:
        train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
        train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('/Users/asma/^_^/L_layerDeepLearning/test_catvnoncat.h5', "r")

    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def initialize_parameters(layersize):
    np.random.seed(2)
    L = len(layersize)
    parameters = dict()
    for i in range(1, L):
        parameters['W' + str(i)] = np.random.randn(layersize[i], layersize[i - 1]) * 0.01
        parameters['b' + str(i)] = np.zeros((layersize[i], 1))
    return parameters

def sigmoid(Z):
    A = 1 / 1 + np.exp(-1 * Z)

    cache = Z

    return A, cache

def relu(Z):
    A = np.maximum(0, Z)

    cache = Z

    return A, cache

def relu_backward(dA, cache):

    Z = cache

    dZ = np.array(dA, copy=True)

    dZ[Z <= 0] = 0

    return dZ

def sigmoid_backward(dA, cache):

    Z = cache

    p = 1. / 1 + np.exp(-Z)

    dZ  = dA * p * (1 - p)
    return dZ


def cost_function(AL, Y):
    m = Y.shape[1]

    cost = -1 / m * np.sum(np.dot(Y.T, np.log(AL)) + np.sum(np.dot((1 - Y).T, np.log(1 - AL))))

    cost = np.squeeze(cost)

    return cost

def linear_forward(A_prev , W, b ):

    Z = np.dot(W, A_prev) + b

    cache = ( A_prev, W, b)

    return Z,cache

def linear_forward_activation(A_prev, W, b , activation):

    Z , linear_cache = linear_forward(A_prev, W, b)

    if activation == "relu":
        A,activation_cache = relu(Z)
    else:
        A, activation_cache = sigmoid(Z)

    cache = (linear_cache, activation_cache)

    return A,cache

def L_model_forward(X, parameters):
    caches = []
    L = len(parameters) // 2
    A_prev = X
    for i in range(1, L):
        A_prev, cache = linear_forward_activation(A_prev, parameters['W' +str(i)], parameters['b' + str(i)], "relu")
        caches.append(cache)
    AL, cache = linear_forward_activation(A_prev, parameters['W' + str(L)],
                                          parameters['b' + str(L)], "sigmoid")

    caches.append(cache)
    return AL, caches

def linear_backward(dZ, linear_cache):
    m = dZ.shape[1]

    AL_1, W, b = linear_cache
    dw = 1 / m * np.dot(dZ, AL_1.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dAL_1 = np.dot(W.T, dZ)

    return dAL_1, dw, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)

        dAL_1, dw, db = linear_backward(dZ, linear_cache)
    elif activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dAL_1, dw, db = linear_backward(dZ, linear_cache)

    return dAL_1, dw, db


def L_model_backward(AL, Y, cache):

    grad = {}
    L = len(cache)
    dAL = - np.divide(Y, AL) - np.divide((1 - Y), (1 - AL))
    current_cache = cache[L - 1]
    grad["dA" + str(L)], grad["dw" + str(L)], grad["db" + str(L)] = linear_activation_backward(dAL, current_cache,
                                                                                               "sigmoid")

    for l in reversed(range(L - 1)):
        current_cache = cache[l]
        grad["dA" + str(l + 1)], grad["dw" + str(l + 1)], grad["db" + str(l + 1)] = linear_activation_backward(
            grad["dA" + str(l + 2)],
            current_cache, 'relu')

    return grad

def update_parameters(parameters, grads, learning_rate):
    for i in range(1, len(parameters) // 2 + 1):
        parameters['W' + str(i)] = parameters['W' + str(i)] - grads['dw' + str(i)] * learning_rate
        parameters['b' + str(i)] = parameters['b' + str(i)] - grads['db' + str(i)] * learning_rate

    return parameters

def predict(X, y, parameters):

    m = X.shape[1]

    n = len(parameters) // 2 #number of nueral network layers

    p = np.zeros((1, m))

    probs,cache = L_model_forward(X, parameters)

    for i  in range(0,probs.shape[1]):

        if probs[0,i] >= 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    print ("Accuracy : " + str (np.sum((y==p)) / m))

    return p


