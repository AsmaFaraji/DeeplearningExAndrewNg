import numpy as np
def linear_forward_testcase():
    np.random.seed(10)
    A = np.random.randn(3,4)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)

    return  A,W, b


def linear_forward_activation_testcase():
    np.random.seed(9)
    AL_1 = np.random.randn(3,4)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)

    return AL_1, W, b


def L_model_forward_testCase():
    np.random.seed(8)
    X = np.random.randn(3,4)
    w1 = np.random.randn(2,3)
    b1 = np.random.randn(2,1)
    w2 = np.random.randn(1, 2)
    b2 = np.random.randn(1,1)

    parameters = {"W1" : w1,
                  "b1" : b1,
                  "W2" : w2,
                  "b2" :b2 }

    return X, parameters, 2


def cost_function_testCase():
    np.random.seed(7)
    AL = np.array([[0.1, 0.3, 0.9]])
    Y = np.array([[0, 1, 1]])

    return AL , Y

def linear_backward_testCase():
    np.random.seed(6)
    dz = np.random.randn(2,4)
    AL_1 = np.random.randn(3,4)
    w = np.random.randn(2,3)
    b = np.random.randn(2,1)

    linear_cache = (AL_1, w, b)
    return dz, linear_cache

def linear_activation_backward_testCase():
    np.random.seed(5)
    dA = np.random.randn(2,4)
    Z = np.random.randn(2,4)
    AL_1 = np.random.randn(3, 4)
    w = np.random.randn(2, 3)
    b = np.random.randn(2, 1)

    linear_cache = (AL_1, w, b)
    activation_cache = Z

    cache = (linear_cache, activation_cache)

    return dA, cache

def L_model_backward_testCase():
    np.random.seed(4)
    A2 = np.array([[0.1, 0.3, 0.9]])
    Y = np.array([[0, 1, 1]])
    Z2 = np.random.randn(1,3)
    w2 = np.random.randn(1,2)
    b2 = np.random.randn(1,1)
    A1 = np.random.randn(2,3)
    Z1 = np.random.randn(2,3)
    w1 = np.random.randn(2,5)
    b1 = np.random.randn(2,1)
    X = np.random.rand(5,3)

    cache = []
    cache.append(((X, w1,b1), Z1))
    cache.append(((A1, w2, b2), Z2))

    return A2, Y, cache

def update_parameters_testCase():
    np.random.seed(3)
    w1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    w2 = np.random.randn(2,3)
    b2 = np.random.randn(2,1)

    parameters = {'W1': w1 , "b1": b1 , 'W2' : w2 , 'b2' : b2}

    np.random.seed(2)
    dw1 = np.random.randn(3,4)
    db1 = np.random.randn(3, 1)
    dw2 = np.random.randn(2, 3)
    db2 = np.random.randn(2, 1)

    grad =  {'dw1': w1 , "db1": b1 , 'dw2' : w2 , 'db2' : b2}

    return parameters,grad


