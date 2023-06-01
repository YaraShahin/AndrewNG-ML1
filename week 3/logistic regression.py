import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])           

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def compute_cost_logistic(x, y, w, b):
    m = x.shape[0]
    cost = 0
    for i in range (m):
        fx = sigmoid(w*x[i]+b)
        cost+= -(y[i]*np.log(fx))-((1-y[i])*np.log(1-fx))
    return cost/m

def compute_gradient_logistic(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range (m):
        error=sigmoid(w*x[i]+b)-y[i]
        dj_dw+=error*x[i]
        dj_db+=error
    return dj_dw/m, dj_db/m

def gradient_descent(x, y, w, b, iters, alpha):
    for i in range(iters):
        dj_dw, dj_db = compute_gradient_logistic(x, y, w, b)
        w-=alpha*dj_dw
        b-=alpha*dj_db
    return w,b

if __name__ == "__main__":
    w = 0
    b = 0
    iters = 10000
    alpha = 0.1
    w, b = gradient_descent(x_train, y_train, w, b, iters, alpha)

    print("weights:", w, "| bias:", b)
    print("Minimum cost:", compute_cost_logistic(x_train, y_train, w, b))




