import copy
import numpy as np
import matplotlib.pyplot as plt

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
y_train = np.array([0, 0, 0, 1, 1, 1])                                 #(m,)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def compute_cost_logistic(X, y, w, b):
    m = X.shape[0]
    cost = 0
    for i in range (m):
        fx = sigmoid(np.dot(w,X[i])+b)
        cost+= -(y[i]*np.log(fx))-((1-y[i])*np.log(1-fx))
    return cost/m

def compute_gradient_logistic(X, y, w, b):
    m,n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0
    for i in range(m):
        error = sigmoid(np.dot(X[i],w)+b)-y[i]
        for j in range(n):
            dj_dw[j]+=error*X[i,j]
        dj_db += error
    return dj_dw/m, dj_db/m

def gradient_descent(X, y, w_in, b_in, alpha, num_iters): 
    w = copy.deepcopy(w_in)
    b = b_in
    i = 0
    while(i<num_iters):
        dj_dw, dj_db = compute_gradient_logistic(X,y,w,b)
        w = w - alpha*dj_dw
        b-=alpha*dj_db
        i+=1
    return w, b

if __name__ == "__main__":
    m,n = X_train.shape
    w = np.zeros(n)
    b = 0
    alpha = 0.1
    iters = 10000

    w, b = gradient_descent(X_train, y_train, w, b, alpha, iters)
    print("weights:", w, "| bias:", b)
    print("Minimum cost:", compute_cost_logistic(X_train, y_train, w, b))




