#multiple linear regression non-normalized regularized using gradient descent
import numpy as np

X_train = [[3, 4], [1, 2], [6, 7]]
y_train = [3, 1, 6]

def compute_prediction(x, w, b):
    return np.dot(x,w)+b

def compute_cost(X, y, w, b, lamda = 1):
    m = X.shape[0]
    n = X.shape[1]
    term1 = 0
    term2 = 0
    for i in range(m):
        term1+=(np.dot(X[i], w)+b-y[i])**2
    term1/=(2*m)

    for i in range(n):
        term2+=(w[i]**2)
    term2*=(lamda/(2*m))

    return term1+term2

def compute_gradient(X, y, w, b, lamda):
    m = X.shape[0]
    n = X.shape[1]
    dj_dw = np.zeros(n)
    dj_db = 0
    for i in range (m):
        diff_i = np.dot(X[i], w)+b-y[i]
        dj_db += diff_i
        for j in range(n):
            dj_dw[j]+=diff_i*X[i, j]
    reg = np.zeros(n)
    
    return dj_db/m, (dj_dw/m)+(lamda/m)*w

def gradient_descent(X, y, w, b, alpha, lamda, iters):
    for i in range(iters):
        dj_db, dj_dw = compute_gradient(X, y, w, b, lamda)
        w-=alpha*dj_dw
        b-=alpha*dj_db
    return w, b

if __name__ == "__main__":
    """
    m,n = X_train.shape
    w = np.zeros(n)
    b = 0
    alpha = 0.1
    lamda = 1
    iters = 10000

    w, b = gradient_descent(X_train, y_train, w, b, alpha, lamda, iters)
    print("weights:", w, "| bias:", b)
    print("Minimum cost:", compute_cost(X_train, y_train, w, b))
    x = [8, 9]
    print("Prediction for x of", x, ":", compute_prediction(x, w, b))
    """
    np.random.seed(1)
    X_tmp = np.random.rand(5,3)
    y_tmp = np.array([0,1,0,1,0])
    w_tmp = np.random.rand(X_tmp.shape[1])
    b_tmp = 0.5
    lambda_tmp = 0.7
    dj_db_tmp, dj_dw_tmp =  compute_gradient(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

    print(f"dj_db: {dj_db_tmp}", )
    print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )