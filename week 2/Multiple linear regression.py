import numpy as np
import matplotlib.pyplot as plt

#factors to change: data normalization, sample size, learning rate, convergence test

#dataset 1
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

#dataset 2
X_train = np.array([[1.24e+03, 3.00e+00, 1.00e+00, 6.40e+01],
 [1.95e+03, 3.00e+00, 2.00e+00, 1.70e+01],
 [1.72e+03, 3.00e+00, 2.00e+00, 4.20e+01],
 [1.96e+03, 3.00e+00, 2.00e+00, 1.50e+01],
 [1.31e+03, 2.00e+00, 1.00e+00, 1.40e+01],
 [8.64e+02, 2.00e+00, 1.00e+00, 6.60e+01],
 [1.84e+03, 3.00e+00, 1.00e+00, 1.70e+01],
 [1.03e+03, 3.00e+00, 1.00e+00, 4.30e+01],
 [3.19e+03, 4.00e+00, 2.00e+00, 8.70e+01],
 [7.88e+02, 2.00e+00, 1.00e+00, 8.00e+01],
 [1.20e+03, 2.00e+00, 2.00e+00, 1.70e+01],
 [1.56e+03, 2.00e+00, 1.00e+00, 1.80e+01],
 [1.43e+03, 3.00e+00, 1.00e+00, 2.00e+01],
 [1.22e+03, 2.00e+00, 1.00e+00, 1.50e+01],
 [1.09e+03, 2.00e+00, 1.00e+00, 6.40e+01],
 [8.48e+02, 1.00e+00, 1.00e+00, 1.70e+01],
 [1.68e+03, 3.00e+00, 2.00e+00, 2.30e+01],
 [1.77e+03, 3.00e+00, 2.00e+00, 1.80e+01],
 [1.04e+03, 3.00e+00, 1.00e+00, 4.40e+01],
 [1.65e+03, 2.00e+00, 1.00e+00, 2.10e+01],
 [1.09e+03, 2.00e+00, 1.00e+00, 3.50e+01],
 [1.32e+03, 3.00e+00, 1.00e+00, 1.40e+01],
 [1.59e+03, 0.00e+00, 1.00e+00, 2.00e+01],
 [9.72e+02, 2.00e+00, 1.00e+00, 7.30e+01],
 [1.10e+03, 3.00e+00, 1.00e+00, 3.70e+01],
 [1.00e+03, 2.00e+00, 1.00e+00, 5.10e+01],
 [9.04e+02, 3.00e+00, 1.00e+00, 5.50e+01],
 [1.69e+03, 3.00e+00, 1.00e+00, 1.30e+01],
 [1.07e+03, 2.00e+00, 1.00e+00, 1.00e+02],
 [1.42e+03, 3.00e+00, 2.00e+00, 1.90e+01],
 [1.16e+03, 3.00e+00, 1.00e+00, 5.20e+01],
 [1.94e+03, 3.00e+00, 2.00e+00, 1.20e+01],
 [1.22e+03, 2.00e+00, 2.00e+00, 7.40e+01],
 [2.48e+03, 4.00e+00, 2.00e+00, 1.60e+01],
 [1.20e+03, 2.00e+00, 1.00e+00, 1.80e+01],
 [1.84e+03, 3.00e+00, 2.00e+00, 2.00e+01],
 [1.85e+03, 3.00e+00, 2.00e+00, 5.70e+01],
 [1.66e+03, 3.00e+00, 2.00e+00, 1.90e+01],
 [1.10e+03, 2.00e+00, 2.00e+00, 9.70e+01],
 [1.78e+03, 3.00e+00, 2.00e+00, 2.80e+01],
 [2.03e+03, 4.00e+00, 2.00e+00, 4.50e+01],
 [1.78e+03, 4.00e+00, 2.00e+00, 1.07e+02],
 [1.07e+03, 2.00e+00, 1.00e+00, 1.00e+02],
 [1.55e+03, 3.00e+00, 1.00e+00, 1.60e+01],
 [1.95e+03, 3.00e+00, 2.00e+00, 1.60e+01],
 [1.22e+03, 2.00e+00, 2.00e+00, 1.20e+01],
 [1.62e+03, 3.00e+00, 1.00e+00, 1.60e+01],
 [8.16e+02, 2.00e+00, 1.00e+00, 5.80e+01],
 [1.35e+03, 3.00e+00, 1.00e+00, 2.10e+01],
 [1.57e+03, 3.00e+00, 1.00e+00, 1.40e+01],
 [1.49e+03, 3.00e+00, 1.00e+00, 5.70e+01],
 [1.51e+03, 2.00e+00, 1.00e+00, 1.60e+01],
 [1.10e+03, 3.00e+00, 1.00e+00, 2.70e+01],
 [1.76e+03, 3.00e+00, 2.00e+00, 2.40e+01],
 [1.21e+03, 2.00e+00, 1.00e+00, 1.40e+01],
 [1.47e+03, 3.00e+00, 2.00e+00, 2.40e+01],
 [1.77e+03, 3.00e+00, 2.00e+00, 8.40e+01],
 [1.65e+03, 3.00e+00, 1.00e+00, 1.90e+01],
 [1.03e+03, 3.00e+00, 1.00e+00, 6.00e+01],
 [1.12e+03, 2.00e+00, 2.00e+00, 1.60e+01],
 [1.15e+03, 3.00e+00, 1.00e+00, 6.20e+01],
 [8.16e+02, 2.00e+00, 1.00e+00, 3.90e+01],
 [1.04e+03, 3.00e+00, 1.00e+00, 2.50e+01],
 [1.39e+03, 3.00e+00, 1.00e+00, 6.40e+01],
 [1.60e+03, 3.00e+00, 2.00e+00, 2.90e+01],
 [1.22e+03, 3.00e+00, 1.00e+00, 6.30e+01],
 [1.07e+03, 2.00e+00, 1.00e+00, 1.00e+02],
 [2.60e+03, 4.00e+00, 2.00e+00, 2.20e+01],
 [1.43e+03, 3.00e+00, 1.00e+00, 5.90e+01],
 [2.09e+03, 3.00e+00, 2.00e+00, 2.60e+01],
 [1.79e+03, 4.00e+00, 2.00e+00, 4.90e+01],
 [1.48e+03, 3.00e+00, 2.00e+00, 1.60e+01],
 [1.04e+03, 3.00e+00, 1.00e+00, 2.50e+01],
 [1.43e+03, 3.00e+00, 1.00e+00, 2.20e+01],
 [1.16e+03, 3.00e+00, 1.00e+00, 5.30e+01],
 [1.55e+03, 3.00e+00, 2.00e+00, 1.20e+01],
 [1.98e+03, 3.00e+00, 2.00e+00, 2.20e+01],
 [1.06e+03, 3.00e+00, 1.00e+00, 5.30e+01],
 [1.18e+03, 2.00e+00, 1.00e+00, 9.90e+01],
 [1.36e+03, 2.00e+00, 1.00e+00, 1.70e+01],
 [9.60e+02, 3.00e+00, 1.00e+00, 5.10e+01],
 [1.46e+03, 3.00e+00, 2.00e+00, 1.60e+01],
 [1.45e+03, 3.00e+00, 2.00e+00, 2.50e+01],
 [1.21e+03, 2.00e+00, 1.00e+00, 1.50e+01],
 [1.55e+03, 3.00e+00, 2.00e+00, 1.60e+01],
 [8.82e+02, 3.00e+00, 1.00e+00, 4.90e+01],
 [2.03e+03, 4.00e+00, 2.00e+00, 4.50e+01],
 [1.04e+03, 3.00e+00, 1.00e+00, 6.20e+01],
 [1.62e+03, 3.00e+00, 1.00e+00, 1.60e+01],
 [8.03e+02, 2.00e+00, 1.00e+00, 8.00e+01],
 [1.43e+03, 3.00e+00, 2.00e+00, 2.10e+01],
 [1.66e+03, 3.00e+00, 1.00e+00, 6.10e+01],
 [1.54e+03, 3.00e+00, 1.00e+00, 1.60e+01],
 [9.48e+02, 3.00e+00, 1.00e+00, 5.30e+01],
 [1.22e+03, 2.00e+00, 2.00e+00, 1.20e+01],
 [1.43e+03, 2.00e+00, 1.00e+00, 4.30e+01],
 [1.66e+03, 3.00e+00, 2.00e+00, 1.90e+01],
 [1.21e+03, 3.00e+00, 1.00e+00, 2.00e+01],
 [1.05e+03, 2.00e+00, 1.00e+00, 6.50e+01]])
y_train=np.array([300.,   509.8,  394.,   540.,   415.,   230.,   560.,   294.,   718.2,  200.,
 302.,   468.,   374.2 , 388.   ,282.  , 311.8  ,401. ,  449.8,  301.  , 502.,
 340.,   400.28, 572.  , 264.   ,304.  , 298.   ,219.8,  490.7,  216.96, 368.2,
 280.,   526.87, 237.  , 562.43 ,369.8 , 460.   ,374. ,  390. ,  158.  , 426.,
 390.,   277.77, 216.96, 425.8  ,504.  , 329.   ,464. ,  220. ,  358.  , 478.,
 334.,   426.98, 290.,   463.,   390.8,  354. ,  350.  , 460.  , 237. ,  288.3,
 282.,   249.,   304.,   332.,   351.8,  310. ,  216.96, 666.34, 330. ,  480.,
 330.3,  348.,   304.,   384.,   316. ,  430.4,  450.  , 284.  , 275. ,  414.,
 258.,   378.,   350.,   412.,   373. ,  225. ,  390.  , 267.4 , 464. ,  174.,
 340.,   430.,   440.,   216.,   329. ,  388. ,  390.  , 356.  , 257.8 ])

#m: number of datapoints, n: number of features
m = X_train.shape[0]
n = X_train.shape[1]

#initial w and b assumptions
w = np.zeros((n,))
b = 0

#learning rate alpha
a = 5.0e-7                  #too small when set is normalized: slow and bad convergence
a = 0.1                     #good alpha for normalized 

#scaling: 1. x-min/max-min 2. subtract by mean then divide by max-min 3. subtract by mean then divide by standard deviation 
def z_score_normalize(x):
    mu = np.mean(x, axis = 0)
    deviation = np.std(x, axis = 0)
    x = (x-mu)/deviation
    return mu, deviation, x


def cost(w, b):
    error = 0
    for i in range (m):
        error+=(np.dot(w,X_train[i])+b-y_train[i])**2
    error = error / (2*m)
    return error

def sigma(w, b):
    dj_db = 0
    dj_dw = np.zeros((n,))
    for i in range(m):
        dot = (np.dot(w, X_train[i])+b)-y_train[i]
        dj_db+=dot
        for j in range (n):
            dj_dw[j] += dot*X_train[i, j]
    return dj_dw/m, dj_db/m

def predict(x, w, b, mu=0, dev=1):
    x = (x-mu)/dev
    return np.dot(w,x)+b

mu, dev, X_train = z_score_normalize(X_train)
current = cost(w, b)
prev = current+10
i = 0
xaxis = [i]
yaxis = [current]
while(prev-current>0.01):
#while (i<1000):
    dj_dw, dj_db = sigma(w, b)
    w = w - a* dj_dw
    b = b - a* dj_db
    prev = current
    current = cost(w, b)
    i+=1
    xaxis.append(i)
    yaxis.append(current)

print("Number of iterations:", i)
print("w parameters:", w)
print("bias:", b)
print("cost:",cost(w, b))

x_test = np.array([1200, 3, 1, 40])
print("Prediction: for x of",x_test,", the house price is", predict(x_test,w, b, mu, dev) )

#plot a #iterations vs. cost curve to see if the learning rate is good enough
plt.plot(xaxis, yaxis, c = "r")
plt.xlabel("Number of iterations")
plt.ylabel("cost")
plt.title("Learning curve")
plt.show()