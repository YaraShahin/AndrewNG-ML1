x_train = [1, 2]
y_train = [300, 500]
m = len(x_train)

def cost(w, b):
    error = 0
    for i in range(m):
        error+=((w*x_train[i]+b)-y_train[i])**2
    error = error/(2*m)
    return error

#finds the sum of the partial derivative of w (type 1), or b (type 2)
def segma(w, b):
    dj_dw = 0
    dj_db = 0
    for i in range (m):
        dj_dw+=(w*x_train[i]+b-y_train[i])*x_train[i]
        dj_db+=(w*x_train[i]+b-y_train[i])
    return dj_dw, dj_db


def main():
    w = 0
    b = 0
    a = 0.01
    current = cost(w, b)
    prev = current + 10

    while (abs(prev-current)>0.0000000005):
        dj_dw, dj_db = segma(w, b)
        w = w - (a/m) * dj_dw
        b = b - (a/m) * dj_db
        prev = current
        current = cost(w, b)
    
    print(w, b)

main()