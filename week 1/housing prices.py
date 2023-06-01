x_train = [1, 2]
y_train = [300, 500]

def cost(w, b):
    error = 0
    for i in range(len(x_train)):
        error+=((w*x_train[i]+b)-y_train[i])**2
    error = error/(2*len(x_train))
    return error

def main():
    w = 0
    b = 0
    prev = cost(w, b)
    while True:
        current1 = cost(w+1,b+1) 
        current2 = cost(w-1,b+1)
        current3 = cost(w+1, b-1)
        current4 = cost(w-1, w-1)
        current5 = cost(w, b+1)
        current6 = cost(w, b-1)
        current7 = cost(w+1, b)
        current8 = cost(w-1, b)
        mn = min(current1, current2, current3, current4, current5, current6, current7, current8)
        print("-----------", prev, "----------")
        print("###", mn)
#        if mn>=prev:
#            break
        if (mn==current1):
            w = w+1
            b = b+1
            if mn>=prev:
                break
            prev = current1
        if (mn==current2):
            w = w-1
            b = b+1
            if mn>=prev:
                break
            prev = current2
        if (mn==current3):
            w = w+1
            b = b-1
            if mn>=prev:
                break
            prev = current3
        if (mn==current4):
            w = w-1
            b = b-1
            if mn>=prev:
                break
            prev = current4
        if (mn==current5):
            b = b+1
            if mn>=prev:
                break
            prev = current5
        if (mn==current6):
            b = b-1
            if mn>=prev:
                break
            prev = current6
        if (mn==current7):
            w = w+1
            if mn>=prev:
                break
            prev = current7
        if (mn==current8):
            w = w-1
            if mn>=prev:
                break
            prev = current8
        print(">>>", w, b)
        
    print(w, b)

main()

