import numpy as np
import matplotlib.pyplot as plt
DEBUG = False 

def sigmoid(Xm):
    Xmcp = np.copy(Xm)
    for x in np.nditer(Xmcp, op_flags=['readwrite']):
        x[...] = 1/(1 + np.exp(-x))
    return Xmcp

# Inputs: 
# W1m = [[w10,w11,w12]
#        [w20,w21,w22]]
def y_fw_propagate(W1m,W2m,Xm): 
    Bm = np.matmul(W1m,Xm.transpose())
    Zm = sigmoid(Bm)
    #Zpm =  (np.concatenate((np.array([[1,1,1,1]]),Zm),axis=0))
    Zpm =  (np.concatenate((np.ones((1,Xm.shape[0])),Zm),axis=0))
    if(DEBUG):
        print("Zpm:")
        print(Zpm)
    A1m = np.matmul(W2m,Zpm)
    return sigmoid(A1m),Zm

def stepGradient(Xm,Y1m,Tm,Zm,W1m,W2m,gamma):
    #compute W^(2) gradient 
    #W10,W11,W12

    #remove W10
    #W2pm = [[W11,W12]]
    W2pm = np.delete(W2m,0).reshape(1,2)

    Zpm =  (np.concatenate((np.array([[1,1,1,1]]),Zm),axis=0))
    DT2m = (Y1m - Tm)*Y1m*(1 - Y1m)
    gradient_W2m = DT2m*Zpm
    if(DEBUG):
        print("gradient_W2m")
        print(gradient_W2m)
    gradient_W2m = gradient_W2m.sum(axis=1) #returns row vector

    #compute W^(1) gradient
    # Wji, ji = 10,20,11,21,12,22
    # 4 columns  for n = 1,2,3,4
    # 2 rows for W(j=1), W(j=2)
    gradient_W1i0m = DT2m*W2pm.transpose()*Zm*(1-Zm)*Xm[:,[0]].transpose() #W(j=1,2)(i=0), n=1,2,3
    gradient_W1i1m = DT2m*W2pm.transpose()*Zm*(1-Zm)*Xm[:,[1]].transpose() #W(j=1,2)(i=1), n=1,2,3
    gradient_W1i2m = DT2m*W2pm.transpose()*Zm*(1-Zm)*Xm[:,[2]].transpose() #W(j=1,2)(i=2), n=1,2,3
    gradient_W1m = np.concatenate((gradient_W1i0m,gradient_W1i1m),axis=0)
    gradient_W1m = np.concatenate((gradient_W1m,gradient_W1i2m),axis=0)

    
    if(DEBUG):
        print("gradient_W1m")
        print(gradient_W2m)
    gradient_W1m = gradient_W1m.sum(axis=1) # sum and return Wji only 
    #At this point, returns gradient_W1m row vector 10,20,11,21,12,22
    gradient_W1m = gradient_W1m.reshape(3,2).transpose()

    W1m_next = W1m - gamma*gradient_W1m
    W2m_next = W2m - gamma*gradient_W2m
    return W1m_next,W2m_next


def main():
    Xm = [
         [1,0,0],
         [1,0,1],
         [1,1,0],
         [1,1,1]
        ]
    Xm = np.array(Xm)
    Tm = [[0,1,1,0]]
    Tm = np.array(Tm)
    W2m = np.random.randn(1,3)*np.sqrt(2.0/3.0)
    W2m = np.array(W2m)
    W1m = np.random.randn(2,3)*np.sqrt(2.0/6.0)
    W1m = np.array(W1m)

    Ym = None
    Zm = None
    gamma = 0.25
    epochs = 10000

    for i in range(epochs):
        Ym,Zm = y_fw_propagate(W1m,W2m,Xm)
        W1m,W2m = stepGradient(Xm,Ym,Tm,Zm,W1m,W2m,gamma)


    #training complete
    stdout = "W1_ji = \n" +\
    "[[W10,W11,W12],\n" +\
    "[W20,W21,W22]]"
    print(stdout)
    print(W1m)
    stdout = "W2_kj = \n" +\
    "[[W10,W11,W12]]"
    print(stdout)
    print(W2m)

    #Apply inputs and print output
    Ym,Zm = y_fw_propagate(W1m,W2m,Xm)
    stdout = "\ny(%s) =\n %s\n"%(Xm,Ym.transpose())
    print(stdout)

    #Apply inputs and print output
    X2m = [
            [1,0,0],
            [1,0,0],
            [1,0,0],
            [1,1,1],
            [1,1,0],
            [1,1,0],
            [1,0,1],
            [1,1,1]
          ]
    X2m = np.array(X2m)
    Ym,Zm = y_fw_propagate(W1m,W2m,X2m)
    stdout = "y(%s) =\n %s\n"%(X2m,Ym.transpose())
    print(stdout)


if __name__ == "__main__":
    main()