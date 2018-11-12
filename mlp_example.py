import numpy as np
import matplotlib.pyplot as plt
DEBUG = True 


def f(x):
    return 1/(1+np.exp(-x))

def main():
    #W1_ji
    l_size = 6
    w1_10 = np.random.randn()*np.sqrt(2.0/l_size)
    w1_11 = np.random.randn()*np.sqrt(2.0/l_size)
    w1_12 = np.random.randn()*np.sqrt(2.0/l_size)
    w1_20 = np.random.randn()*np.sqrt(2.0/l_size)
    w1_21 = np.random.randn()*np.sqrt(2.0/l_size)
    w1_22 = np.random.randn()*np.sqrt(2.0/l_size)

    #w2_kj, k = 1
    i_size = 3
    w2_10 = np.random.randn()*np.sqrt(2.0/i_size)
    w2_11 = np.random.randn()*np.sqrt(2.0/i_size)
    w2_12 = np.random.randn()*np.sqrt(2.0/i_size)

    #Xi inputs, n = 0,..,N-1, N = 4
    x0_0,x1_0,x2_0 = 1,0,0
    x0_1,x1_1,x2_1 = 1,0,1
    x0_2,x1_2,x2_2 = 1,1,0
    x0_3,x1_3,x2_3 = 1,1,1


    #expected output 
    t1_0,t1_1,t1_2,t1_3 = 0,1,1,0
    gamma = 1
    epochs = 10000
    
    for i in range(epochs):
            
        b1_0 = w1_10*x0_0 + w1_11*x1_0 + w1_12*x2_0
        b1_1 = w1_10*x0_1 + w1_11*x1_1 + w1_12*x2_1
        b1_2 = w1_10*x0_2 + w1_11*x1_2 + w1_12*x2_2
        b1_3 = w1_10*x0_3 + w1_11*x1_3 + w1_12*x2_3

        b2_0 = w1_20*x0_0 + w1_21*x1_0 + w1_22*x2_0
        b2_1 = w1_20*x0_1 + w1_21*x1_1 + w1_22*x2_1
        b2_2 = w1_20*x0_2 + w1_21*x1_2 + w1_22*x2_2
        b2_3 = w1_20*x0_3 + w1_21*x1_3 + w1_22*x2_3

        z0_0,z0_1,z0_2,z0_3 = 1,1,1,1
        z1_0 = f(b1_0)
        z1_1 = f(b1_1)
        z1_2 = f(b1_2)
        z1_3 = f(b1_3)

        z2_0 = f(b2_0)
        z2_1 = f(b2_1)
        z2_2 = f(b2_2)
        z2_3 = f(b2_3)

        a1_0 = w2_10*z0_0 + w2_11*z1_0 + w2_12*z2_0
        a1_1 = w2_10*z0_1 + w2_11*z1_1 + w2_12*z2_1
        a1_2 = w2_10*z0_2 + w2_11*z1_2 + w2_12*z2_2
        a1_3 = w2_10*z0_3 + w2_11*z1_3 + w2_12*z2_3

        y1_0 = f(a1_0)
        y1_1 = f(a1_1)
        y1_2 = f(a1_2)
        y1_3 = f(a1_3)

        dt2_0 = (y1_0 - t1_0)*y1_0*(1-y1_0)
        dt2_1 = (y1_1 - t1_1)*y1_1*(1-y1_1)
        dt2_2 = (y1_2 - t1_2)*y1_2*(1-y1_2)
        dt2_3 = (y1_3 - t1_3)*y1_3*(1-y1_3)

        #W(2) gradient 
        DeDw2_10 = dt2_0*z0_0 + dt2_1*z0_1 + dt2_2*z0_2 + dt2_3*z0_3
        DeDw2_11 = dt2_0*z1_0 + dt2_1*z1_1 + dt2_2*z1_2 + dt2_3*z1_3
        DeDw2_12 = dt2_0*z2_0 + dt2_1*z2_1 + dt2_2*z2_2 + dt2_3*z2_3

        #W(1) gradient 
        DeDw1_10 = z1_0*(1-z1_0)*dt2_0*w2_11*x0_0 +\
        z1_1*(1-z1_1)*dt2_1*w2_11*x0_1 +\
        z1_2*(1-z1_2)*dt2_2*w2_11*x0_2 +\
        z1_3*(1-z1_3)*dt2_3*w2_11*x0_3

        DeDw1_11 = z1_0*(1-z1_0)*dt2_0*w2_11*x1_0 +\
        z1_1*(1-z1_1)*dt2_1*w2_11*x1_1 +\
        z1_2*(1-z1_2)*dt2_2*w2_11*x1_2 +\
        z1_3*(1-z1_3)*dt2_3*w2_11*x1_3

        DeDw1_12 = z1_0*(1-z1_0)*dt2_0*w2_11*x2_0 +\
        z1_1*(1-z1_1)*dt2_1*w2_11*x2_1 +\
        z1_2*(1-z1_2)*dt2_2*w2_11*x2_2 +\
        z1_3*(1-z1_3)*dt2_3*w2_11*x2_3

        DeDw1_20 = z2_0*(1-z2_0)*dt2_0*w2_12*x0_0 +\
        z2_1*(1-z2_1)*dt2_1*w2_12*x0_1 +\
        z2_2*(1-z2_2)*dt2_2*w2_12*x0_2 +\
        z2_3*(1-z2_3)*dt2_3*w2_12*x0_3

        DeDw1_21 = z2_0*(1-z2_0)*dt2_0*w2_12*x1_0 +\
        z2_1*(1-z2_1)*dt2_1*w2_12*x1_1 +\
        z2_2*(1-z2_2)*dt2_2*w2_12*x1_2 +\
        z2_3*(1-z2_3)*dt2_3*w2_12*x1_3

        DeDw1_22 = z2_0*(1-z2_0)*dt2_0*w2_12*x2_0 +\
        z2_1*(1-z2_1)*dt2_1*w2_12*x2_1 +\
        z2_2*(1-z2_2)*dt2_2*w2_12*x2_2 +\
        z2_3*(1-z2_3)*dt2_3*w2_12*x2_3

        #gradient descent

        #w(1)
        w1_10 = w1_10 - gamma*DeDw1_10
        w1_11 = w1_11 - gamma*DeDw1_11
        w1_12 = w1_12 - gamma*DeDw1_12
        w1_20 = w1_20 - gamma*DeDw1_20
        w1_21 = w1_21 - gamma*DeDw1_21
        w1_22 = w1_22 - gamma*DeDw1_22

        #w(2)
        w2_10 = w2_10 - gamma*DeDw2_10
        w2_11 = w2_11 - gamma*DeDw2_11
        w2_12 = w2_12 - gamma*DeDw2_12
        #print([y1_0,y1_1,y1_2,y1_3])
    print([y1_0,y1_1,y1_2,y1_3])


if __name__ == "__main__":
    main()