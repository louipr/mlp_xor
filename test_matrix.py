import numpy as np
import matplotlib.pyplot as plt


def matrix_intro():
    Am = np.arange(10).reshape(2,5)
    print(Am)
    stdout = "Am(ri,ci) = Am(%d,%d)=%d"%(1,3,Am[1][3])
    print(stdout)

    #define simple matrix, Bm
    print("=== Creating Bm ===")
    Bm = np.array([1,2,3,4]).reshape(2,2) #turn vector to 2x2 Matrix
    print(Bm)
    stdout = "Bm(ri,ci) = Bm(%d,%d)=%d"%(1,0,Bm[1][0])
    print(stdout)

    #Element-wise multiplication
    print("\n== Bm.*Bm ==")
    print(Bm*Bm)

    #Matrix multiplication 
    print("\n== Bm x Bm ==")
    print(np.matmul(Bm,Bm))

    #define vector Cv
    Cv = np.array([5,6])
    print("\n== Cv.*Bm ==")
    print(Cv*Bm)

    #define column vector Dv 
    #Expect Dv = [5,6]^T .* [[1,2]
    #                        [3,4]]
    # = [[5,10]]
    #    [18,24]]

    print("\n== Dv ==")
    Dv = np.array([5,6]).reshape(2,1)
    print(Dv)
    print("\n== Dv*Bm")
    print(Dv*Bm)

    #apply function to matrix 
    Xm = np.copy(Bm)
    print("\n==Xm & Xm^3==")
    print(Xm)
    for x in np.nditer(Xm, op_flags=['readwrite']):
        x[...] = x**3

    print(Xm)

    #insert [1,1]
    print("\n==inserting [[1,1]]==")
    Xm =np.concatenate((np.array([[1,1]]),Xm),axis=0)
    print(Xm)

    print("\n==print first column Xm")
    print(Xm[:,[0]])
    print("\n==print last column Xm")
    print(Xm[[2],])

    print("\n==deleting first & second row==")
    Xm = np.delete(Xm,0,0)
    Xm = np.delete(Xm,0,0)
    print(Xm)

    print("\n== delete first element ==")
    Em = np.array([[1,2,3]])
    print(Em)
    print(np.delete(Em,1))

    print("\n==reshape [10,20,11,21,12,22] ==")
    Fm = np.array([10,20,11,21,12,22])
    print(Fm)
    Fm = Fm.reshape(3,2)
    print(Fm)
    Fm = np.transpose(Fm)
    print(Fm)

    print("\n==Printing dimensions==")
    print(Fm.shape)
    print(Fm.shape[1] + 1)

def main():
    matrix_intro()

if __name__ == "__main__":
    main()