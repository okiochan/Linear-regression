import data
import PCAwithRestore as pca
import numpy as np
import matplotlib.pyplot as plt

X_train, Y_train, X_control, Y_control = data.DataFactory().createData('plant_with_control')

X_train = pca.Normalize(X_train)
X_control = pca.Normalize(X_control)

D, U = pca.SingularDecomposition(X_train)

X_train_rotated = X_train.dot(U)
X_control_rotated = X_control.dot(U)

X_train_PCA = X_train_rotated[...,[0]]
X_control_PCA = X_control_rotated[...,[0]]

def RidgeRegression(X,y,C):
    l = X.shape[0]
    n = X.shape[1]

    # bias trick - concatenate ones in front of matrix
    ones = np.atleast_2d(np.ones(l)).T
    X = np.concatenate((ones,X),axis=1)

    # learn linear MNK
    res = np.linalg.inv(X.T.dot(X) + np.eye(n+1) * C).dot(X.T.dot(y))
    return res[1:(n+1)], res[0]

def LSLoss(X,y,a,b):
    l = X.shape[0]
    loss = 0
    for i in range(l):
        # just ax+ b
        loss += (a.dot(X[i,...])+b-y[i])**2
    return loss * 0.5 / l

def Work(X_train, Y_train, X_control, Y_control):
    w, w0 = RidgeRegression(X_train, Y_train, 1e-7)
    print("Regression params: ", w, w0)

    error_train = LSLoss(X_train, Y_train, w, w0)
    error_control = LSLoss(X_control, Y_control, w, w0)

    print("Error on train: ", error_train)
    print("Error on control: ", error_control)

print("Full:")
Work(X_train,Y_train,X_control,Y_control)
print("\n\nPCA:")
Work(X_train_PCA,Y_train,X_control_PCA,Y_control)
print()

a, b = RidgeRegression(X_train_PCA, Y_train, 1e-7)
plt.scatter(X_train_PCA, Y_train, s=2)
x0, y0 = -2, a*-2+b
x1, y1 = 2, a*2+b
plt.plot([x0,x1],[y0,y1],c='red')
plt.show()