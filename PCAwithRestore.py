import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

def Normalize(X):
    X = X-X.mean(axis=0)
    return X

def ShowPercentage(X):
    L, U = SingularDecomposition(X)
    L /= np.sum(L)
    for i in range(1,L.size,1):
        L[i] += L[i-1]
    print([L[i] for i in range(0,L.size,1)])

def GetComponents(X, m):
    L, U = SingularDecomposition(X)
    G = X.dot(U)
    return G[:,np.arange(m)]

def SingularDecomposition(X):
    l = X.shape[0]
    n = X.shape[1]
    Xcov = np.dot(X.T,X)/l
    L, U = np.linalg.eigh(Xcov)
    reverse = n - 1 - np.arange(n)
    return L[reverse], U[:,reverse]

def RestoreOriginal(G, X):
    L, U = SingularDecomposition(X)
    n = U.shape[1]
    m = G.shape[1]
    l = G.shape[0]
    zeros = np.zeros((l, n-m))
    G = np.hstack((G,zeros))
    return G.dot(U.T)

def QwadraticError(X, count):
    G = GetComponents(X, count)
    Xrestored = RestoreOriginal(G, X)
    return np.sum((X - Xrestored) ** 2)/ X.shape[0] / X.shape[1]
