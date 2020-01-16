import numpy as np


'''
Code out K Nearest Neighbors algorithm for regression
    1. Find distance between each observation pair
    2. Select k closest observations
    3. Prediction = mean of responses of k closest observations
O(n ** 2)
'''


def knn(X, y, k, distance='euclidean'):
    '''
    Inputs:
        X = N x M 2D array
        y = 1D array of length N
        k = number of nearest neighbors, integer >= 1
        distance = only Euclidean distance for now
    Outputs:
        N x 1 matrix of predictions
    '''
    N, M = X.shape

    dist = np.zeros((N, N))

    for n in range(N):
        if distance == 'euclidean':
            diff = X - X[n]
            dist[n] = np.sqrt((diff * diff).sum(axis=1))
        else:
            raise AttributeError('Only euclidean distance supported!')

    idx = np.argsort(dist, axis=1)

    preds = np.array([y[idx[i][1:(k + 1)]].mean() for i in range(N)])

    return(preds)
