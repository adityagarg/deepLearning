import time
import numpy as np

def pca_naive(X, K):
    """
    PCA -- naive version

    Inputs:
    - X: (float) A numpy array of shape (N, D) where N is the number of samples,
         D is the number of features
    - K: (int) indicates the number of features you are going to keep after
         dimensionality reduction

    Returns a tuple of:
    - P: (float) A numpy array of shape (K, D), representing the top K
         principal components
    - T: (float) A numpy vector of length K, showing the score of each
         component vector
    """
    N=X.shape[0]
    cov=(X.T.dot(X))/N

    eVals, eVects=np.linalg.eig(cov)

    topK=np.argsort(eVals)[::-1][:K]


    T=eVals[topK]
    P=eVects[:, topK].T
    # P=X.T.dot(V)

    # print(eVects==eVects.T)
    # print("eVals-", eVals.shape)
    # print(eVals[:10])
    # print("eVects-", eVects.shape)


    ###############################################
    #TODO: Implement PCA by extracting eigenvector#
    ###############################################


    ###############################################
    #              End of your code               #
    ###############################################
    # return None
    return (P, T)
