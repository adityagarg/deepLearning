import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
      Softmax loss function, naive implementation (with loops)

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
      - reg: (float) regularization strength

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wst W, an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)



    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):
      probs=X[i, :].dot(W)
      probs-=np.max(probs)

      probs=np.exp(probs)/np.sum(np.exp(probs))
      loss-=np.log(probs[y[i]])

      probs[y[i]]-=1

      for j in range(num_classes):
          dW[:, j]+=X[i, :]*probs[j]



    loss=loss/num_train
    loss+=0.5*reg*np.sum(W*W)

    dW=dW/num_train
    dW+=reg*W





    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)



    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # print("X-",X.shape)
    # print("y-",y.shape)
    # print("W-",W.shape)

    
    num_classes = W.shape[1]
    num_train = X.shape[0]
    
  
    x=np.dot(X, W)
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    print(probs.shape)
    loss = -np.sum(np.log(probs[range(num_train), y])) / num_train
    regLoss=0.5*reg*np.sum(W*W)
    loss+=regLoss

  
    probs[range(num_train), y] -= 1
    dW=X.T.dot(probs)
    dW /= num_train
    dW+=reg*W




    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    return loss, dW

