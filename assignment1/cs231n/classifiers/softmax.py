import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
      Softmax loss function, naive implementation (with loops)

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: A numpy array of shape (D, C) containing weights.
      - X: A numpy array of shape (N, D) containing a minibatch of data.
      - y: A numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
      - reg: (float) regularization strength

      Returns a tuple of:
      - loss as single float
      - gradient with respect to weights W; an array of same shape as W
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
    for i in xrange(num_train):
        scores = X[i].dot(W)
        
        # avoid numerical problem when score > 0, exp(score) will very large
        scores -= np.max(scores)
        
        # unnormalized log probabilities of the classes. 
        log_prob = np.exp(scores) / np.sum(np.exp(scores))

        # softmax loss -> cross entropy between y (real) distribution and log_prob (predict) distribution
        # where y is the 'real' label distribution and
        # log_prob is the 'predict' score distribution (unnormalized log probabilities)
        class_indicat = np.zeros(num_classes)
        class_indicat[y[i]] = 1
        loss += -1* np.sum(class_indicat * np.log(log_prob))
        
        # log_prob derive by score func
        log_prob[y[i]] -= 1
        
        # derive by x using chain rule
        dW += X[i][:,np.newaxis]*log_prob
    
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    
    dW /= num_train
    dW += 2 * 0.5 * reg * W 
    
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

    num_train = X.shape[0]
    scores = X.dot(W)
    scores -= np.max(scores, axis=1)[:,np.newaxis]
    pro_scores = np.exp(scores) / np.sum(np.exp(scores),axis=1)[:,np.newaxis]
    loss = -1* np.sum(np.log(pro_scores[range(num_train),y]))
    
    pro_scores[range(num_train),y] -= 1
    dW = np.dot(X.T, pro_scores)
    
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    
    dW /= num_train
    dW += 2 * 0.5 * reg * W 
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

