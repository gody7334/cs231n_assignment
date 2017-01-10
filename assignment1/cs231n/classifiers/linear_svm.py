import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):

    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in xrange(num_classes):
          if j == y[i]:
            continue
          margin = scores[j] - correct_class_score + 1 # note delta = 1
          if margin > 0:
            loss += margin

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

################### Hint ##############################################
#    scores vector s
#    loop through scores, count wrong boundary sum(1(wjxi - wyixi > 0)) for ith model
#    compute (1(wjxi-wuixi>0) for rest of the models
#    s * x get dW matrix
#######################################################################

################### someone's example for debugging #######################
#     # initialize the gradient as zero
#     dW = np.zeros(W.shape) 
#
#     # compute the loss and the gradient
#     num_classes = W.shape[1]
#     num_train = X.shape[0]
#     loss = 0.0
#     for i in xrange(num_train):
#         scores = X[i].dot(W)
#         correct_class_score = scores[y[i]]
#         for j in xrange(num_classes):
#           if j == y[i]:
#             if margin > 0:
#                 continue
#           margin = scores[j] - correct_class_score + 1 # note delta = 1
#           if margin > 0:
#             dW[:, y[i]] += -X[i]
#             dW[:, j] += X[i] # gradient update for incorrect rows
#             loss += margin
##########################################################################
    
    num_classes = W.shape[1]
    num_train = X.shape[0]
    dW = np.zeros(W.shape)
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        scores_diff = scores - correct_class_score + 1
        
        # get the differernt between correct class score and wrong class score
        diff = np.where(scores_diff > 0, 1, 0)
        
        # sum(diff) - 1, remove correct class model itself as 0+1 always > 0
        diff[y[i]] = (np.sum(diff)-1) * (-1)
        
        X_temp = X[i][:,np.newaxis]
        dW += (diff * X_temp)
        
    dW /= num_train
    
    # L2 regularization gradient
    dW += 2 * 0.5 * reg * W 
    
    #############################################################################
    #                       END OF YOUR CODE                                    #
    #############################################################################
    
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
