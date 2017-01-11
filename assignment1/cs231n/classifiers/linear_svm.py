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
        
        # get the differernt between correct class score and wrong class score
        diff = np.where(scores - correct_class_score + 1 > 0, 1, 0)
        
        # remove correct class model itself (0+1)
        diff[y[i]] = 0
        
        # calculate right class model's gradient
        diff[y[i]] = (np.sum(diff)) * (-1)
        
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
    num_train = X.shape[0]
    
    # compute scores
    scores = X.dot(W)
    
    # create 1 dim index 1...n
    train_index = np.arange(num_train)
    
    # select correct class scores in 2d array
    correct_class_score = scores[train_index, y]
    
    # compute unhappiness for each model
    score_diff = scores - correct_class_score[:,np.newaxis]+1
    
    # filter > 0 but keep dimension
    # score_diff_filter = np.where(score_diff>0, score_diff, 0)
    
    # or use maximum which is a element wise operation function, it matches the equation better
    score_diff_filter = np.maximum(score_diff, 0)
    
    # remove correct class model itself (0+1)
    score_diff_filter[train_index,y] = 0
    
    # alculate right class model's loss
    loss = np.sum(score_diff_filter)
    
    # scale and regularize
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    
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
    
    # wrong class gradient
    score_diff_filter = np.where(score_diff > 0, 1, 0)
    # remove right class result
    score_diff_filter[train_index,y] = 0
    # calculate right class gradient
    score_diff_filter[train_index,y] = np.sum(score_diff_filter, axis=1)*-1
    
    # sum the gradient across dataset
    # 3 dim broadcasting seems to be slow as it create high dim intermediate tensor
    # dW = np.sum(X[:,:,np.newaxis] * score_diff_filter[:,np.newaxis,:], axis = 0) 
    
    # inner dot is fast
    dW = X.T.dot(score_diff_filter)

    # scale and regularization
    dW /= num_train
    dW += 2 * 0.5 * reg * W 
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
