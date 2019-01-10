import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  # 下面实现了课堂上讲的公式
  for i in xrange(num_train):
    scores = X[i].dot(W)  # score of the ith sample of X
    correct_class_score = scores[y[i]]  # a scalar
    for j in xrange(num_classes):
      if j == y[i]:  # the formula that j != y_i
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin

        # detailed explaination:  https://blog.csdn.net/yc461515457/article/details/51921607
        dW[:, j] += (X[i]).T  # correspond to scores[j]
        dW[:, y[i]] -= (X[i]).T  # correspond to correct_class_score

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

  #############################################################################
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
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
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]

  scores = X.dot(W)
  # note that must use 'range(num_train)', instead of ':' or '0:500' (':' will give 500x500 output)
  correct_class_scores = scores[range(num_train), y].reshape(-1, 1)  # (N, 1) 2-d array
  margins = np.maximum(0, scores - correct_class_scores + 1)
  margins[range(num_train), y] = 0  # skip j == y[i], so set back to 0

  loss += np.sum(margins) / num_train + reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  coeff_mat = np.zeros(np.shape(margins))
  coeff_mat[margins > 0] = 1
  row_sum = np.sum(coeff_mat, axis=1)  # row_sum(1,N) coeff_mat中每一行的和
  coeff_mat[range(num_train), y] = -row_sum

  dW = (X.T).dot(coeff_mat) / num_train + 2 * reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
