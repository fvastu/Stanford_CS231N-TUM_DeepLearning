"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def cross_entropoy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

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
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    trainNumber = X.shape[0]
    classNumber = dW.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    for i in range(trainNumber):
        #Copy of data in a numpy vector
        x_i =np.copy(X[i,:])
        W=np.copy(W)
        #Two transpose in order to match the shape of the variables
        theta_xi = np.array(W.transpose().dot(x_i.transpose()))
        #calculate the maximum for the numerical stability
        maxThetai = -theta_xi.max()
        exp_theta_xi = np.exp(theta_xi+maxThetai)
        sum_exp_theta_xi = np.sum(exp_theta_xi, axis = 0)
        for j in range(classNumber):
            if j == y[i]:
                #gradient wrt weights
                dW[:,j] += -x_i.transpose() + (exp_theta_xi[j] / sum_exp_theta_xi) * x_i.transpose()
            else:
                dW[:,j] += (exp_theta_xi[j] / sum_exp_theta_xi) * x_i.transpose()
        numerator = np.exp(theta_xi[y[i]]+maxThetai)
        denominator = np.sum(np.exp(theta_xi+maxThetai),axis = 0)
        loss += -np.log(numerator / float(denominator))
        #Add a regularization term in order to prevent overfitting
    loss = loss / float(trainNumber) + 0.5 * reg * np.sum(W*W)
    dW = dW / float(trainNumber) + reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def cross_entropoy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropoy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    trainNumber = X.shape[0]
    classNumber = W.shape[1]
    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    X=np.copy(X)
    W=np.copy(W)
    theta_xi = W.transpose().dot(X.transpose())
    theta_xi = theta_xi- np.max(theta_xi , axis=0)
    exp_theta_xi = np.exp(theta_xi)
    sum_exp_theta_xi = np.sum(exp_theta_xi , axis = 0)
    loss = np.log(sum_exp_theta_xi)
    loss = loss - theta_xi[y,np.arange(trainNumber)]
    loss = np.sum(loss) / float(trainNumber) + 0.5 * reg * np.sum(W*W)
    Grad= exp_theta_xi/ sum_exp_theta_xi
    Grad[y,np.arange(trainNumber)] += -1.0
    dW = (Grad.dot(X)).transpose()/ float(trainNumber) + reg*W
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW



class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropoy_loss_vectorized(self.W, X_batch, y_batch, reg)
