import torch
import os
import random
import matplotlib.pyplot as plt
import torchvision
from torchvision.datasets import CIFAR10
import statistics
from abc import abstractmethod
from typing import Dict, List, Callable, Optional


def _extract_tensors(dset, num=None, x_dtype=torch.float32):
    """
    Extract the data and labels from a CIFAR10 dataset object and convert them to
    tensors.

    Input:
    - dset: A torchvision.datasets.CIFAR10 object
    - num: Optional. If provided, the number of samples to keep.
    - x_dtype: Optional. data type of the input image

    Returns:
    - x: `x_dtype` tensor of shape (N, 3, 32, 32)
    - y: int64 tensor of shape (N,)
    """
    x = torch.tensor(dset.data, dtype=x_dtype).permute(0, 3, 1, 2).div_(255)
    y = torch.tensor(dset.targets, dtype=torch.int64)

    if num is not None:
        if num <= 0 or num > x.shape[0]:
            raise ValueError("Invalid value num=%d; must be in the range [0, %d]" % (num, x.shape[0]))
        x = x[:num].clone()
        y = y[:num].clone()

    return x, y

def cifar10(num_train=None, num_test=None, x_dtype=torch.float32):
    """
    Return the CIFAR10 dataset, automatically downloading it if necessary.
    This function can also subsample the dataset.

    Inputs:
    - num_train: [Optional] How many samples to keep from the training set.
      If not provided, then keep the entire training set.
    - num_test: [Optional] How many samples to keep from the test set.
      If not provided, then keep the entire test set.
    - x_dtype: [Optional] Data type of the input image

    Returns:
    - x_train: `x_dtype` tensor of shape (num_train, 3, 32, 32)
    - y_train: int64 tensor of shape (num_train, 3, 32, 32)
    - x_test: `x_dtype` tensor of shape (num_test, 3, 32, 32)
    - y_test: int64 tensor of shape (num_test, 3, 32, 32)
    """
    
    download = not os.path.isdir("cifar-10-batches-py")
    dset_train = CIFAR10(root="../data", download=download, train=True)
    dset_test = CIFAR10(root="../data", train=False)

    x_train, y_train = _extract_tensors(dset_train, num_train, x_dtype)
    x_test, y_test = _extract_tensors(dset_test, num_test, x_dtype)

    return x_train, y_train, x_test, y_test

def tensor_to_image(tensor):
    """
    Convert a torch tensor into a numpy ndarray for visualization.

    Inputs:
    - tensor: A torch tensor of shape (3, H, W) with elements in the range [0, 1]

    Returns:
    - ndarr: A uint8 numpy array of shape (H, W, 3)
    """
    tensor = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    ndarr = tensor.to("cpu", torch.uint8).numpy()
    return ndarr

def preprocess_cifar10(
    cuda=True,
    show_examples=True,
    bias_trick=False,
    flatten=True,
    validation_ratio=0.2,
    dtype=torch.float32,
):
    """
    Returns a preprocessed version of the CIFAR10 dataset, automatically
    downloading if necessary. We perform the following steps:

    (0) [Optional] Visualize some images from the dataset
    (1) Normalize the data by subtracting the mean
    (2) Reshape each image of shape (3, 32, 32) into a vector of shape (3072,)
    (3) [Optional] Bias trick: add an extra dimension of ones to the data
    (4) Carve out a validation set from the training set

    Inputs:
    - cuda: If true, move the entire dataset to the GPU
    - validation_ratio: Float in the range (0, 1) giving the fraction of the train
      set to reserve for validation
    - bias_trick: Boolean telling whether or not to apply the bias trick
    - show_examples: Boolean telling whether or not to visualize data samples
    - dtype: Optional, data type of the input image X

    Returns a dictionary with the following keys:
    - 'X_train': `dtype` tensor of shape (N_train, D) giving training images
    - 'X_val': `dtype` tensor of shape (N_val, D) giving val images
    - 'X_test': `dtype` tensor of shape (N_test, D) giving test images
    - 'y_train': int64 tensor of shape (N_train,) giving training labels
    - 'y_val': int64 tensor of shape (N_val,) giving val labels
    - 'y_test': int64 tensor of shape (N_test,) giving test labels

    N_train, N_val, and N_test are the number of examples in the train, val, and
    test sets respectively. The precise values of N_train and N_val are determined
    by the input parameter validation_ratio. D is the dimension of the image data;
    if bias_trick is False, then D = 32 * 32 * 3 = 3072;
    if bias_trick is True then D = 1 + 32 * 32 * 3 = 3073.
    """
    X_train, y_train, X_test, y_test = cifar10(x_dtype=dtype)

    # Move data to the GPU
    if cuda:
        X_train = X_train.cuda()
        y_train = y_train.cuda()
        X_test = X_test.cuda()
        y_test = y_test.cuda()

    # 0. Visualize some examples from the dataset.
    if show_examples:
        classes = [
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        samples_per_class = 12
        samples = []

        random.seed(0)
        torch.manual_seed(0)

        for y, cls in enumerate(classes):
            plt.text(-4, 34 * y + 18, cls, ha="right")
            (idxs,) = (y_train == y).nonzero(as_tuple=True)
            for i in range(samples_per_class):
                idx = idxs[random.randrange(idxs.shape[0])].item()
                samples.append(X_train[idx])
        img = torchvision.utils.make_grid(samples, nrow=samples_per_class)
        plt.imshow(tensor_to_image(img))
        plt.axis("off")
        plt.show()

    # 1. Normalize the data: subtract the mean RGB (zero mean)
    mean_image = X_train.mean(dim=(0, 2, 3), keepdim=True)
    X_train -= mean_image
    X_test -= mean_image

    # 2. Reshape the image data into rows
    if flatten:
      X_train = X_train.reshape(X_train.shape[0], -1)
      X_test = X_test.reshape(X_test.shape[0], -1)

    # 3. Add bias dimension and transform into columns
    if bias_trick:
        ones_train = torch.ones(X_train.shape[0], 1, device=X_train.device)
        X_train = torch.cat([X_train, ones_train], dim=1)
        ones_test = torch.ones(X_test.shape[0], 1, device=X_test.device)
        X_test = torch.cat([X_test, ones_test], dim=1)

    # 4. take the validation set from the training set
    # Note: It should not be taken from the test set
    num_training = int(X_train.shape[0] * (1.0 - validation_ratio))
    num_validation = X_train.shape[0] - num_training

    # return the dataset
    data_dict = {}
    data_dict["X_val"] = X_train[num_training : num_training + num_validation]
    data_dict["y_val"] = y_train[num_training : num_training + num_validation]
    data_dict["X_train"] = X_train[0:num_training]
    data_dict["y_train"] = y_train[0:num_training]

    data_dict["X_test"] = X_test
    data_dict["y_test"] = y_test

    return data_dict

def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-7):
    """
    Utility function to perform numeric gradient checking. We use the centered
    difference formula to compute a numeric derivative:

    f'(x) =~ (f(x + h) - f(x - h)) / (2h)

    Rather than computing a full numeric gradient, we sparsely sample a few
    dimensions along which to compute numeric derivatives.

    Inputs:
    - f: A function that inputs a torch tensor and returns a torch scalar
    - x: A torch tensor of the point at which to evaluate the numeric gradient
    - analytic_grad: A torch tensor giving the analytic gradient of f at x
    - num_checks: The number of dimensions along which to check
    - h: Step size for computing numeric derivatives
    """

    for i in range(num_checks):
        ix = tuple([random.randrange(m) for m in x.shape])

        oldval = x[ix].item()
        x[ix] = oldval + h  # increment by h
        fxph = f(x).item()  # evaluate f(x + h)
        x[ix] = oldval - h  # increment by h
        fxmh = f(x).item()  # evaluate f(x - h)
        x[ix] = oldval  # reset

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error_top = abs(grad_numerical - grad_analytic)
        rel_error_bot = abs(grad_numerical) + abs(grad_analytic) + 1e-12
        rel_error = rel_error_top / rel_error_bot
        msg = "numerical: %f analytic: %f, relative error: %e"
        print(msg % (grad_numerical, grad_analytic, rel_error))


#--------------------------------------------------------------------
class LinearClassifier:
    """An abstarct class for the linear classifiers"""

    def __init__(self):
        random.seed(0)
        torch.manual_seed(0)
        self.W = None

    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        learning_rate: float = 1e-3,
        reg: float = 1e-5,
        num_iters: int = 100,
        batch_size: int = 200,
        verbose: bool = False,
    ):
        train_args = (
            self.loss,
            self.W,
            X_train,
            y_train,
            learning_rate,
            reg,
            num_iters,
            batch_size,
            verbose,
        )
        self.W, loss_history = train_linear_classifier(*train_args)
        return loss_history

    def predict(self, X: torch.Tensor):
        return predict_linear_classifier(self.W, X)

    @abstractmethod
    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - W: A PyTorch tensor of shape (D, C) containing (trained) weight of a model.
        - X_batch: A PyTorch tensor of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A PyTorch tensor of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an tensor of the same shape as W
        """
        raise NotImplementedError

    def _loss(self, X_batch: torch.Tensor, y_batch: torch.Tensor, reg: float):
        self.loss(self.W, X_batch, y_batch, reg)

    def save(self, path: str):
        torch.save({"W": self.W}, path)
        print("Saved in {}".format(path))

    def load(self, path: str):
        W_dict = torch.load(path, map_location="cpu")
        self.W = W_dict["W"]
        if self.W is None:
            raise Exception("Failed to load your checkpoint")

class LinearSVM(LinearClassifier):
    """A subclass that uses the Multiclass SVM loss function"""

    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ):
        return svm_loss_vectorized(W, X_batch, y_batch, reg)
    
class Softmax(LinearClassifier):
    """A subclass that uses the Softmax + Cross-entropy loss function"""

    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ):
        return softmax_loss_vectorized(W, X_batch, y_batch, reg)
    
# **************************************************#
################## Section 1: SVM ##################
# **************************************************#

def svm_loss_naive(W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing weights.
    - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as torch scalar
    - gradient of loss with respect to weights W; a tensor of same shape as W
    """
    dW = torch.zeros_like(W)  

    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    for i in range(num_train):
        scores = W.t().mv(X[i])
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                
                dW[:,j]+=X[i]
                dW[:,y[i]]-=X[i]
    
    loss /= num_train
    loss += reg * torch.sum(W * W)

    dW/=num_train
    dW+=2*reg*W

    return loss,dW
    
def svm_loss_vectorized(W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float):
    """
    Structured SVM loss function, vectorized implementation. 
    The inputs and outputs are the same as svm_loss_naive.

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing weights.
    - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as torch scalar
    - gradient of loss with respect to weights W; a tensor of same shape as W
    """
    loss = 0.0
    dW = torch.zeros_like(W)  

    num_train=X.shape[0]
    scores=X.mm(W)
    right_class_score=scores[torch.arange(num_train), y]
    margins=scores-right_class_score.view(-1,1)+1
    margins[torch.arange(num_train), y]=0
    margins[margins<0]=0
    loss=margins.sum()/num_train+reg*torch.sum(W*W)
    
    '''margins[i,j]>0 <=> dW[:,j]+=X[i] , dW[:,y[i]]-=X[i]'''
    margins[margins>0]=1
    margins[torch.arange(num_train),y]=torch.sum(margins,dim=1)*(-1)

    dW=margins.t().mm(X).t()
    dW=dW/num_train+2*reg*W

    return loss, dW    

def sample_batch(X: torch.Tensor, y: torch.Tensor, num_train: int, batch_size: int):
    """
    Sample batch_size elements from the training data and their
    corresponding labels to use in this round of gradient descent.
    """
    X_batch = None
    y_batch = None
    
    indices=torch.randint(num_train,(batch_size,))
    X_batch=X[indices]
    y_batch=y[indices]
    
    return X_batch, y_batch

def train_linear_classifier(
    loss_func: Callable,
    W: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    learning_rate: float = 1e-3,
    reg: float = 1e-5,
    num_iters: int = 100,
    batch_size: int = 200,
    verbose: bool = False,
):
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - loss_func: loss function to use when training. It should take W, X, y
      and reg as input, and output a tuple of (loss, dW)
    - W: A PyTorch tensor of shape (D, C) giving the initial weights of the
      classifier. If W is None then it will be initialized here.
    - X: A PyTorch tensor of shape (N, D) containing training data; there are N
      training samples each of dimension D.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c
      means that X[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Returns: A tuple of:
    - W: The final value of the weight matrix and the end of optimization
    - loss_history: A list of Python scalars giving the values of the loss at each
      training iteration.
    """
    # assume y takes values 0...K-1 where K is number of classes
    num_train, dim = X.shape
    if W is None:
        # lazily initialize W
        num_classes = torch.max(y) + 1
        W = 0.000001 * torch.randn(dim, num_classes, device=X.device, dtype=X.dtype)
    else:
        num_classes = W.shape[1]

    # Run stochastic gradient descent to optimize W
    loss_history = []
    for it in range(num_iters):
        X_batch, y_batch = sample_batch(X, y, num_train, batch_size)

        # evaluate loss and gradient
        loss, grad = loss_func(W, X_batch, y_batch, reg)
        loss_history.append(loss.item())

        W-=learning_rate*grad

        if verbose and it % 100 == 0:
            print("iteration %d / %d: loss %f" % (it, num_iters, loss))

    return W, loss_history

def predict_linear_classifier(W: torch.Tensor, X: torch.Tensor):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - W: A PyTorch tensor of shape (D, C), containing weights of a model
    - X: A PyTorch tensor of shape (N, D) containing training data; there are N
      training samples each of dimension D.

    Returns:
    - y_pred: PyTorch int64 tensor of shape (N,) giving predicted labels for each
      elemment of X. Each element of y_pred should be between 0 and C - 1.
    """
    y_pred = torch.zeros(X.shape[0], dtype=torch.int64)

    scores=X.mm(W)
    _,y_pred=torch.max(scores,dim=1)
    
    return y_pred

def svm_get_search_params():
    """
    Return candidate hyperparameters for the SVM model. 

    Returns:
    - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
    - regularization_strengths: regularization strengths candidates
                                e.g. [1e0, 1e1, ...]
    """

    learning_rates=[1e-3,7e-3,5e-3,1e-2]
    regularization_strengths=[0.01,0.1,0.5]

    return learning_rates, regularization_strengths

def test_one_param_set(
    cls: LinearClassifier,
    data_dict: Dict[str, torch.Tensor],
    lr: float,
    reg: float,
    num_iters: int = 2000,
):
    """
    Train a single LinearClassifier instance and return the learned instance
    with train/val accuracy.

    Inputs:
    - cls (LinearClassifier): a newly-created LinearClassifier instance.
                              Train/Validation should perform over this instance
    - data_dict (dict): a dictionary that includes
                        ['X_train', 'y_train', 'X_val', 'y_val']
                        as the keys for training a classifier
    - lr (float): learning rate parameter for training a SVM instance.
    - reg (float): a regularization weight for training a SVM instance.
    - num_iters (int, optional): a number of iterations to train

    Returns:
    - cls (LinearClassifier): a trained LinearClassifier instances with
                              (['X_train', 'y_train'], lr, reg)
                              for num_iter times.
    - train_acc (float): training accuracy of the svm_model
    - val_acc (float): validation accuracy of the svm_model
    """
    
    cls.train(data_dict['X_train'], data_dict['y_train'], lr, reg,num_iters,200,False)
    y_train_pred=cls.predict(data_dict['X_train'])
    train_acc=((y_train_pred==data_dict['y_train']).double().mean().item())*100.0

    y_val_pred=cls.predict(data_dict['X_val'])
    val_acc=((y_val_pred==data_dict['y_val']).double().mean().item())*100.0

    return cls, train_acc, val_acc


# **************************************************#
################ Section 2: Softmax ################
# **************************************************#


def softmax_loss_naive(W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float):
    """
    Softmax loss function, naive implementation (with loops). 

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing weights.
    - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; a tensor of same shape as W
    """
    loss = 0.0
    dW = torch.zeros_like(W)

    num_train=X.shape[0]
    num_classes=W.shape[1]

    for i in range(num_train) :
        pred=W.t().mv(X[i])
        norm=pred-pred.max()
        exp_score=torch.exp(norm)
        scores=exp_score/exp_score.sum()
        loss+=-torch.log(scores[[y[i]]])

        for j in range(num_classes) :
            if j==y[i] :
                dW[:,j]+=(scores[j]-1)*X[i]
            else :
                dW[:,j]+=scores[j]*X[i]

    loss/=num_train
    loss+=reg*torch.sum(W*W)

    dW/=num_train
    dW+=2*reg*W
    
    return loss, dW


def softmax_loss_vectorized(W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float):
    """
    Softmax loss function, vectorized version. 

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = torch.zeros_like(W)

    num_classes=W.shape[1]
    num_train=X.shape[0]

    preds=X.mm(W)
    max_preds,_=preds.max(dim=1,keepdim=True)
    norm=preds-max_preds
    exp_score=torch.exp(norm)
    scores=exp_score/exp_score.sum(dim=1,keepdim=True)
    loss=-torch.log(scores[torch.arange(num_train),y]).sum()
    loss/=num_train
    loss+=reg*torch.sum(W*W)

    scores[torch.arange(num_train),y]-=1
    dW=X.t().mm(scores)
    dW/=num_train
    dW+=2*reg*W

    return loss, dW


def softmax_get_search_params():
    """
    Return candidate hyperparameters for the Softmax model. 

    Returns:
    - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
    - regularization_strengths: regularization strengths candidates
                                e.g. [1e0, 1e1, ...]
    """
    
    learning_rates = [1e-2,5e-2,7e-2,1e-1]
    regularization_strengths = [0.01,0.05]
    
    return learning_rates, regularization_strengths