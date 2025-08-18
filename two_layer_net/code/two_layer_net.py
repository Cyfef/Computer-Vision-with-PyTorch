import torch
import random
import statistics
import os
import math
import matplotlib.pyplot as plt
import torchvision
from torchvision.datasets import CIFAR10
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

def plot_stats(stat_dict):
    # Plot the loss function and train / validation accuracies
    plt.subplot(1, 2, 1)
    plt.plot(stat_dict["loss_history"], "o")
    plt.title("Loss history")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(stat_dict["train_acc_history"], "o-", label="train")
    plt.plot(stat_dict["val_acc_history"], "o-", label="val")
    plt.title("Classification accuracy history")
    plt.xlabel("Epoch")
    plt.ylabel("Clasification accuracy")
    plt.legend()

    plt.gcf().set_size_inches(14, 4)
    plt.show()


def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    Reshape a 4D tensor of image data to a grid for easy visualization.

    Inputs:
    - Xs: Data of shape (N, H, W, C)
    - ubound: Output grid will have values scaled to the range [0, ubound]
    - padding: The number of blank pixels between elements of the grid
    """
    (N, H, W, C) = Xs.shape
    # print(Xs.shape)
    grid_size = int(math.ceil(math.sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = torch.zeros((grid_height, grid_width, C), device=Xs.device)
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = torch.min(img), torch.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    return grid


def show_net_weights(net):
    W1 = net.params["W1"]
    W1 = W1.reshape(3, 32, 32, -1).transpose(0, 3)
    plt.imshow(visualize_grid(W1, padding=3).type(torch.uint8).cpu())
    plt.gca().axis("off")
    plt.show()

def plot_acc_curves(stat_dict):
    plt.subplot(1, 2, 1)
    for key, single_stats in stat_dict.items():
        plt.plot(single_stats["train_acc_history"], label=str(key))
    plt.title("Train accuracy history")
    plt.xlabel("Epoch")
    plt.ylabel("Clasification accuracy")

    plt.subplot(1, 2, 2)
    for key, single_stats in stat_dict.items():
        plt.plot(single_stats["val_acc_history"], label=str(key))
    plt.title("Validation accuracy history")
    plt.xlabel("Epoch")
    plt.ylabel("Clasification accuracy")
    plt.legend()

    plt.gcf().set_size_inches(14, 5)
    plt.show()


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


def get_toy_data(
    num_inputs=5,
    input_size=4,
    hidden_size=10,
    num_classes=3,
    dtype=torch.float32,
    device="cuda",
):
    """
    Get toy data for use when developing a two-layer-net.

    Inputs:
    - num_inputs: Integer N giving the data set size
    - input_size: Integer D giving the dimension of input data
    - hidden_size: Integer H giving the number of hidden units in the model
    - num_classes: Integer C giving the number of categories
    - dtype: torch datatype for all returned data
    - device: device on which the output tensors will reside

    Returns a tuple of:
    - toy_X: `dtype` tensor of shape (N, D) giving data points
    - toy_y: int64 tensor of shape (N,) giving labels, where each element is an
      integer in the range [0, C)
    - params: A dictionary of toy model parameters, with keys:
      - 'W1': `dtype` tensor of shape (D, H) giving first-layer weights
      - 'b1': `dtype` tensor of shape (H,) giving first-layer biases
      - 'W2': `dtype` tensor of shape (H, C) giving second-layer weights
      - 'b2': `dtype` tensor of shape (C,) giving second-layer biases
    """
    N = num_inputs
    D = input_size
    H = hidden_size
    C = num_classes

    # We set the random seed for repeatable experiments.
    random.seed(0)
    torch.manual_seed(0)

    # Generate some random parameters, storing them in a dict
    params = {}
    params["W1"] = 1e-4 * torch.randn(D, H, device=device, dtype=dtype)
    params["b1"] = torch.zeros(H, device=device, dtype=dtype)
    params["W2"] = 1e-4 * torch.randn(H, C, device=device, dtype=dtype)
    params["b2"] = torch.zeros(C, device=device, dtype=dtype)

    # Generate some random inputs and labels
    toy_X = 10.0 * torch.randn(N, D, device=device, dtype=dtype)
    toy_y = torch.tensor([0, 1, 2, 2, 1], device=device, dtype=torch.int64)

    return toy_X, toy_y, params

def compute_numeric_gradient(f, x, dLdf=None, h=1e-7):
    """
    Compute the numeric gradient of f at x using a finite differences
    approximation. We use the centered difference:

    df    f(x + h) - f(x - h)
    -- ~= -------------------
    dx           2 * h

    Function can also expand this easily to intermediate layers using the
    chain rule:

    dL   df   dL
    -- = -- * --
    dx   dx   df

    Inputs:
    - f: A function that inputs a torch tensor and returns a torch scalar
    - x: A torch tensor giving the point at which to compute the gradient
    - dLdf: optional upstream gradient for intermediate layers
    - h: epsilon used in the finite difference calculation
    Returns:
    - grad: A tensor of the same shape as x giving the gradient of f at x
    """
    flat_x = x.contiguous().flatten()
    grad = torch.zeros_like(x)
    flat_grad = grad.flatten()

    # Initialize upstream gradient to be ones if not provide
    if dLdf is None:
        y = f(x)
        dLdf = torch.ones_like(y)
    dLdf = dLdf.flatten()

    # iterate over all indexes in x
    for i in range(flat_x.shape[0]):
        oldval = flat_x[i].item()  # Store the original value
        flat_x[i] = oldval + h  # Increment by h
        fxph = f(x).flatten()  # Evaluate f(x + h)
        flat_x[i] = oldval - h  # Decrement by h
        fxmh = f(x).flatten()  # Evaluate f(x - h)
        flat_x[i] = oldval  # Restore original value

        # compute the partial derivative with centered formula
        dfdxi = (fxph - fxmh) / (2 * h)

        # use chain rule to compute dLdx
        flat_grad[i] = dLdf.dot(dfdxi).item()

    # Note that since flat_grad was only a reference to grad,
    # we can just return the object in the shape of x by returning grad
    return grad


def rel_error(x, y, eps=1e-10):
    """
    Compute the relative error between a pair of tensors x and y,
    which is defined as:

                            max_i |x_i - y_i]|
    rel_error(x, y) = -------------------------------
                      max_i |x_i| + max_i |y_i| + eps

    Inputs:
    - x, y: Tensors of the same shape
    - eps: Small positive constant for numeric stability

    Returns:
    - rel_error: Scalar giving the relative error between x and y
    """
    """ returns relative error between x and y """
    top = (x - y).abs().max().item()
    bot = (x.abs() + y.abs()).clamp(min=eps).max().item()
    return top / bot

#-------------------------------------------------------------------

class TwoLayerNet(object):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
        std: float = 1e-4,
    ):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        - dtype: Optional, data type of each initial weight params
        - device: Optional, whether the weight params is on GPU or CPU
        - std: Optional, initial weight scaler.
        """
        # reset seed before start
        random.seed(0)
        torch.manual_seed(0)

        self.params = {}
        self.params["W1"] = std * torch.randn(input_size, hidden_size, dtype=dtype, device=device)
        self.params["b1"] = torch.zeros(hidden_size, dtype=dtype, device=device)
        self.params["W2"] = std * torch.randn(hidden_size, output_size, dtype=dtype, device=device)
        self.params["b2"] = torch.zeros(output_size, dtype=dtype, device=device)

    def loss(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        reg: float = 0.0,
    ):
        return nn_forward_backward(self.params, X, y, reg)

    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        learning_rate: float = 1e-3,
        learning_rate_decay: float = 0.95,
        reg: float = 5e-6,
        num_iters: int = 100,
        batch_size: int = 200,
        verbose: bool = False,
    ):
        # fmt: off
        return nn_train(
            self.params, nn_forward_backward, nn_predict, X, y,
            X_val, y_val, learning_rate, learning_rate_decay,
            reg, num_iters, batch_size, verbose,
        )
        # fmt: on

    def predict(self, X: torch.Tensor):
        return nn_predict(self.params, nn_forward_backward, X)

    def save(self, path: str):
        torch.save(self.params, path)
        print("Saved in {}".format(path))

    def load(self, path: str):
        checkpoint = torch.load(path, map_location="cpu")
        self.params = checkpoint
        if len(self.params) != 4:
            raise Exception("Failed to load your checkpoint")

        for param in ["W1", "b1", "W2", "b2"]:
            if param not in self.params:
                raise Exception("Failed to load your checkpoint")
        # print("load checkpoint file: {}".format(path))


def nn_forward_pass(params: Dict[str, torch.Tensor], X: torch.Tensor):
    """
    The first stage of our neural network implementation: Run the forward pass
    of the network to compute the hidden layer features and classification
    scores. The network architecture should be:

    FC layer -> ReLU (hidden) -> FC layer (scores)

    Inputs:
    - params: a dictionary of PyTorch Tensor that store the weights of a model.
      It should have following keys with shape
          W1: First layer weights; has shape (D, H)
          b1: First layer biases; has shape (H,)
          W2: Second layer weights; has shape (H, C)
          b2: Second layer biases; has shape (C,)
    - X: Input data of shape (N, D). Each X[i] is a training sample.

    Returns a tuple of:
    - scores: Tensor of shape (N, C) giving the classification scores for X
    - hidden: Tensor of shape (N, H) giving the hidden layer representation
      for each input value (after the ReLU).
    """
    # Unpack variables from the params dictionary
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]
    N, D = X.shape

    # Compute the forward pass
    hidden=X.mm(W1)+b1
    hidden[hidden<0]=0
    scores=hidden.mm(W2)+b2

    return scores, hidden


def nn_forward_backward(
    params: Dict[str, torch.Tensor],
    X: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    reg: float = 0.0
):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network. 

    Inputs: First two parameters (params, X) are same as nn_forward_pass
    - params: a dictionary of PyTorch Tensor that store the weights of a model.
      It should have following keys with shape
          W1: First layer weights; has shape (D, H)
          b1: First layer biases; has shape (H,)
          W2: Second layer weights; has shape (H, C)
          b2: Second layer biases; has shape (C,)
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a tensor scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]
    N, D = X.shape

    scores, h1 = nn_forward_pass(params, X)
    # If the targets are not given then jump out, we're done
    if y is None:
        return scores

    # Compute the loss
    max_score,_=scores.max(dim=1, keepdim=True)
    norm=scores-max_score
    exp_scores=torch.exp(norm)
    probs=exp_scores/exp_scores.sum(dim=1, keepdim=True)
    loss=-torch.log(probs[torch.arange(N), y]).sum()
    loss=loss/N
    loss+=reg*torch.sum(W1*W1)+reg*torch.sum(W2*W2)

    # Backward pass: compute gradients
    grads = {}

    dL_ds=probs.clone()
    dL_ds[torch.arange(N), y]-=1
    dL_ds/=N

    grads['b2']=dL_ds.sum(dim=0)
    grads['W2']=h1.t().mm(dL_ds)+2*reg*W2

    dL_dh1=dL_ds.mm(W2.t())
    dL_dh1[h1<=0]=0

    grads['b1']=dL_dh1.sum(dim=0)
    grads['W1']=X.t().mm(dL_dh1)+2*reg*W1

    return loss, grads


def nn_train(
    params: Dict[str, torch.Tensor],
    loss_func: Callable,
    pred_func: Callable,
    X: torch.Tensor,
    y: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    learning_rate: float = 1e-3,
    learning_rate_decay: float = 0.95,
    reg: float = 5e-6,
    num_iters: int = 100,
    batch_size: int = 200,
    verbose: bool = False,
):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - params: a dictionary of PyTorch Tensor that store the weights of a model.
      It should have following keys with shape
          W1: First layer weights; has shape (D, H)
          b1: First layer biases; has shape (H,)
          W2: Second layer weights; has shape (H, C)
          b2: Second layer biases; has shape (C,)
    - loss_func: a loss function that computes the loss and the gradients.
      It takes as input:
      - params: Same as input to nn_train
      - X_batch: A minibatch of inputs of shape (B, D)
      - y_batch: Ground-truth labels for X_batch
      - reg: Same as input to nn_train
      And it returns a tuple of:
        - loss: Scalar giving the loss on the minibatch
        - grads: Dictionary mapping parameter names to gradients of the loss with
          respect to the corresponding parameter.
    - pred_func: prediction function that im
    - X: A PyTorch tensor of shape (N, D) giving training data.
    - y: A PyTorch tensor of shape (N,) giving training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - X_val: A PyTorch tensor of shape (N_val, D) giving validation data.
    - y_val: A PyTorch tensor of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.

    Returns: A dictionary giving statistics about the training process
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train // batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
        X_batch, y_batch = sample_batch(X, y, num_train, batch_size)

        # Compute loss and gradients using the current minibatch
        loss, grads = loss_func(params, X_batch, y=y_batch, reg=reg)
        loss_history.append(loss.item())

        params['W1']-= learning_rate * grads['W1']
        params['b1']-= learning_rate * grads['b1']
        params['W2']-= learning_rate * grads['W2']
        params['b2']-= learning_rate * grads['b2']

        if verbose and it % 100 == 0:
            print("iteration %d / %d: loss %f" % (it, num_iters, loss.item()))

        # Every epoch, check train and val accuracy and decay learning rate.
        if it % iterations_per_epoch == 0:
            # Check accuracy
            y_train_pred = pred_func(params, loss_func, X_batch)
            train_acc = (y_train_pred == y_batch).float().mean().item()
            y_val_pred = pred_func(params, loss_func, X_val)
            val_acc = (y_val_pred == y_val).float().mean().item()
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

            # Decay learning rate
            learning_rate *= learning_rate_decay

    return {
        "loss_history": loss_history,
        "train_acc_history": train_acc_history,
        "val_acc_history": val_acc_history,}


def nn_predict(params: Dict[str, torch.Tensor], loss_func: Callable, X: torch.Tensor):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - params: a dictionary of PyTorch Tensor that store the weights of a model.
      It should have following keys with shape
          W1: First layer weights; has shape (D, H)
          b1: First layer biases; has shape (H,)
          W2: Second layer weights; has shape (H, C)
          b2: Second layer biases; has shape (C,)
    - loss_func: a loss function that computes the loss and the gradients
    - X: A PyTorch tensor of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A PyTorch tensor of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """

    results,_=nn_forward_pass(params,X)
    _,y_pred=results.max(dim=1)

    return y_pred


def nn_get_search_params():
    """
    Return candidate hyperparameters for a TwoLayerNet model.

    Returns:
    - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
    - hidden_sizes: hidden value sizes, e.g. [8, 16, ...]
    - regularization_strengths: regularization strengths candidates
                                e.g. [1e0, 1e1, ...]
    - learning_rate_decays: learning rate decay candidates
                                e.g. [1.0, 0.95, ...]
    """

    learning_rates=[0.1,1,2]
    hidden_sizes=[32,64,128]
    regularization_strengths=[1e-5,1e-4,1e-3]
    learning_rate_decays=[0.85,0.9,0.95,0.99]
    
    return (
        learning_rates,
        hidden_sizes,
        regularization_strengths,
        learning_rate_decays,)


def find_best_net(data_dict: Dict[str, torch.Tensor], get_param_set_fn: Callable):
    """
    Tune hyperparameters using the validation set.
    Store your best trained TwoLayerNet model in best_net, with the return value
    of ".train()" operation in best_stat and the validation accuracy of the
    trained best model in best_val_acc. Your hyperparameters should be received
    from in nn_get_search_params

    Inputs:
    - data_dict (dict): a dictionary that includes
                        ['X_train', 'y_train', 'X_val', 'y_val']
                        as the keys for training a classifier
    - get_param_set_fn (function): A function that provides the hyperparameters
                                   (e.g., nn_get_search_params)
                                   that gives (learning_rates, hidden_sizes,
                                   regularization_strengths, learning_rate_decays)
                                   You should get hyperparameters from
                                   get_param_set_fn.

    Returns:
    - best_net (instance): a trained TwoLayerNet instances with
                           (['X_train', 'y_train'], batch_size, learning_rate,
                           learning_rate_decay, reg)
                           for num_iter times.
    - best_stat (dict): return value of "best_net.train()" operation
    - best_val_acc (float): validation accuracy of the best_net
    """

    best_val_acc = 0.0

    learning_rates, hidden_sizes, regularization_strengths, learning_rate_decays = get_param_set_fn()
    for reg in regularization_strengths :
        for lr in learning_rates:
            for hs in hidden_sizes:
                print('train with hidden_size: {}'.format(hs))
                print('train with learning_rate: {}'.format(lr))
                print('train with regularization: {}'.format(reg))

                net = TwoLayerNet(3 * 32 * 32, hs, 10, device=data_dict['X_train'].device,dtype=data_dict['X_train'].dtype)
                stats = net.train(data_dict['X_train'], data_dict['y_train'], data_dict['X_val'], data_dict['y_val'],
                                  num_iters=3000, batch_size=1000,learning_rate=lr, learning_rate_decay=0.95,reg=reg, verbose=False)
                if max(stats['val_acc_history']) > best_val_acc:
                    best_val_acc = max(stats['val_acc_history'])
                    best_net = net
                    best_stat = stats

    return best_net, best_stat, best_val_acc