import torch
import pickle
import os
import time
import random
import matplotlib.pyplot as plt
import torchvision
from torchvision.datasets import CIFAR10



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

def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[torch.arange(N), y]
    margins = (x - correct_class_scores[:, None] + 1.0).clamp(min=0.)
    margins[torch.arange(N), y] = 0.
    loss = margins.sum() / N
    num_pos = (margins > 0).sum(dim=1)
    dx = torch.zeros_like(x)
    dx[margins > 0] = 1.
    dx[torch.arange(N), y] -= num_pos.to(dx.dtype)
    dx /= N
    return loss, dx

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for
      the jth class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label
      for x[i] and 0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - x.max(dim=1, keepdim=True).values
    Z = shifted_logits.exp().sum(dim=1, keepdim=True)
    log_probs = shifted_logits - Z.log()
    probs = log_probs.exp()
    N = x.shape[0]
    loss = (-1.0 / N) * log_probs[torch.arange(N), y].sum()
    dx = probs.clone()
    dx[torch.arange(N), y] -= 1
    dx /= N
    return loss, dx


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

#------------------------------------------------------------------------
class Solver(object):
    """
    A Solver encapsulates all the logic necessary for training classification
    models. The Solver performs stochastic gradient descent using different
    update rules.
    The solver accepts both training and validation data and labels so it can
    periodically check classification accuracy on both training and validation
    data to watch out for overfitting.
    To train a model, you will first construct a Solver instance, passing the
    model, dataset, and various options (learning rate, batch size, etc) to the
    constructor. You will then call the train() method to run the optimization
    procedure and train the model.
    After the train() method returns, model.params will contain the parameters
    that performed best on the validation set over the course of training.
    In addition, the instance variable solver.loss_history will contain a list
    of all losses encountered during training and the instance variables
    solver.train_acc_history and solver.val_acc_history will be lists of the
    accuracies of the model on the training and validation set at each epoch.
    Example usage might look something like this:
    data = {
      'X_train': # training data
      'y_train': # training labels
      'X_val': # validation data
      'y_val': # validation labels
    }
    model = MyAwesomeModel(hidden_size=100, reg=10)
    solver = Solver(model, data,
            update_rule=sgd,
            optim_config={
              'learning_rate': 1e-3,
            },
            lr_decay=0.95,
            num_epochs=10, batch_size=100,
            print_every=100,
            device='cuda')
    solver.train()
    A Solver works on a model object that must conform to the following API:
    - model.params must be a dictionary mapping string parameter names to torch
      tensors containing parameter values.
    - model.loss(X, y) must be a function that computes training-time loss and
      gradients, and test-time classification scores, with the following inputs
      and outputs:
      Inputs:
      - X: Array giving a minibatch of input data of shape (N, d_1, ..., d_k)
      - y: Array of labels, of shape (N,) giving labels for X where y[i] is the
      label for X[i].
      Returns:
      If y is None, run a test-time forward pass and return:
      - scores: Array of shape (N, C) giving classification scores for X where
      scores[i, c] gives the score of class c for X[i].
      If y is not None, run a training time forward and backward pass and
      return a tuple of:
      - loss: Scalar giving the loss
      - grads: Dictionary with the same keys as self.params mapping parameter
      names to gradients of the loss with respect to those parameters.
      - device: device to use for computation. 'cpu' or 'cuda'
    """

    def __init__(self, model, data, **kwargs):
        """
        Construct a new Solver instance.
        Required arguments:
        - model: A model object conforming to the API described above
        - data: A dictionary of training and validation data containing:
          'X_train': Array, shape (N_train, d_1, ..., d_k) of training images
          'X_val': Array, shape (N_val, d_1, ..., d_k) of validation images
          'y_train': Array, shape (N_train,) of labels for training images
          'y_val': Array, shape (N_val,) of labels for validation images
        Optional arguments:
        - update_rule: A function of an update rule. Default is sgd.
        - optim_config: A dictionary containing hyperparameters that will be
          passed to the chosen update rule. Each update rule requires different
          hyperparameters but all update rules require a
          'learning_rate' parameter so that should always be present.
        - lr_decay: A scalar for learning rate decay; after each epoch the
          learning rate is multiplied by this value.
        - batch_size: Size of minibatches used to compute loss and gradient
          during training.
        - num_epochs: The number of epochs to run for during training.
        - print_every: Integer; training losses will be printed every
          print_every iterations.
        - print_acc_every: We will print the accuracy every
          print_acc_every epochs.
        - verbose: Boolean; if set to false then no output will be printed
          during training.
        - num_train_samples: Number of training samples used to check training
          accuracy; default is 1000; set to None to use entire training set.
        - num_val_samples: Number of validation samples to use to check val
          accuracy; default is None, which uses the entire validation set.
        - checkpoint_name: If not None, then save model checkpoints here every
          epoch.
        """
        self.model = model
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_val = data["X_val"]
        self.y_val = data["y_val"]

        # Unpack keyword arguments
        self.update_rule = kwargs.pop("update_rule", self.sgd)
        self.optim_config = kwargs.pop("optim_config", {})
        self.lr_decay = kwargs.pop("lr_decay", 1.0)
        self.batch_size = kwargs.pop("batch_size", 100)
        self.num_epochs = kwargs.pop("num_epochs", 10)
        self.num_train_samples = kwargs.pop("num_train_samples", 1000)
        self.num_val_samples = kwargs.pop("num_val_samples", None)

        self.device = kwargs.pop("device", "cpu")

        self.checkpoint_name = kwargs.pop("checkpoint_name", None)
        self.print_every = kwargs.pop("print_every", 10)
        self.print_acc_every = kwargs.pop("print_acc_every", 1)
        self.verbose = kwargs.pop("verbose", True)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError("Unrecognized arguments %s" % extra)

        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Make a deep copy of the optim_config for each parameter
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d

    def _step(self):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """
        # Make a minibatch of training data
        num_train = self.X_train.shape[0]
        batch_mask = torch.randperm(num_train)[: self.batch_size]
        X_batch = self.X_train[batch_mask].to(self.device)
        y_batch = self.y_train[batch_mask].to(self.device)

        # Compute loss and gradient
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss.item())

        # Perform a parameter update
        with torch.no_grad():
            for p, w in self.model.params.items():
                dw = grads[p]
                config = self.optim_configs[p]
                next_w, next_config = self.update_rule(w, dw, config)
                self.model.params[p] = next_w
                self.optim_configs[p] = next_config

    def _save_checkpoint(self):
        if self.checkpoint_name is None:
            return
        checkpoint = {
            "model": self.model,
            "update_rule": self.update_rule,
            "lr_decay": self.lr_decay,
            "optim_config": self.optim_config,
            "batch_size": self.batch_size,
            "num_train_samples": self.num_train_samples,
            "num_val_samples": self.num_val_samples,
            "epoch": self.epoch,
            "loss_history": self.loss_history,
            "train_acc_history": self.train_acc_history,
            "val_acc_history": self.val_acc_history,
        }
        filename = "%s_epoch_%d.pkl" % (self.checkpoint_name, self.epoch)
        if self.verbose:
            print('Saving checkpoint to "%s"' % filename)
        with open(filename, "wb") as f:
            pickle.dump(checkpoint, f)

    @staticmethod
    def sgd(w, dw, config=None):
        """
        Performs vanilla stochastic gradient descent.
        config format:
        - learning_rate: Scalar learning rate.
        """
        if config is None:
            config = {}
        config.setdefault("learning_rate", 1e-2)

        w -= config["learning_rate"] * dw
        return w, config

    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        """
        Check accuracy of the model on the provided data.
        Inputs:
        - X: Array of data, of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,)
        - num_samples: If not None, subsample the data and only test the model
          on num_samples datapoints.
        - batch_size: Split X and y into batches of this size to avoid using
          too much memory.
        Returns:
        - acc: Scalar giving the fraction of instances that were correctly
          classified by the model.
        """

        # Maybe subsample the data
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = torch.randperm(N, device=self.device)[:num_samples]
            N = num_samples
            X = X[mask]
            y = y[mask]
        X = X.to(self.device)
        y = y.to(self.device)

        # Compute predictions in batches
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(torch.argmax(scores, dim=1))

        y_pred = torch.cat(y_pred)
        acc = (y_pred == y).to(torch.float).mean()

        return acc.item()

    def train(self, time_limit=None, return_best_params=True):
        """
        Run optimization to train the model.
        """
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch
        prev_time = start_time = time.time()

        for t in range(num_iterations):

            cur_time = time.time()
            if (time_limit is not None) and (t > 0):
                next_time = cur_time - prev_time
                if cur_time - start_time + next_time > time_limit:
                    print(
                        "(Time %.2f sec; Iteration %d / %d) loss: %f"
                        % (
                            cur_time - start_time,
                            t,
                            num_iterations,
                            self.loss_history[-1],
                        )
                    )
                    print("End of training; next iteration "
                          "will exceed the time limit.")
                    break
            prev_time = cur_time

            self._step()

            # Maybe print training loss
            if self.verbose and t % self.print_every == 0:
                print(
                    "(Time %.2f sec; Iteration %d / %d) loss: %f"
                    % (
                        time.time() - start_time,
                        t + 1,
                        num_iterations,
                        self.loss_history[-1],
                    )
                )

            # At the end of every epoch, increment the epoch counter and decay
            # the learning rate.
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]["learning_rate"] *= self.lr_decay

            # Check train and val accuracy on the first iteration, the last
            # iteration, and at the end of each epoch.
            with torch.no_grad():
                first_it = t == 0
                last_it = t == num_iterations - 1
                if first_it or last_it or epoch_end:
                    train_acc = \
                        self.check_accuracy(self.X_train,
                                            self.y_train,
                                            num_samples=self.num_train_samples)
                    val_acc = \
                        self.check_accuracy(self.X_val,
                                            self.y_val,
                                            num_samples=self.num_val_samples)
                    self.train_acc_history.append(train_acc)
                    self.val_acc_history.append(val_acc)
                    self._save_checkpoint()

                    if self.verbose and self.epoch % self.print_acc_every == 0:
                        print(
                            "(Epoch %d / %d) train acc: %f; val_acc: %f"
                            % (self.epoch, self.num_epochs, train_acc, val_acc)
                        )

                    # Keep track of the best model
                    if val_acc > self.best_val_acc:
                        self.best_val_acc = val_acc
                        self.best_params = {}
                        for k, v in self.model.params.items():
                            self.best_params[k] = v.clone()

        # At the end of training swap the best params into the model
        if return_best_params:
            self.model.params = self.best_params

#------------------------------------------------------------------------

class Linear(object):

    @staticmethod
    def forward(x, w, b):
        """
        Computes the forward pass for an linear (fully-connected) layer.
        The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
        examples, where each example x[i] has shape (d_1, ..., d_k). We will
        reshape each input into a vector of dimension D = d_1 * ... * d_k, and
        then transform it to an output vector of dimension M.
        Inputs:
        - x: A tensor containing input data, of shape (N, d_1, ..., d_k)
        - w: A tensor of weights, of shape (D, M)
        - b: A tensor of biases, of shape (M,)
        Returns a tuple of:
        - out: output, of shape (N, M)
        - cache: (x, w, b)
        """
        
        x_v=x.view(x.shape[0], -1)
        out=x_v.mm(w)+b
        cache = (x, w, b)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for an linear layer.
        Inputs:
        - dout: Upstream derivative, of shape (N, M)
        - cache: Tuple of:
          - x: Input data, of shape (N, d_1, ... d_k)
          - w: Weights, of shape (D, M)
          - b: Biases, of shape (M,)
        Returns a tuple of:
        - dx: Gradient with respect to x, of shape
          (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        """
        x, w, b = cache
        db=dout.sum(dim=0)
        dx=dout.mm(w.t()).view(x.shape)
        dw=x.view(x.shape[0], -1).t().mm(dout)
        return dx, dw, db


class ReLU(object):

    @staticmethod
    def forward(x):
        """
        Computes the forward pass for a layer of rectified
        linear units (ReLUs).
        Input:
        - x: Input; a tensor of any shape
        Returns a tuple of:
        - out: Output, a tensor of the same shape as x
        - cache: x
        """

        out=x.clone()
        out[x<0]=0
        cache = x
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for a layer of rectified
        linear units (ReLUs).
        Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout
        Returns:
        - dx: Gradient with respect to x
        """
        dx, x = None, cache
        
        dx=dout.clone()
        dx[x<0]=0
        return dx


class Linear_ReLU(object):

    @staticmethod
    def forward(x, w, b):
        """
        Convenience layer that performs an linear transform
        followed by a ReLU.

        Inputs:
        - x: Input to the linear layer
        - w, b: Weights for the linear layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        a, fc_cache = Linear.forward(x, w, b)
        out, relu_cache = ReLU.forward(a)
        cache = (fc_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-relu convenience layer
        """
        fc_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.
    The architecure should be linear - relu - linear - softmax.
    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to PyTorch tensors.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0,
                 dtype=torch.float32, device='cpu'):
        """
        Initialize a new network.
        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        - dtype: A torch data type object; all computations will be
          performed using this datatype. float is faster but less accurate,
          so you should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg

        self.params['W1'] = weight_scale*torch.randn(input_dim, hidden_dim, dtype=dtype, device=device)
        self.params['b1'] = torch.zeros(hidden_dim, dtype=dtype, device=device)
        self.params['W2'] = weight_scale*torch.randn(hidden_dim, num_classes, dtype=dtype, device=device)
        self.params['b2'] = torch.zeros(num_classes, dtype=dtype, device=device)

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'params': self.params,
        }

        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.reg = checkpoint['reg']
        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Tensor of input data of shape (N, d_1, ..., d_k)
        - y: int64 Tensor of labels, of shape (N,). y[i] gives the
          label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model
        and return:
        - scores: Tensor of shape (N, C) giving classification scores,
          where scores[i, c] is the classification score for X[i]
          and class c.
        If y is not None, then run a training-time forward and backward
        pass and return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping
          parameter names to gradients of the loss with respect to
          those parameters.
        """
        
        out_LR,cache_LR=Linear_ReLU.forward(X, self.params['W1'], self.params['b1'])
        scores,cache_L=Linear.forward(out_LR, self.params['W2'], self.params['b2'])

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        loss,dout=softmax_loss(scores,y)
        loss+=((self.params['W1']*self.params['W1']).sum()+(self.params['W2']*self.params['W2']).sum())*self.reg

        dx,dw,db=Linear.backward(dout, cache_L)
        grads['W2'] = dw + 2*self.params['W2']*self.reg
        grads['b2'] = db

        dx,dw,db=Linear_ReLU.backward(dx, cache_LR)
        grads['W1'] = dw+2*self.params['W1']*self.reg
        grads['b1'] = db
        
        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function.
    For a network with L layers, the architecture will be:

    {linear - relu - [dropout]} x (L - 1) - linear - softmax

    where dropout is optional, and the {...} block is repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0.0, reg=0.0, weight_scale=1e-2, seed=None,
                 dtype=torch.float, device='cpu'):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each
          hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving the drop probability
          for networks with dropout. If dropout=0 then the network
          should not use dropout.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - seed: If not None, then pass this random seed to the dropout
          layers. This will make the dropout layers deteriminstic so we
          can gradient check the model.
        - dtype: A torch data type object; all computations will be
          performed using this datatype. float is faster but less accurate,
          so you should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.use_dropout = dropout != 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        last_dim = input_dim
        for n,hidden_dim in enumerate(hidden_dims):
            i=n+1
            self.params[f'W{i}']=weight_scale*torch.randn(last_dim,hidden_dim,dtype=dtype,device=device)
            self.params[f'b{i}']=torch.zeros(hidden_dim,dtype=dtype,device=device)
            last_dim = hidden_dim
        i+=1
        self.params[f'W{i}'] = weight_scale * torch.randn(last_dim, num_classes, dtype=dtype, device=device)
        self.params[f'b{i}'] = torch.zeros(num_classes, dtype=dtype, device=device)

        # When using dropout we need to pass a dropout_param dictionary
        # to each dropout layer so that the layer knows the dropout
        # probability and the mode (train / test). You can pass the same
        # dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
          'num_layers': self.num_layers,
          'use_dropout': self.use_dropout,
          'dropout_param': self.dropout_param,
        }

        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.use_dropout = checkpoint['use_dropout']
        self.dropout_param = checkpoint['dropout_param']

        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.
        Input / output: Same as TwoLayerNet above.
        """
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param
        # since they behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        scores = None
        
        cache_dict={}
        last_out=X
        for i in range(1, self.num_layers):
            last_out ,cache_dict[f'cache_LR{i}']=Linear_ReLU.forward(last_out,self.params[f'W{i}'],self.params[f'b{i}'])
            if self.use_dropout :
                last_out,cache_dict[f'cache_Dropout{i}']=Dropout.forward(last_out,self.dropout_param)
            n=i
        n+=1
        last_out,cache_dict[f'cache_L{n}']=Linear.forward(last_out,self.params[f'W{n}'], self.params[f'b{n}'])
        scores=last_out

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        
        loss, dout = softmax_loss(scores, y)
        loss += (self.params[f'W{n}'] * self.params[f'W{n}']).sum() * self.reg
        last_dout, dw, db = Linear.backward(dout, cache_dict[f'cache_L{n}'])
        grads[f'W{n}'] = dw + 2 * self.params[f'W{n}'] * self.reg
        grads[f'b{n}'] = db
        for n in range(self.num_layers - 1)[::-1]:
            i = n + 1
            if self.use_dropout:
                last_dout = Dropout.backward(last_dout, cache_dict[f'cache_Dropout{i}'])
            last_dout, dw, db = Linear_ReLU.backward(last_dout, cache_dict[f'cache_LR{i}'])
            grads[f'W{i}'] = dw + 2 * self.params[f'W{i}'] * self.reg
            grads[f'b{i}'] = db
            loss += (self.params[f'W{i}'] * self.params[f'W{i}']).sum() * self.reg

        return loss, grads


def create_solver_instance(data_dict, dtype, device):
    model = TwoLayerNet(hidden_dim=200, dtype=dtype, device=device)
    solver=Solver(model,data_dict,device=device,num_epochs=100)
    return solver


def get_three_layer_network_params():
    weight_scale = 1   
    learning_rate = 1e-2 
    return weight_scale, learning_rate


def get_five_layer_network_params():
    learning_rate = 0.05  
    weight_scale = 0.15   
    return weight_scale, learning_rate


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.
    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.
    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to
      store a moving average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', torch.zeros_like(w))

    v=config['momentum']*v-config['learning_rate']*dw
    next_w=w+v
    config['velocity'] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.
    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', torch.zeros_like(w))

    config['cache']=config['decay_rate']*config['cache']+(1-config['decay_rate'])*dw**2
    next_w=w-config['learning_rate']*dw/(torch.sqrt(config['cache'])+config['epsilon'])

    return next_w, config


def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.
    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', torch.zeros_like(w))
    config.setdefault('v', torch.zeros_like(w))
    config.setdefault('t', 0)

    config['t']+=1
    config['m']=config['beta1']*config['m']+(1-config['beta1'])*dw
    mt=config['m']/(1-config['beta1']**config['t'])
    config['v']=config['beta2']*config['v']+(1-config['beta2'])*(dw**2)
    vc=config['v']/(1-config['beta2']**config['t'])
    next_w=w-(config['learning_rate']*mt)/(torch.sqrt(vc)+config['epsilon'])

    return next_w, config


class Dropout(object):

    @staticmethod
    def forward(x, dropout_param):
        """
        Performs the forward pass for (inverted) dropout.
        Inputs:
        - x: Input data: tensor of any shape
        - dropout_param: A dictionary with the following keys:
          - p: Dropout parameter. We *drop* each neuron output with
            probability p.
          - mode: 'test' or 'train'. If the mode is train, then
            perform dropout;
          if the mode is test, then just return the input.
          - seed: Seed for the random number generator. Passing seed
            makes this
            function deterministic, which is needed for gradient checking
            but not in real networks.
        Outputs:
        - out: Tensor of the same shape as x.
        - cache: tuple (dropout_param, mask). In training mode, mask
          is the dropout mask that was used to multiply the input; in
          test mode, mask is None.
        """
        p, mode = dropout_param['p'], dropout_param['mode']
        if 'seed' in dropout_param:
            torch.manual_seed(dropout_param['seed'])

        if mode == 'train':
            mask=(torch.rand(x.shape,device=x.device)>p)/(1-p)
            out=x*mask
        elif mode == 'test':
            out=x

        cache = (dropout_param, mask)

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Perform the backward pass for (inverted) dropout.
        Inputs:
        - dout: Upstream derivatives, of any shape
        - cache: (dropout_param, mask) from Dropout.forward.
        """
        dropout_param, mask = cache
        mode = dropout_param['mode']

        if mode == 'train':
            dx=dout*mask
        elif mode == 'test':
            dx = dout
        return dx
