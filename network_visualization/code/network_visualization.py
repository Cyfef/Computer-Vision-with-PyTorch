import torch
import os
import numpy as np
import torchvision.transforms as T
from scipy.ndimage.filters import gaussian_filter1d


SQUEEZENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float)
SQUEEZENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float)

def load_imagenet_val(num=None, path="./datasets/imagenet_val_25.npz"):
    """Load a handful of validation images from ImageNet.
    Inputs:
    - num: Number of images to load (max of 25)
    Returns:
    - X: numpy array with shape [num, 224, 224, 3]
    - y: numpy array of integer image labels, shape [num]
    - class_names: dict mapping integer label to class name
    """
    imagenet_fn = os.path.join(path)
    if not os.path.isfile(imagenet_fn):
        print("file %s not found" % imagenet_fn)
        print("Run the above cell to download the data")
        assert False, "Need to download imagenet_val_25.npz"
    f = np.load(imagenet_fn, allow_pickle=True)
    X = f["X"]
    y = f["y"]
    class_names = f["label_map"].item()
    if num is not None:
        X = X[:num]
        y = y[:num]
    return X, y, class_names

def preprocess(img, size=224):
    transform = T.Compose(
        [
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(mean=SQUEEZENET_MEAN.tolist(), std=SQUEEZENET_STD.tolist()),
            T.Lambda(lambda x: x[None]),
        ]
    )
    return transform(img)

def deprocess(img, should_rescale=True):
    # should_rescale true for style transfer
    transform_list = [
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=(1.0 / SQUEEZENET_STD).tolist()),
        T.Normalize(mean=(-SQUEEZENET_MEAN).tolist(), std=[1, 1, 1]),
    ]

    if should_rescale:
        transform_list.append(T.Lambda(lambda x: (x - x.min()) / (x.max() - x.min())))

    transform_list.append(T.ToPILImage())

    transform = T.Compose(transform_list)
    return transform(img)

def blur_image(X, sigma=1):
    X_np = X.cpu().clone().numpy()
    X_np = gaussian_filter1d(X_np, sigma, axis=2)
    X_np = gaussian_filter1d(X_np, sigma, axis=3)
    X.copy_(torch.Tensor(X_np).type_as(X))
    return X

def jitter(X, ox, oy):
    """
    Helper function to randomly jitter an image.

    Inputs
    - X: PyTorch Tensor of shape (N, C, H, W)
    - ox, oy: Integers giving number of pixels to jitter along W and H axes

    Returns: A new PyTorch Tensor of shape (N, C, H, W)
    """
    if ox != 0:
        left = X[:, :, :, :-ox]
        right = X[:, :, :, -ox:]
        X = torch.cat([right, left], dim=3)
    if oy != 0:
        top = X[:, :, :-oy]
        bottom = X[:, :, -oy:]
        X = torch.cat([bottom, top], dim=2)
    return X


#---------------------------------------------------------------------------


def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make input tensor require gradient
    X.requires_grad_()

    N = X.shape[0]
    output = model(X)
    loss = torch.sum(output[torch.arange(N), y])
    loss.backward()
    saliency, _ = torch.max(X.grad.data.abs(), dim=1)
    return saliency


def make_adversarial_attack(X, target_y, model, max_iter=100, verbose=True):
    """
    Generate an adversarial attack that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image; Tensor of shape (1, 3, 224, 224)
    - target_y: An integer in the range [0, 1000)
    - model: A pretrained CNN
    - max_iter: Upper bound on number of iteration to perform
    - verbose: If True, it prints the pogress (you can use this flag for debugging)

    Returns:
    - X_adv: An image that is close to X, but that is classifed as target_y
    by the model.
    """
    
    X_adv = X.clone()
    X_adv = X_adv.requires_grad_()

    learning_rate = 1
    
    for i in range(max_iter):
        pred_y = model(X_adv)
        # output_probs = torch.nn.functional.softmax(pred_y, dim=1)
        score = pred_y[0, target_y].squeeze()
        model.zero_grad()
        score.backward()
        if torch.argmax(pred_y[0, :], dim=0) == target_y:
            print("Model is fooled")
            break
        if verbose:
            print(f"Iteration {i:d}: target score {score:.3f}, max score {pred_y.amax():.3f}")
        with torch.no_grad():
            X_adv += learning_rate * X_adv.grad / X_adv.grad.norm()
            X_adv.grad.zero_()
    
    return X_adv


def class_visualization_step(img, target_y, model, **kwargs):
    """
    Performs gradient step update to generate an image that maximizes the
    score of target_y under a pretrained model.

    Inputs:
    - img: random image with jittering as a PyTorch tensor
    - target_y: Integer in the range [0, 1000) giving the index of the class
    - model: A pretrained CNN that will be used to generate the image

    Keyword arguments:
    - l2_reg: Strength of L2 regularization on the image
    - learning_rate: How big of a step to take
    """

    l2_reg = kwargs.pop("l2_reg", 1e-3)
    learning_rate = kwargs.pop("learning_rate", 25)
    
    pred_y = model(img)
    loss = pred_y[0, target_y] - l2_reg * img.norm() ** 2
    loss.backward()
    img.data = img.data + learning_rate * img.grad
    img.grad.zero_()
    
    return img
