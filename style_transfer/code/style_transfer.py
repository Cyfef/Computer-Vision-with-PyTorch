import torch
import torch.nn as nn
import pickle
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as T
from scipy.ndimage.filters import gaussian_filter1d
from torch.utils.data import DataLoader, sampler

SQUEEZENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float)
SQUEEZENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float)

def check_scipy():
    import scipy

    vnum = int(scipy.__version__.split(".")[1])
    major_vnum = int(scipy.__version__.split(".")[0])

    assert (
        vnum >= 16 or major_vnum >= 1
    ), "You must install SciPy >= 0.16.0 to complete this notebook."

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

def get_zero_one_masks(img, size):
    """
    Helper function to get [0, 1] mask from a mask PIL image (black and white).

    Inputs
    - img: a PIL image of shape (3, H, W)
    - size: image size after reshaping

    Returns: A torch tensor with values of 0 and 1 of shape (1, H, W)
    """
    transform = T.Compose(
        [
            T.Resize(size),
            T.ToTensor(),
        ]
    )
    img = transform(img).sum(dim=0, keepdim=True)
    mask = torch.where(img < 1, 0, 1).to(torch.float)
    return mask

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

def content_loss(content_weight, content_current, content_original):
    """
    Compute the content loss for style transfer.
    
    Inputs:
    - content_weight: Scalar giving the weighting for the content loss.
    - content_current: features of the current image; this is a PyTorch Tensor of shape
      (1, C_l, H_l, W_l).
    - content_original: features of the content image, Tensor with shape (1, C_l, H_l, W_l).

    Returns:
    - scalar content loss
    """
    
    C_l=content_current.shape[1]
    content_current_reshape=content_current.reshape(C_l,-1)
    content_original_reshape=content_original.reshape(C_l,-1)
    content_loss_sum=content_weight*torch.sum((content_current_reshape-content_original_reshape)**2)
    return content_loss_sum


def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.
    
    Inputs:
    - features: PyTorch Tensor of shape (N, C, H, W) giving features for
      a batch of N images.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
    
    Returns:
    - gram: PyTorch Tensor of shape (N, C, C) giving the
      (optionally normalized) Gram matrices for the N input images.
    """

    N,C,H,W=features.shape
    w=1/(H*W*C) if normalize else 1
    features=features.reshape(N,C,-1)
    gram=w*torch.bmm(features,features.transpose(1,2))
    return gram


def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.
    
    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Tensor giving the Gram matrix of the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
      
    Returns:
    - style_loss: A PyTorch Tensor holding a scalar giving the style loss.
    """
    
    style_loss=0.0
    for i,layer in enumerate(style_layers):
        grammatrix=gram_matrix(feats[layer])
        style_loss += style_weights[i]*torch.sum((grammatrix-style_targets[i])**2)
    return style_loss


def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    
    return tv_weight * (((img[:, :, 1:, :] - img[:, :, :-1, :]) ** 2).sum() + ((img[:, :, :, 1:] - img[:, :, :, :-1]) ** 2).sum())


def guided_gram_matrix(features, masks, normalize=True):
  """
  Inputs:
    - features: PyTorch Tensor of shape (N, R, C, H, W) giving features for
      a batch of N images.
    - masks: PyTorch Tensor of shape (N, R, H, W)
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
    
    Returns:
    - gram: PyTorch Tensor of shape (N, R, C, C) giving the
      (optionally normalized) guided Gram matrices for the N input images.
  """
  
  N, R, C, H, W = features.shape

  array_flattened = features.view(N, R, C, -1)
  guidance_flattened = masks.view(N, R, 1, -1)

  guided_gram = torch.zeros(N, R, C, C, device=features.device)
  for c in range(R):
      array_guided = torch.mul(guidance_flattened[:, c], array_flattened[:, c])

      guided_gram[:, c] = torch.bmm(array_guided, array_guided.transpose(1, 2))
  if normalize:
      guided_gram = guided_gram / (H * W * C)
  return guided_gram


def guided_style_loss(feats, style_layers, style_targets, style_weights, content_masks):
    """
    Computes the style loss at a set of layers.
    
    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Tensor giving the guided Gram matrix of the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
    - content_masks: List of the same length as feats, giving a binary mask to the
      features of each layer.
      
    Returns:
    - style_loss: A PyTorch Tensor holding a scalar giving the style loss.
    """
    loss = 0
    for i in range(len(style_layers)):
        style = guided_gram_matrix(feats[style_layers[i]], content_masks[style_layers[i]])
        loss = loss + style_weights[i] * torch.sum((style - style_targets[i]) ** 2)
    return loss