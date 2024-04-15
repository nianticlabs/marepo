# Copyright © Niantic, Inc. 2024.

import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

def save_image_saliancy(tensor, path, normalize: bool = False, scale_each: bool = False,):
    """
    Modification based on TORCHVISION.UTILS
    ::param: tensor (batch, channel, H, W)
    """
    # grid = make_grid(tensor.detach(), normalize=normalize, scale_each=scale_each, nrow=32)
    grid = make_grid(tensor.detach(), normalize=normalize, scale_each=scale_each, nrow=6)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    fig = plt.figure()
    plt.imshow(ndarr[:,:,0], cmap='jet') # viridis, plasma
    plt.axis('off')
    fig.savefig(path, bbox_inches='tight',dpi=fig.dpi,pad_inches=0.0)
    plt.close()

def plot_image(tensor, normalize: bool = False, scale_each: bool = False,):
    """
    Modification based on TORCHVISION.UTILS
    ::param: tensor (batch, channel, H, W)
    """

    grid = make_grid(tensor.detach(), normalize=normalize, scale_each=scale_each, nrow=6).permute(1, 2, 0).to('cpu').numpy()# [H,W,C]
    fig = plt.figure()
    plt.imshow(grid) # viridis, plasma
    plt.axis('off')
    # fig.savefig(path, bbox_inches='tight',dpi=fig.dpi,pad_inches=0.0)
    # plt.close()
    return fig


def plot_image_saliancy(tensor, normalize: bool = False, scale_each: bool = False,):
    """
    Modification based on TORCHVISION.UTILS
    ::param: tensor (batch, channel, H, W)
    """

    # grid = make_grid(tensor.detach(), normalize=normalize, scale_each=scale_each, nrow=32)
    grid = make_grid(tensor.detach(), normalize=normalize, scale_each=scale_each, nrow=6)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()# [H,W,C]
    fig = plt.figure()
    plt.imshow(ndarr[:,:,0], cmap='jet') # viridis, plasma
    plt.axis('off')
    # fig.savefig(path, bbox_inches='tight',dpi=fig.dpi,pad_inches=0.0)
    # plt.close()
    return fig

def plot_image_saliancy_with_blending(img, tensor, normalize: bool = False, scale_each: bool = False,):
    """
    Modification based on TORCHVISION.UTILS
    ::param: image (batch, channel, H, W)
    ::param: tensor (batch, channel, H, W)
    """
    img = TF.resize(img, tensor.shape[1], antialias=True)
    grid_img = make_grid(img.detach(), normalize=normalize, scale_each=False, nrow=6).permute(1,2,0).to('cpu').numpy()

    fig = plt.figure()
    plt.imshow(grid_img)

    grid = make_grid(tensor.detach(), normalize=normalize, scale_each=scale_each, nrow=6)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()# [H,W,C]

    plt.imshow(ndarr[:,:,0], cmap='jet', alpha=0.5) # viridis, plasma
    plt.axis('off')
    # fig.savefig(path, bbox_inches='tight',dpi=fig.dpi,pad_inches=0.0)
    # plt.close()
    return fig

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor