#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2024.
import torch
import logging
from marepo.load_dataset_aceformer import  load_multiple_scene
import sys, time, os
import os.path as osp
import numpy as np


import argparse
from pathlib import Path
from distutils.util import strtobool

import matplotlib.pyplot as plt

DEBUG=False

_logger = logging.getLogger(__name__)

def _strtobool(x):
    return bool(strtobool(x))
def get_opts():
    parser = argparse.ArgumentParser(
            description='Fast training of a scene coordinate regression network.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--encoder_path', type=Path, default=Path(__file__).parent / "ace_encoder_pretrained.pt",
                            help='file containing pre-trained encoder weights')
    parser.add_argument('--head_network_path', type=Path,
                            default=Path(__file__).parent / "logs/wayspots_bears/wayspots_bears.pt",
                            help='file containing pre-trained ACE head weights, (Not used in Marepo)')
    parser.add_argument('--dataset_path', type=Path,
                            default="",
                            help='path to the dataset folder, e.g. "~/storage/map_free_training_scenes/". '
                                 'When this config is set, it means we are training with every scenes in the dataset')
    parser.add_argument('--dataset_head_network_path', type=Path,
                            default="",
                            help='path to the pre-trained ACE head weights of entire dataset, e.g. "logs/mapfree/". '
                                 'When this config is set, it means we are training with every scenes in the dataset')
    parser.add_argument('--use_half', type=_strtobool, default=True,
                            help='train with half precision')
    parser.add_argument('--image_resolution', type=int, default=480,
                            help='base image resolution')
    parser.add_argument('--preprocessing', type=_strtobool, default=False,
                            help='use pretrained ACE networks to generate scene coordinate maps')
    parser.add_argument('--trainskip', type=int, default=1,
                            help='uniformly subsample train set by 1/trainskip')
    parser.add_argument('--testskip', type=int, default=1,
                            help='uniformly subsample val/test set by 1/testskip')

    # new params for this file
    parser.add_argument('--load_scheme2_sc_map', type=_strtobool, default=False,
                        help='use saved SC maps (subtract mean) and GT pose (subtract mean)'
                             'instead of use original SC map and GT pose')
    parser.add_argument('--load_scheme3_sc_map', type=_strtobool, default=False,
                        help='use additional saved augmented SC maps (subtract mean) and GT pose (subtract mean)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='number of train set batch size')
    parser.add_argument('--val_batch_size', type=int, default=32,
                        help='number of val set batch size')
    parser.add_argument('--test_batch_size', type=int, default=32,
                        help='number of test set batch size')
    return parser.parse_args()

def plot_histogram(aug_pp_axis, gnum, prefix, suffix):
    '''
    aug_pp_axis: data
    gnum: if -1, we plot all data, if >= 0, we plot per group data with group number gnum
    prefix:
    suffix:
    '''
    fig, ax = plt.subplots()
    num_bins = 100
    n, bins, patches = ax.hist(aug_pp_axis, num_bins, density=True)
    ax.set_xlabel('Principle Point Value after Augmentation')
    ax.set_ylabel('Probability density')
    ax.set_title(r'Histogram of Augmented Principle Point Value')

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()
    if gnum==-1: # plot all data
        plt.savefig(f'tmp/stats/{prefix}_{suffix}.png')
    else: # plot per group data
        plt.savefig(f'tmp/stats/{prefix}_{gnum:03d}_{suffix}.png')
    plt.close()

def plot_cumulative(aug_pp_axis, gnum, prefix, suffix):
    fig, ax = plt.subplots()
    num_bins = 100
    # plot the cumulative histogram
    n, bins, patches = ax.hist(aug_pp_axis, num_bins, density=True, histtype='step',
                               cumulative=True, label='Empirical')
    ax.grid(True)
    ax.legend(loc='right')
    ax.set_xlabel('Principle Point Value after Augmentation')
    ax.set_ylabel('Cumulative Distribution') # Cumulative Distribution Function
    ax.set_title(r'Approximated CDF of Augmented Principle Point Value')
    plt.show()
    if gnum==-1: # plot all data
        plt.savefig(f'tmp/stats/{prefix}_{suffix}.png')
    else: # plot per group data
        plt.savefig(f'tmp/stats/{prefix}_{gnum:03d}_{suffix}.png')
    plt.close()

def plot_heatmap(aug_ppx, aug_ppy, gnum, prefix, suffix):

    # x, y = np.meshgrid(np.linspace(0, 480, 481), np.linspace(0, 640, 641))
    x = np.arange(0,481,1)
    y = np.arange(0,641,1)

    z = np.zeros((480, 640))

    if gnum==-1: # all groups
        for g in range(len(aug_ppx)):
            for i in range(len(aug_ppx[g])):
                z[int(np.floor(aug_ppx[g][i])), int(np.floor(aug_ppy[g][i]))] += 1

    else: # single group
        for i in range(len(aug_ppx)):
            z[int(np.floor(aug_ppx[i])), int(np.floor(aug_ppy[i]))] += 1
    fig, ax = plt.subplots()
    z_min, z_max = 0, np.abs(z).max()

    c = ax.pcolormesh(y, x, z, cmap='RdBu', vmin=z_min, vmax=z_max)
    # c = ax.pcolormesh(y, x, z, cmap='viridis', vmin=z_min, vmax=z_max)
    ax.set_title('Principle Point Color Map')
    # set the limits of the plot to the limits of the data
    ax.axis([y.min(), y.max(), x.min(), x.max()])
    fig.colorbar(c, ax=ax)
    plt.show()

    if gnum == -1:  # all groups
        plt.savefig(f'tmp/stats/{prefix}_{suffix}.png')
    else:  # single group
        plt.savefig(f'tmp/stats/{prefix}_{gnum:03d}_{suffix}.png')
    plt.close()



if __name__ == '__main__':

    # Setup logging levels.
    logging.basicConfig(level=logging.INFO)
    options = get_opts()

    # plot preprocessed data distribution
    train_dl, val_dl, test_dl = load_multiple_scene(options)

    aug_group = train_dl.dataset.aug_group # assume 8
    rgb_files = train_dl.dataset.rgb_files

    aug_ppx = [[] for i in range(aug_group)] # vertical
    aug_ppy = [[] for i in range(aug_group)] # horizontal

    for rgb_file in rgb_files:

        # load aug. intrinsic files
        for  gnum in range(aug_group):
            # parse aug. intrinsic file names based on rgb filenames
            rgb_stem = rgb_file.stem
            intrinsic_file = str(rgb_file.parent.parent)
            intrinsic_file = intrinsic_file + f"/aug/intrinsics_{gnum:03d}/" + "intrinsic_" + rgb_stem[-5:] + ".txt.npy" # TODO: we saved bad suffix
            # load intrinsic file
            intrinsic = np.load(intrinsic_file)

            # append values to list
            aug_ppx[gnum].append(intrinsic[1,2]) # around 240
            aug_ppy[gnum].append(intrinsic[0,2]) # around 320


    print("heat map begin...")
    # plot per heat map
    for gnum in range(aug_group):
        plot_heatmap(aug_ppx[gnum], aug_ppy[gnum], gnum, 'heatmap_group', '')

    # plot all heat map distribution
    plot_heatmap(aug_ppx, aug_ppy, -1, 'heatmap_group', '')
    print("heat map done...")


    sys.exit()
