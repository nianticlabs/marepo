'''
MIT License

Copyright (c) 2022 Active Vision Laboratory

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Code modified from DFNet: Enhance Absolute Pose Regression with Direct Feature Matching (https://arxiv.org/abs/2204.00559)
'''

import numpy as np
from tqdm import tqdm

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, args, patience=5, verbose=False, delta=0, default_ckpt_save_path=""):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        # self.val_on_psnr = args.val_on_psnr
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.ckpt_save_path = default_ckpt_save_path

    def __call__(self, val_loss, epoch=-1, save_multiple=False, save_all=False, val_psnr=None):
        '''
        Given the current validation loss
        return ckpt_save_path: name of the best checkpoint to be saved
               or None: meaning no best checkpoint is saved in this epoch
        '''

        ckpt_save_path = None
        # find minimum loss
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            ckpt_save_path =  self.generate_save_checkpoint_path(val_loss, epoch=epoch, save_multiple=save_multiple)
        elif score < self.best_score + self.delta:
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True

            if save_all: # save all ckpt
                ckpt_save_path =  self.generate_save_checkpoint_path(val_loss, epoch=epoch, save_multiple=True, update_best=False)
        else: # save best ckpt only
            self.best_score = score
            ckpt_save_path = self.generate_save_checkpoint_path(val_loss, epoch=epoch, save_multiple=save_multiple)
            self.counter = 0
        return ckpt_save_path

    def generate_save_checkpoint_path(self, val_loss, epoch=-1, save_multiple=False, update_best=True):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            tqdm.write(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        ckpt_save_path = self.ckpt_save_path
        if save_multiple:
            ckpt_save_path = ckpt_save_path[:-3]+f'-{epoch:04d}-{val_loss:.4f}.pt'

        if update_best:
            self.val_loss_min = val_loss

        return ckpt_save_path
    
    def isBestModel(self):
        ''' Check if current model the best one.
        get early stop counter, if counter==0: it means current model has the best validation loss
        '''
        return self.counter==0