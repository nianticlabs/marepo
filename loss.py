# Copyright © Niantic, Inc. 2024.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def weighted_tanh(repro_errs, weight):
    return weight * torch.tanh(repro_errs / weight).sum()


class ReproLoss:
    """
    Compute per-pixel reprojection loss using different configurable approaches.

    - tanh:     tanh loss with a constant scale factor given by the `soft_clamp` parameter (when a pixel's reprojection
                error is equal to `soft_clamp`, its loss is equal to `soft_clamp * tanh(1)`).
    - dyntanh:  Used in the paper, similar to the tanh loss above, but the scaling factor decreases during the course of
                the training from `soft_clamp` to `soft_clamp_min`. The decrease is linear, unless `circle_schedule`
                is True (default), in which case it applies a circular scheduling. See paper for details.
    - l1:       Standard L1 loss, computed only on those pixels having an error lower than `soft_clamp`
    - l1+sqrt:  L1 loss for pixels with reprojection error smaller than `soft_clamp` and
                `sqrt(soft_clamp * reprojection_error)` for pixels with a higher error.
    - l1+logl1: Similar to the above, but using log L1 for pixels with high reprojection error.
    """

    def __init__(self,
                 total_iterations,
                 soft_clamp,
                 soft_clamp_min,
                 type='dyntanh',
                 circle_schedule=True):

        self.total_iterations = total_iterations
        self.soft_clamp = soft_clamp
        self.soft_clamp_min = soft_clamp_min
        self.type = type
        self.circle_schedule = circle_schedule

    def compute(self, repro_errs_b1N, iteration):
        if repro_errs_b1N.nelement() == 0:
            return 0

        if self.type == "tanh":
            return weighted_tanh(repro_errs_b1N, self.soft_clamp)

        elif self.type == "dyntanh":
            # Compute the progress over the training process.
            schedule_weight = iteration / self.total_iterations

            if self.circle_schedule:
                # Optionally scale it using the circular schedule.
                schedule_weight = 1 - np.sqrt(1 - schedule_weight ** 2)

            # Compute the weight to use in the tanh loss.
            loss_weight = (1 - schedule_weight) * self.soft_clamp + self.soft_clamp_min

            # Compute actual loss.
            return weighted_tanh(repro_errs_b1N, loss_weight)

        elif self.type == "l1":
            # L1 loss on all pixels with small-enough error.
            softclamp_mask_b1 = repro_errs_b1N > self.soft_clamp
            return repro_errs_b1N[~softclamp_mask_b1].sum()

        elif self.type == "l1+sqrt":
            # L1 loss on pixels with small errors and sqrt for the others.
            softclamp_mask_b1 = repro_errs_b1N > self.soft_clamp
            loss_l1 = repro_errs_b1N[~softclamp_mask_b1].sum()
            loss_sqrt = torch.sqrt(self.soft_clamp * repro_errs_b1N[softclamp_mask_b1]).sum()

            return loss_l1 + loss_sqrt

        else:
            # l1+logl1: same as above, but use log(L1) for pixels with a larger error.
            softclamp_mask_b1 = repro_errs_b1N > self.soft_clamp
            loss_l1 = repro_errs_b1N[~softclamp_mask_b1].sum()
            loss_logl1 = torch.log(1 + (self.soft_clamp * repro_errs_b1N[softclamp_mask_b1])).sum()

            return loss_l1 + loss_logl1

class PoseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_func = nn.MSELoss()  # maybe need smarter loss?

    def forward(self, predict_pose, pose):
        pose_loss = self.loss_func(predict_pose, pose)
        return pose_loss

class PoseLossMapFree(nn.Module):
    '''
    A replica of MapFree RPR pose regression loss
    '''
    def __init__(self, soft_clamp=False):
        """
        soft_clamp: if set True, we use tanh to soft clamp large errors, to reduce influence of unsolvable queries during training
        """
        super().__init__()
        self.soft_clamp = soft_clamp
        if self.soft_clamp:
            self.tanh = nn.Tanh()

    def forward(self, predict_pose, pose):
        '''
        predict_pose: [B,3,4] or [B,4,4]
        pose: [B,3,4] or [B,4,4]
        '''
        t = predict_pose[:,:3,3:].transpose(1, 2)
        tgt = pose[:,:3,3:].transpose(1, 2)
        R = predict_pose[:, :3, :3]
        Rgt = pose[:, :3, :3]

        if self.soft_clamp:
            loss = 100 * self.tanh(self.trans_l1_loss(t, tgt) / 100) + 45 * self.tanh(self.rot_angle_loss(R, Rgt)/45)
        else:
            trans_loss = self.trans_l1_loss(t, tgt)
            rot_loss = self.rot_angle_loss(R, Rgt)
            loss =  trans_loss + rot_loss
        return loss, trans_loss, rot_loss

    def trans_l1_loss(self, t, tgt):
        """Computes L1 loss for translation vector
        Input:
        t - estimated translation vector [B, 1, 3]
        tgt - ground-truth translation vector [B, 1, 3]
        Output: translation_loss
        """
        return F.l1_loss(t, tgt)

    def rot_angle_loss(self, R, Rgt):
        """
        Computes rotation loss using L2 error of residual rotation angle [radians]
        Input:
        R - estimated rotation matrix [B, 3, 3]
        Rgt - groundtruth rotation matrix [B, 3, 3]
        Output:  rotation_loss
        """
        residual = R.transpose(1, 2) @ Rgt
        trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
        cosine = (trace - 1) / 2
        cosine = torch.clip(cosine, -0.99999, 0.99999)  # handle numerical errors and NaNs
        R_err = torch.acos(cosine)
        loss = F.l1_loss(R_err, torch.zeros_like(R_err))
        return loss

class PoseLossMapFree_C2F(nn.Module):
    '''
    A replica of MapFree RPR pose regression loss, computing losses for multiple coarse to fine stages
    '''
    def __init__(self, soft_clamp=False):
        """
        soft_clamp: if set True, we use tanh to soft clamp large errors, to reduce influence of unsolvable queries during training
        """
        super().__init__()
        self.soft_clamp = soft_clamp
        if self.soft_clamp:
            self.tanh = nn.Tanh()

    def forward(self, predict_poses, pose):
        '''
        predict_pose: list of [B,3,4] or [B,4,4]
        pose: [B,3,4] or [B,4,4]
        '''
        tgt = pose[:,:3,3:].transpose(1, 2)
        Rgt = pose[:,:3,:3]
        total_loss = 0
        total_trans_loss = 0
        total_rot_loss = 0

        for predict_pose in predict_poses:
            t = predict_pose[:,:3,3:].transpose(1, 2)
            R = predict_pose[:,:3,:3]
            trans_loss = self.trans_l1_loss(t, tgt)
            rot_loss = self.rot_angle_loss(R, Rgt)
            loss =  trans_loss + rot_loss
            total_loss += loss
            total_trans_loss += trans_loss
            total_rot_loss += rot_loss

        return total_loss, total_trans_loss, total_rot_loss

    def trans_l1_loss(self, t, tgt):
        """Computes L1 loss for translation vector
        Input:
        t - estimated translation vector [B, 1, 3]
        tgt - ground-truth translation vector [B, 1, 3]
        Output: translation_loss
        """

        return F.l1_loss(t, tgt)

    def rot_angle_loss(self, R, Rgt):
        """
        Computes rotation loss using L2 error of residual rotation angle [radians]
        Input:
        R - estimated rotation matrix [B, 3, 3]
        Rgt - groundtruth rotation matrix [B, 3, 3]
        Output:  rotation_loss
        """
        residual = R.transpose(1, 2) @ Rgt
        trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
        cosine = (trace - 1) / 2
        cosine = torch.clip(cosine, -0.99999, 0.99999)  # handle numerical errors and NaNs
        R_err = torch.acos(cosine)
        loss = F.l1_loss(R_err, torch.zeros_like(R_err))
        return loss