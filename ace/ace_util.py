# Copyright © Niantic, Inc. 2023.

import numpy as np
import torch
from einops.einops import repeat

def get_pixel_grid(subsampling_factor):
    """
    Generate target pixel positions according to a subsampling factor, assuming prediction at center pixel.
    """
    pix_range = torch.arange(np.ceil(5000 / subsampling_factor), dtype=torch.float32)
    yy, xx = torch.meshgrid(pix_range, pix_range, indexing='ij')
    return subsampling_factor * (torch.stack([xx, yy]) + 0.5)

def to_homogeneous(input_tensor, dim=1):
    """
    Converts tensor to homogeneous coordinates by adding ones to the specified dimension
    """

    ones = torch.ones_like(input_tensor.select(dim, 0).unsqueeze(dim))
    output = torch.cat([input_tensor, ones], dim=dim)
    return output

def normalize_shape(tensor_in):
    """Bring tensor from shape BxCxHxW to NxC"""
    return tensor_in.transpose(0, 1).flatten(1).transpose(0, 1)

def exp_projection_error(pred_scene_coords_b3HW, gt_inv_poses_b34, Ks_b33, sc_mask, pixel_grid_2HW):
    ''' Experimental reprojection errors
    pred_scene_coords_b3HW: [B,3,H,W]
    gt_inv_poses_b34: [B,3,4]
    Ks_b33: [B,3,3]
    invKs_b33: [B,3,3]
    sc_mask: [B,1,H,W]
    '''
    B,C,H,W = pred_scene_coords_b3HW.shape

    # expand gt_inv_poses_b34, Ks_b33, invKs_b33 to bxHxW, mimic ACE
    gt_inv_poses_b34 = repeat(gt_inv_poses_b34, 'b r c -> (b hw) r c', hw=H * W) # [32x60x80,3,4]
    Ks_b33 = repeat(Ks_b33, 'b r c -> (b hw) r c', hw=H * W)

    if sc_mask != None:
        sc_mask_b1 = sc_mask.reshape((-1,1))

    # Create a tensor with the pixel coordinates of every feature vector.
    pixel_positions_B2HW = pixel_grid_2HW[:, :H, :W].clone()  # It's 2xHxW (actual H and W) now.
    pixel_positions_B2HW = pixel_positions_B2HW[None]  # 1x2xHxW
    pixel_positions_B2HW = pixel_positions_B2HW.expand(B, 2, H, W)  # Bx2xHxW

    target_px_b2= normalize_shape(pixel_positions_B2HW) # [32x60x80,2]

    # Back to the original shape. Convert to float32 as well.
    pred_scene_coords_b31 = pred_scene_coords_b3HW.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(-1).float() # # [32x60x80, 3, 1]

    # Make 3D points homogeneous so that we can easily matrix-multiply them.
    pred_scene_coords_b41 = to_homogeneous(pred_scene_coords_b31) # [32x60x80, 4, 1]

    # Scene coordinates to camera coordinates.
    pred_cam_coords_b31 = torch.bmm(gt_inv_poses_b34, pred_scene_coords_b41)

    # Project scene coordinates.
    pred_px_b31 = torch.bmm(Ks_b33, pred_cam_coords_b31)

    # Avoid division by zero.
    # Note: negative values are also clamped at +depth_min=0.1. The predicted pixel would be wrong,
    # but that's fine since we mask them out later.
    depth_min=0.1 # 10cm
    depth_max=1000 # 1000m
    repro_loss_hard_clamp=1000 # 1000px

    pred_px_b31[:, 2].clamp_(min=depth_min)
    # pred_px_b31[:, 2] = torch.clamp(pred_px_b31[:, 2], min=depth_min) # inplace version

    # Dehomogenise.
    pred_px_b21 = pred_px_b31[:, :2] / pred_px_b31[:, 2, None]

    # Measure reprojection error.
    reprojection_error_b2 = pred_px_b21.squeeze() - target_px_b2
    reprojection_error_b1 = torch.norm(reprojection_error_b2, dim=1, keepdim=True, p=1)

    # Compute masks used to ignore invalid pixels.
    #
    # Predicted coordinates behind or close to camera plane.
    invalid_min_depth_b1 = pred_cam_coords_b31[:, 2] < depth_min
    # Very large reprojection errors.
    invalid_repro_b1 = reprojection_error_b1 > repro_loss_hard_clamp
    # Predicted coordinates beyond max distance.
    invalid_max_depth_b1 = pred_cam_coords_b31[:, 2] > depth_max

    # Invalid mask is the union of all these. Valid mask is the opposite.
    invalid_mask_b1 = (invalid_min_depth_b1 | invalid_repro_b1 | invalid_max_depth_b1)
    valid_mask_b1 = ~invalid_mask_b1

    # Also apply invalid scene coordinate mask due to data augmentation
    # valid_mask_b1 = (valid_mask_b1&sc_mask_b1)
    # idx_True = torch.where(invalid_repro_b1==True)[0]
    idx_false = torch.where(valid_mask_b1 == False)[0]

    # torch.autograd.set_detect_anomaly(True)
    reprojection_error_b1 = torch.clamp(reprojection_error_b1,max=repro_loss_hard_clamp) # temp fix, # inplace version
    # reprojection_error_b1[idx_false] = torch.clamp(reprojection_error_b1[idx_false], max=repro_loss_hard_clamp)
    # reprojection_error_b1[invalid_repro_b1].clamp_(max=repro_loss_hard_clamp) # not working
    # reprojection_error_b1[idx_false].clamp_(max=repro_loss_hard_clamp)
    # reprojection_error_b1.clamp_(max=repro_loss_hard_clamp)
    # reprojection_error_b1 = reprojection_error_b1.clamp_(max=repro_loss_hard_clamp)
    # reprojection_error_b1[idx_false] = reprojection_error_b1[idx_false].clamp_(max=repro_loss_hard_clamp) 
    # reprojection_error_b1[idx_false] = repro_loss_hard_clamp # hard clamped reprojection errors, unable to BP. Old Implementation

    # Also apply invalid scene coordinate mask due to data augmentation
    if sc_mask != None:
        reprojection_error_b1 = reprojection_error_b1*sc_mask_b1
    # TODO: consider add soft_clamp
    reprojection_error_map_B1HW = reprojection_error_b1.reshape(B,H,W,1).permute(0,3,1,2)
    return reprojection_error_map_B1HW

def concat_embed_feature_with_conf_map(feat, conf_map, B, H, W, nhead, old_d_model):
    '''
    concat the embedded feature with confidence map
    feat: [N,L,C]
    conf_map: [N,L]
    return:
        feat: [N,L,C+1]
    '''
    feat = feat.reshape((B, H * W, nhead, old_d_model //  nhead))  # [32,4800,4,32]

    # concat the embedded feature with confidence map
    conf_map = conf_map[:, :, None, None]  # [32,4800,1,1]
    conf_map = repeat(conf_map, 'n c l1 l2 -> n c (repeat l1) l2', repeat=nhead)  # [32,4800,4,1]
    feat = torch.cat([feat, conf_map], dim=3)
    feat = feat.reshape((B, (H * W), -1))
    return feat

def concat_embed_feature_with_reproj_err_map(feat, reproj_err_map, B, H, W, nhead, old_d_model):
    '''
    concat the embedded feature with confidence map
    feat: [N,L,C]
    reproj_err_map: [N,1,H,W]
    return:
        feat: [N,L,C+1]
    '''
    feat = feat.reshape((B,H*W,nhead,-1))  # [32,4800,4,32]
    reproj_err_map = reproj_err_map.permute(0, 2, 3, 1).reshape((B, H * W, 1))[:,:,:,None] # [32,4800,1,1]

    # concat the embedded feature with confidence map
    reproj_err_map = repeat(reproj_err_map, 'n c l1 l2 -> n c (repeat l1) l2', repeat=nhead)  # [32,4800,4,1]
    # breakpoint()
    feat = torch.cat([feat[:,:,:,:old_d_model//nhead], reproj_err_map], dim=3) # [32,4800,4,33]
    feat = feat.reshape((B, (H * W), -1))
    return feat