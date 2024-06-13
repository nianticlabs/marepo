# Copyright © Niantic, Inc. 2024.
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .linear_attention import LinearAttention, FullAttention
from einops.einops import rearrange, repeat
from transformer.position_encoding import PositionEncodingSine
import pytorch3d.transforms as transforms

import importlib.util
if importlib.util.find_spec('flash_attn'):
    from flash_attn import flash_attn_func

RANDOM_RESCALE_RATIO=1 # Hyperparam that control the percentage we use random recale? Default is set to 1
import random
random.seed(2024)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, attention='linear'):
        '''
        attention: (optional) 'linear': linear attention like LofTR
                              'flash': flash attention (only support new arch. such as A100)
        '''
        super().__init__()
        # flag for whether to return the attention features for plotting
        self.return_attention=False
        self.dim = d_model // nhead # 512/8
        self.nhead = nhead # 8

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.enable_flash=False
        if attention=='linear':
            self.attention = LinearAttention()
        elif attention=='flash':
            self.attention = flash_attn_func
            self.enable_flash=True
        else:
            self.attention = FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network/FFN
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None, return_attn=False):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)], [32, 4800, 4, 32]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)], [32, 4800, 4, 32]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim) # [32, 4800, 4, 32]

        if return_attn == True:
            attn = self.attention.get_QK_attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)
            return attn
        if self.enable_flash:
            # flash attention only supports fp16 or bf16
            attn = self.attention(query, key, value) # flash attention currently does not support mask https://github.com/Dao-AILab/flash-attention/issues/409
        else:
            attn = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask) #.type(x.dtype)  # [N, L, (H, D)], use in fp32?

        message = self.merge(attn.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)#.type(x.dtype) # keep float16 if needed

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)#.type(x.dtype) # keep float16 if needed
        return x + message

class Transformer(nn.Module):
    """My implementation of Transformer."""

    def __init__(self, config):
        super().__init__()

        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        encoder_layer = TransformerEncoderLayer(self.d_model, config['nhead'], config['attention'])

        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])

        ''' Section 3.3 '''
        self.C2F_PREDICTION_VERSION=-1
        if 'c2f' in config:
            if config['c2f'] == 'V0':
                self.C2F_PREDICTION_VERSION=0 # use shared regression head

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_vanilla_self_attention(self, feat0, feat1=None, mask0=None, mask1=None, return_attn=False):
        ''' Vanilla architecture that only contains SA such as 12T1R '''

        feat0_skip = feat0.clone()
        for layer_idx, (layer, name) in enumerate(zip(self.layers, self.layer_names)):
            if return_attn and (layer_idx==len(self.layer_names) - 1):
                return layer(feat0, feat0, mask0, mask0, return_attn=True)

            feat0 = layer(feat0, feat0, mask0, mask0) # scene coordinates self attention

            # skip connection for every 4 T blocks
            if (layer_idx+1) % 4 == 0:
                # print("skip connection")
                feat0 = feat0_skip + feat0
                feat0_skip = feat0.clone()
        return feat0

    def forward_c2f_self_attention_v0(self, feat0, feat1=None, mask0=None, mask1=None, return_attn=False):
        ''' coarse to fine self attention with output features for every four layer block '''
        feat_out_list = []
        feat0_skip = feat0.clone()
        for layer_idx, (layer, name) in enumerate(zip(self.layers, self.layer_names)):
            if name == 'self':
                if return_attn and (layer_idx==len(self.layer_names) - 1):
                    return layer(feat0, feat0, mask0, mask0, return_attn=True)

                feat0 = layer(feat0, feat0, mask0, mask0) # scene coordinates self attention

                # skip connection for every 4 T blocks
                if (layer_idx+1) % 4 == 0:
                    # print("skip connection")
                    feat0 = feat0_skip + feat0
                    feat0_skip = feat0.clone()
                    feat_out_list.append(feat0.clone())
            else:
                raise KeyError
        return feat_out_list

    def forward(self, feat0, feat1=None, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, L, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, L] (optional)
        """

        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"
        if self.C2F_PREDICTION_VERSION==0:
            feat_out_list = self.forward_c2f_self_attention_v0(feat0, feat1, mask0, mask1)
            return feat_out_list
        else:
            feat_out = self.forward_vanilla_self_attention(feat0, feat1, mask0, mask1)
        return feat_out

class ResConvBlock(nn.Module):
    '''
    1x1 convolution residual block
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # We may need a skip layer if the number of features output by the encoder is different.
        self.head_skip = nn.Identity() if self.in_channels == self.out_channels else nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        self.res_conv1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        self.res_conv2 = nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)
        self.res_conv3 = nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)

    def forward(self, res):
        x = F.relu(self.res_conv1(res))
        x = F.relu(self.res_conv2(x))
        x = F.relu(self.res_conv3(x))
        res = self.head_skip(res) + x
        return res

class Transformer_Head(nn.Module):
    """
    A Transformer Pose Regression Head.
    The network predicts a 6DOF pose from given 3d scene coordinate
    """
    def __init__(self, config):
        """
        input_dim: is the size of input HxW
        """
        super().__init__()
        # Misc
        self.config = config
        self.d_model = self.config['d_model']

        self.use_homogeneous = False
        if 'use_homogeneous_pose_t' in self.config:
            if self.config['use_homogeneous_pose_t'] == 'True':
                self.use_homogeneous = True,
                homogeneous_min_scale = 0.01,
                homogeneous_max_scale = 4.0,

        # positional encoding
        self.pos_encoding = PositionEncodingSine(self.config, in_ch_dim=3, sc_pe=self.config["pre_pos_encoding"], pixel_pe=self.config["pixel_pos_encoding"])

        # section 3.3 of paper
        self.coarse_to_fine_prediction=False
        if 'c2f' in self.config:
            self.coarse_to_fine_prediction=True if config['c2f']=='V0' else NotImplementedError

        # currently we only support '6D' and '9D' rotation representation
        self.rot_representation = self.config['rotation_representation'] if 'rotation_representation' in self.config else '6D'

        ''' Large scene exp. consider using this with large scale scenes'''
        # These parameters are only useful if random_rescale_sc=True at forward()
        self.random_rescale_max = config['random_rescale_max'] if 'random_rescale_max' in config else 10
        self.random_shift_max = config['random_shift_max'] if 'random_shift_max' in config else 0

        # transformer + pose regression part
        self.transformer = Transformer(self.config)
        self.res_conv = nn.ModuleList([copy.deepcopy(ResConvBlock(self.d_model, self.d_model))
                                       for _ in range(self.config["num_resconv_block"])])
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        if self.use_homogeneous:
            self.fc_t = nn.Linear(self.d_model, 4)
            # Use buffers because they need to be saved in the state dict.
            self.register_buffer("max_scale", torch.tensor([homogeneous_max_scale]))
            self.register_buffer("min_scale", torch.tensor([homogeneous_min_scale]))
            self.register_buffer("max_inv_scale", 1. / self.max_scale)
            self.register_buffer("h_beta", math.log(2) / (1. - self.max_inv_scale))
            self.register_buffer("min_inv_scale", 1. / self.min_scale)
        else:
            self.fc_t = nn.Linear(self.d_model, 3)

        if self.rot_representation=='9D':
            self.fc_rot = nn.Linear(self.d_model, 9)
        else:
            self.fc_rot = nn.Linear(self.d_model, 6)
        # Learn scene coordinates relative to a mean coordinate (e.g. center of the scene).
        self.register_buffer("transformer_pose_mean", self.config['transformer_pose_mean'].clone().detach().view(1, 3, 1, 1)) # torch.Size([1, 3, 1, 1])

        self.more_mlps = nn.Sequential(
            nn.Linear(self.d_model,self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model,self.d_model),
            nn.ReLU()
            )

    def pose_regression_head(self, feat, sc_shape=[], scale_factor=None, translation_sc_shift_offset=None):
        """
        This is my implementation on pose regression head
        """
        B, C, H, W = sc_shape
        # reduce feature dimension
        feat = feat.view(B, H, W, self.d_model).permute(0, 3, 1, 2).contiguous()  # [N,C,H,W]

        for i in range(self.config["num_resconv_block"]):
            feat = self.res_conv[i](feat)
        feat = self.avgpool(feat)  # [N, 256, 1, 1]
        feat = feat.view(feat.size(0), -1)

        feat = self.more_mlps(feat)

        # predict absolute pose
        out_t = self.fc_t(feat)  # [N,3]

        if self.use_homogeneous:
            # Dehomogenize coords:
            # Softplus ensures we have a smooth homogeneous parameter with a minimum value = self.max_inv_scale.
            h_slice = F.softplus(out_t[:, 3].unsqueeze(1), beta=self.h_beta.item()) + self.max_inv_scale
            h_slice.clamp_(max=self.min_inv_scale)
            out_t = out_t[:, :3] / h_slice

        # add mapping mean back
        # DO Not remove, used in testing, precision is fp32 from this point
        out_t = out_t.float() + self.transformer_pose_mean.clone().detach().view(3)
        out_r = self.fc_rot(feat)  # [N,6]
        return out_t, out_r

    def svd_orthogonalize(self, m):
        """Convert 9D representation to SO(3) using SVD orthogonalization.

        Args:
          m: [BATCH, 3, 3] 3x3 matrices.

        Returns:
          [BATCH, 3, 3] SO(3) rotation matrices.
        """
        if m.dim() < 3:
            m = m.reshape((-1, 3, 3))
        m_transpose = torch.transpose(torch.nn.functional.normalize(m, p=2, dim=-1), dim0=-1, dim1=-2)
        u, s, v = torch.svd(m_transpose)
        det = torch.det(torch.matmul(v, u.transpose(-2, -1)))
        # Check orientation reflection.
        r = torch.matmul(
            torch.cat([v[:, :, :-1], v[:, :, -1:] * det.view(-1, 1, 1)], dim=2),
            u.transpose(-2, -1)
        )
        return r

    def convert_pose_to_4x4(self, B, out_r, out_t, device):
        if self.rot_representation=='9D':
            out_r = self.svd_orthogonalize(out_r)  # [N,3,3]
        else:
            out_r = transforms.rotation_6d_to_matrix(out_r)  # [N,3,3]
        pose = torch.zeros((B, 4, 4), device=device)
        pose[:, :3, :3] = out_r
        pose[:, :3, 3] = out_t
        pose[:, 3, 3] = 1.
        return pose

    def runtime_augmentation(self, sc_coords, B, C, random_rescale_sc):
        '''
        add random_rescale_sc augmentation during training
        '''
        if random.uniform(0, 1) < RANDOM_RESCALE_RATIO:  # only use in training
            # basically, this is a random data augmentation on scene coordinates, to cope with large scale dataset like Cambridge
            # randomly generate dilation scale random numbers. Inspired by erosion/dilation in CV.
            lower_dilation_bound = 1
            upper_dilation_bound = self.random_rescale_max
            # upper_dilation_bound = 5
            # uniformly generate factor between [1~5]
            scale_factor = (upper_dilation_bound - lower_dilation_bound) * torch.rand((B, 1, 1, 1),
                                                                                         device=sc_coords.device) + lower_dilation_bound
            sc_coords_new = sc_coords * scale_factor
            scale_factor = scale_factor[:, :, 0, 0]  # [B,1,1,1] -> [B,1]

            # randomly generate shift scene coordinates to train the pose regressor for outputting large camera poses
            upper_shift_bound = self.random_shift_max
            # upper_shift_bound = 100 # uniformly generate 100 meter shift on 3 translational axis
            translation_sc_shift_offset = 2 * upper_shift_bound * torch.rand((B, C, 1, 1),
                                                                             device=sc_coords.device) - upper_shift_bound  # values range [-100 ~ 100]
            sc_coords_new = sc_coords_new + translation_sc_shift_offset
            translation_sc_shift_offset = translation_sc_shift_offset[:, :, 0, 0]  # [B,C,1,1] -> [B,C]
        else:
            return sc_coords, None, None

        return sc_coords_new, scale_factor, translation_sc_shift_offset

    def forward(self, sc, intrinsics_B33=None, sc_mask=None, random_rescale_sc=False):
        """
        Parameters:
            sc: scene coordinate map [N,C,H,W]
            intrinsics_B33: 3x3 camera intrinsics [B,3,3]
            sc_mask: valid mask for sc [N,1,H,W]. True is valid, False is invalid
            random_rescale_sc: we only add this config for large scene model
        return:
            pose [N,4,4]
        """
        B,C,H,W = sc.shape
        # subtract mapping mean
        sc_coords = sc - self.transformer_pose_mean # DO Not remove, used in testing

        # SC map positional encoding
        sc_feat, pixel_pe = self.pos_encoding(sc_coords, intrinsics_B33)
        feat = rearrange(sc_feat, 'n c h w -> n (h w) c') # [32, 4800, 128]

        if sc_mask==None:
            mask_c0 = mask_c1 = None  # mask is useful in training
        else:
            sc_mask = sc_mask[:,0].view(B,H*W)
            mask_c0 = mask_c1 = sc_mask

        if self.coarse_to_fine_prediction:
            feat_list = self.transformer(feat, None, mask_c0, None)
            pose_list = []
            for block_idx, feat in enumerate(feat_list):
                out_t, out_r = self.pose_regression_head(feat, [B, C, H, W])
                pose = self.convert_pose_to_4x4(B, out_r, out_t, sc_coords.device)
                pose_list.append(pose)
            return pose_list
        else:
            feat = self.transformer(feat, None, mask_c0, None) # [N, HW, C] [64, 4800, 128]
        out_t, out_r = self.pose_regression_head(feat, [B,C,H,W])

        # convert rotation to SO(3), precision is fp32 from this point
        pose = self.convert_pose_to_4x4(B, out_r, out_t, sc_coords.device)
        return pose


