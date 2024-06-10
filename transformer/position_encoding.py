# Copyright © Niantic, Inc. 2024.
import torch
import torch.nn as nn
from einops import rearrange, repeat
import math
import copy
import matplotlib.pyplot as plt
import os
import os.path as osp
from transformer.position_encoding_nerf import get_NeRF_embedder

class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, config, max_shape=(256, 256), in_ch_dim=3, sc_pe='default', pixel_pe='default'):
                 # d_model, max_shape=(256, 256), mode='default', pixel_pe='default'):
        """
        Args:
            d_model:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            mode: the mode determines how input feature dimensions increase before position encoding
            (Options): 'default': do not increase embedding dimension, from LoFTR
                    'mlp': use mlp to increase input dimension,
                    'conv1x1': use 1x1 conv for better efficiency
                    'nerf': nerf style pose encoding,
                    'nerf2': nerf style pose encoding + concatenate pixel positional encoding
                    'nn_embedding': torch.nn.embedding
            pixel_pe: the mode determines how pixel position encoding is implemented
            (Options): 'default': do not increase embedding dimension, from LoFTR
                    'focal_norm': Subtract - x/y principal points -> div focal length
            in_ch_dim: input channel dimension
        """
        super().__init__()

        d_model = config["d_model"]
        self.in_ch_dim = in_ch_dim
        self.default_img_HW = config["default_img_HW"] # ex: [480,640]

        # Raise input scene coordinate dimensions if necessary
        self.sc_pe=sc_pe
        if self.sc_pe == 'default':
            self.forward_fn = self.forward_default
        elif self.sc_pe == 'nerf': # used in paper section 3.2.1
            # assume out of range value of SC (for experiment only)
            self.use_contract = False
            if 'contract' in config:
                # soft contract like in mip-nerf-360 paper
                self.use_contract=True
                self.oor = config['oor'] if 'oor' in config else 50
                print("warning: exp: use contract method, anything <=  +/-{} for SC map will be < 1 and > +/-{} will be < 2".format(self.oor, self.oor))
            else:
                # we use hard clip like in the marepo paper
                self.oor = config['oor'] if 'oor' in config else 50
                print("warning: exp: assume out of range value +/-{} for SC map, remove later".format(self.oor))

            nerf_frequency_band=config['nerf_frequency_band'] if 'nerf_frequency_band' in config else 5 # we used 5 in our CVPR submission
            self.pre_pe_embedding, out_dim, self.pe_nerf_embed_obj = get_NeRF_embedder(nerf_frequency_band)
            self.pre_pe_dim_manipulator = nn.Conv2d(out_dim,d_model,1,1,0) # adjust input dimension to d_model
            self.forward_fn = self.forward_nerf
        else:
            NotImplementedError

         # Select pixel position encoding
        self.pixel_pe = pixel_pe
        if self.pixel_pe == 'focal_norm': # used in paper section 3.2.1
            self.pe = torch.zeros((d_model, self.default_img_HW[0], self.default_img_HW[1]))
            self.y_position = torch.ones(self.default_img_HW[0], self.default_img_HW[1]).cumsum(0).unsqueeze(0)
            self.x_position = torch.ones(self.default_img_HW[0], self.default_img_HW[1]).cumsum(1).unsqueeze(0)
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2)))
            self.div_term = div_term[None, :, None, None]  # [1, C//4, 1, 1]
        else: # same as loftr
            pe = torch.zeros((d_model, *max_shape))
            y_position = torch.ones(max_shape).cumsum(0).unsqueeze(0) # [1,256,256]
            x_position = torch.ones(max_shape).cumsum(1).unsqueeze(0) # [1,256,256]

            # w_k = 1/(10000^(2k/d)) based on LoFTR paper supp, but use exp(ln()) trick on the right hand side
            # => w_k = exp(k * (-ln(10000)) * (2/d_model))
            div_term = torch.exp(torch.arange(0, d_model//2, 2) * (-math.log(10000.0) / (d_model//2)))
            div_term = div_term[:, None, None]  # [C//4, 1, 1]
            pe[0::4, :, :] = torch.sin(x_position * div_term) # [32, 256, 256]
            pe[1::4, :, :] = torch.cos(x_position * div_term) # [32, 256, 256]
            pe[2::4, :, :] = torch.sin(y_position * div_term) # [32, 256, 256]
            pe[3::4, :, :] = torch.cos(y_position * div_term) # [32, 256, 256]
            self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def contract(self, x):
        """Contracts points towards the origin (Eq 10 of arxiv.org/abs/2111.12077).
            ref: coord.py of https://github.com/google-research/multinerf/blob/main/internal/coord.py
            This function contract magnitude of 0~inf. to 0~2. Thus numerically the return should be between
            return: z: -2~2
        """
        x_norm = torch.clamp(x.abs().amax(dim=-1, keepdim=True), 1e-6)
        z = torch.where(x_norm <= 1, x, ((2 * x_norm - 1) / (x_norm ** 2)) * x)
        return z

    def inv_contract(self, z):
        """The inverse of contract().
            ref: coord.py of https://github.com/google-research/multinerf/blob/main/internal/coord.py
        """
        z_norm = torch.clamp(z.abs().amax(dim=-1, keepdim=True), 1e-6)
        x = torch.where(z_norm <= 1, z, z / (2 * z_norm - z_norm**2))
        return x

    def dynamic_pixel_pe_focal_norm(self, N, intrinsic):
        '''
        perform dynamically pixel position encoding at run time, normalized by focal length and center principle points
        N: batch size
        intrinsic: [B,3,3]
        '''

        y_position = copy.deepcopy(self.y_position).to(intrinsic.device) # 1~480
        x_position = copy.deepcopy(self.x_position).to(intrinsic.device) # 1~640

        y_position = repeat(y_position, 'n1 h w -> (n1 n2) h w', n2=N).contiguous() # N is batch number
        x_position = repeat(x_position, 'n1 h w -> (n1 n2) h w', n2=N).contiguous()

        # the x,y position, needs to be recentered to the principle points
        y_position = y_position - 0.5 - intrinsic[:,1,2][:,None,None] # ppy -239.5 to 239.5 for none random crop image
        x_position = x_position - 0.5 - intrinsic[:,0,2][:,None,None] # ppx -319.5 to 319.5

        # normalize with focal length
        y_position = (y_position/intrinsic[:,1,1][:,None,None])[:,None,::] # fy, [N, 1, H, W], within [-1., +1]. experimental value
        x_position = (x_position/intrinsic[:,0,0][:,None,None])[:,None,::] # fx, [N, 1, H, W]

        # subsample x,y position for SC maps
        y_position = y_position[:,:,0::8,0::8].contiguous() # (0,0) (0,8),...
        x_position = x_position[:,:,0::8,0::8].contiguous()

        # upscale x, y position magnitude, heuristic implementation
        y_position = y_position * 400
        x_position = x_position * 400

        pe = copy.deepcopy(self.pe[:,0::8,0::8]).to(intrinsic.device)
        pe = repeat(pe, 'n1 h w -> n2 n1 h w', n2=N).clone()#.contiguous() # [N, d_model, H, W]
        self.div_term = self.div_term.to(intrinsic.device) # [1, C//4, 1, 1]
        pe[:,0::4, :, :] = torch.sin(x_position * self.div_term)
        pe[:,1::4, :, :] = torch.cos(x_position * self.div_term)
        pe[:,2::4, :, :] = torch.sin(y_position * self.div_term)
        pe[:,3::4, :, :] = torch.cos(y_position * self.div_term)

        return pe

    def forward_default(self, x, intrinsic=None):
        return x + self.pe[:, :, :x.size(2), :x.size(3)]

    def hard_clip(self, x):
        ''' marepo implementation on hard clipping out of range input '''
        x = torch.clip(x, min=-self.oor, max=self.oor)
        x = (x / self.oor) * math.pi
        return x

    def apply_contract(self, x):
        ''' mip-nerf-360 inspired contract implementation '''
        x = (x / self.oor)
        x = self.contract(x) * math.pi / 2 # make sure values are within +/- pi
        return x

    def forward_nerf(self, x, intrinsic=None):
        '''
        Encode scene coordinates with nerf encoding,
        and selectively with pixel position encoding
        return:
            x: NeRF embedded SC +  pixel P.E.
            pixel_pe: pixel P.E. map
        '''

        N, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(N*H*W, C).contiguous()  # [N*H*W,C]

        # # print some interesting stats
        # self.compute_scene_coordinate_stats(x,100)
        # self.compute_scene_coordinate_stats(x,50)
        # self.compute_scene_coordinate_stats(x,20)

        # we want to normalize the input to within +- pi
        if self.use_contract:
            x = self.apply_contract(x)
        else:
            x = self.hard_clip(x)

        # NeRF embedding on SC
        x = self.pre_pe_embedding(x)
        new_C = x.shape[1]
        x = x.reshape(N, H, W, new_C).permute(0, 3, 1, 2).contiguous()

        x = self.pre_pe_dim_manipulator(x) # [N, d_model, H, W]

        if self.pixel_pe == 'focal_norm': # focal norm encoding with SC only
            pe = self.dynamic_pixel_pe_focal_norm(N, intrinsic)
            pixel_pe = pe[:, :, :x.size(2), :x.size(3)].contiguous()
            x = x + pixel_pe
        else: # no focal norm encoding
            pixel_pe = self.pe[:, :, :x.size(2), :x.size(3)].contiguous()#.type(x.dtype)
            x = x + pixel_pe
        return x, pixel_pe

    def forward(self, x, intrinsic=None):
        """
        Args:
            x: [N, C, H, W]
        return
            x: [N, d_model, H, W]
            intrinsic: [N,3,3]
        """

        x, pixel_pe = self.forward_fn(x, intrinsic) # only return encoded scene coordinates
        return x, pixel_pe

    def compute_scene_coordinate_stats(self, x, OOR_value):
        """
        compute stats of manually set out of range scene coordinate stats
        OOR_value:
        """
        xx = x[:, 0]
        xy = x[:, 1]
        xz = x[:, 2]
        print("let's assume scene coordinates >{} or <-{} are invalid".format(OOR_value, OOR_value))
        xx_OOR_high = xx[(xx > OOR_value)].shape[0]
        xx_OOR_low = xx[(xx < -OOR_value)].shape[0]
        xy_OOR_high = xy[(xy > OOR_value)].shape[0]
        xy_OOR_low = xy[(xy < -OOR_value)].shape[0]
        xz_OOR_high = xz[(xz > OOR_value)].shape[0]
        xz_OOR_low = xz[(xz < -OOR_value)].shape[0]
        print("percentage x coordinate out of range: {}".format((xx_OOR_high + xx_OOR_low) / x.shape[0]))
        print("percentage y coordinate out of range: {}".format((xy_OOR_high + xy_OOR_low) / x.shape[0]))
        print("percentage z coordinate out of range: {}".format((xz_OOR_high + xz_OOR_low) / x.shape[0]))

    def vis_pixel_pe(self, pe, folder):
        if torch.is_tensor(pe):
            pe = pe.cpu().detach().numpy()

        if pe.ndim==4:
            pe = pe[0] # choose 0th frame to visualize
        if not osp.exists(f'tmp/{folder}'):
            # If it doesn't exist, create it
            os.makedirs(f'tmp/{folder}')

        fig = plt.figure(figsize=(15, 4))
        for i in range(pe.shape[0]):
            cax = plt.matshow(pe[i, :, :])
            plt.gcf().colorbar(cax)
            plt.savefig(f'tmp/{folder}/pe{i:01d}.png')
            plt.close()

        print("visualized pe to tmp/pe")

    def plot_pixel_pe_with_diff_intrinsics(self, N, intrinsic, folder, index):
        '''
        N: batch size
        intrinsic: [B,3,3]

        '''
        # visualize pixel_pe with customized focal length
        for focal in range(450, 600, 1):
            intrinsic[:,1,1] = focal
            intrinsic[:,0,0] = focal

            pe = self.dynamic_pixel_pe_focal_norm(N, intrinsic)

            if torch.is_tensor(pe):
                pe = pe.cpu().detach().numpy()

            if pe.ndim==4:
                pe = pe[0] # choose 0th frame to visualize

            if not osp.exists(f'tmp/{folder}'):
                # If it doesn't exist, create it
                os.makedirs(f'tmp/{folder}')

            fig = plt.figure(figsize=(15, 4))
            cax = plt.matshow(pe[index, :, :])
            plt.gcf().colorbar(cax)
            plt.savefig(f'tmp/{folder}/pe{index:01d}_f{focal:03d}.png')
            plt.close()

            



