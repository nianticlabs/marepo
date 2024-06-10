#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2024.
import torch
import logging
from ace.ace_network import Regressor
from marepo.load_dataset_marepo import load_single_scene, load_multiple_scene, load_single_map_scene
from marepo.util import Custom_RandomCrop
from marepo.marepo_vis_util import save_image_saliancy, UnNormalize
import sys, time, os
import os.path as osp
import numpy as np
import copy
import random

import argparse
from pathlib import Path
from distutils.util import strtobool

import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from torch.cuda.amp import autocast

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

    parser.add_argument('--not_mapfree', type=_strtobool, default=False,
                        help='A temp fix for data augmentation that is not mapfree data')

    parser.add_argument('--scheme2', type=_strtobool, default=False,
                        help='use scheme2 to save SC maps (subtract mean) and GT pose (subtract mean)')
    parser.add_argument('--scheme2_aug_train_only', type=_strtobool, default=False,
                        help='only apply scheme2 augmentation to training set')
    parser.add_argument('--scheme3', type=_strtobool, default=False,
                        help='use data augmentation to generate SC maps (subtract mean) and GT pose (subtract mean)'
                             'considering combine with scheme2 to produce 1 set of no aug. data + N set of aug. data')
    parser.add_argument('--scheme3_aug_number', nargs='+', type=int, default=[0, 16],
                        help='define the group number of aug. data to be generated. [incl, excl], i.e. [0,4], [4,8]')
    parser.add_argument('--scheme3_aug_train_only', type=_strtobool, default=False,
                        help='only apply scheme3 augmentation to training set')
    parser.add_argument('--create_mapping_buffer', type=_strtobool, default=False,
                        help='create mapping buffer for each scene like ACE, but save them in advance')

    ### for pitch 3 creating mapping buffer ###
    parser.add_argument('--use_aug', type=_strtobool, default=True,
                        help='Use any augmentation.')
    parser.add_argument('--aug_rotation', type=int, default=15,
                        help='max inplane rotation angle')

    parser.add_argument('--aug_scale', type=float, default=1.5,
                        help='max scale factor')

    ### create_pose_bound_support_set, for scene specific normalization
    parser.add_argument('--create_pose_bound_support_set', type=_strtobool, default=True,
                        help='if True, we generate pose bound support set.')
    parser.add_argument('--support_set_path', type=str, default='',
                        help='designate where to store, i.e.: '
                             '/home/shuaichen_nianticlabs_com/storage/map_free_training_scenes_aug_16_support_set')

    # Clustering params, for the ensemble training used in the Cambridge experiments. Disabled by default.
    parser.add_argument('--num_clusters', type=int, default=None,
                        help='split the training sequence in this number of clusters. disabled by default')

    parser.add_argument('--cluster_idx', type=int, default=None,
                        help='train on images part of this cluster. required only if --num_clusters is set.')

    parser.add_argument('--center_crop', type=_strtobool, default=False,
                        help='Flag for datasetloader indicating images need center crop to make them proportional in size to MapFree data')
    return parser.parse_args()

def set_seed(seed):
    """
    Seed all sources of randomness.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class TrainerACE():
    def __init__(self, options):

        self.options = options
        self.device = torch.device('cuda')

        # Setup randomness for reproducibility.
        self.base_seed = 2089
        set_seed(self.base_seed)

        # Create network using the state dict of the pretrained encoder.
        encoder_state_dict = torch.load(self.options.encoder_path, map_location="cpu")
        _logger.info(f"Loaded pretrained encoder from: {self.options.encoder_path}")
        head_state_dict = torch.load(self.options.head_network_path, map_location="cpu") # this model is dynamically load during preprocessing
        _logger.info(f"Loaded pretrained head weights from: {self.options.head_network_path}")

        # Notice that this is the original ACE network
        self.regressor = Regressor.create_from_split_state_dict(encoder_state_dict, head_state_dict)
        self.regressor = self.regressor.to(self.device)

        # Create train/val/test dataset.
        if self.options.dataset_path == "":  # load single scene
            self.train_dl, self.val_dl, self.test_dl = load_single_scene(self.options)
        else:  # load all scenes
            self.train_dl, self.val_dl, self.test_dl = load_multiple_scene(self.options)

        # # When preprocessing, we don't need the dataset mean cam center but individual scene's cam center
        # _logger.info("Loaded training scan from: {} -- {} images, mean: {:.2f} {:.2f} {:.2f}".format(
        #     options.dataset_path,
        #     len(self.train_dl),
        #     self.train_dl.dataset.mean_cam_center[0],
        #     self.train_dl.dataset.mean_cam_center[1],
        #     self.train_dl.dataset.mean_cam_center[2]))

        # Generator used to sample random features (runs on the GPU).
        self.sampling_generator = torch.Generator(device=self.device)
        self.sampling_generator.manual_seed(self.base_seed + 4095)

    def check_n_load_new_head(self, fname, cur_ckpt, split="train"):
        '''
        For each new scene, we need to reload the ACE head checkpoint.
        fname: rgb filename from dataloader
        split: type of dataset: "train", "val", "test", should be same as the dataset loader

        '''
        # parse data scene and model checkpoint
        rgb_dirname = osp.dirname(fname[0])  # ex: PATH_TO_map_free_training_scenes/train/mapfree_s00225/train/rgb/, frame_00100.jpg
        map_query_dir = osp.dirname(rgb_dirname)
        scene_name, self.map_query = osp.split(map_query_dir)
        self.scene = osp.basename(scene_name)  # ex: mapfree_s00225

        # verify if we need to reload new head checkpoint
        if cur_ckpt != (self.scene + '.pt'):
            cur_ckpt = self.scene + '.pt'
            ckpt_weight = osp.join(self.options.dataset_head_network_path, split, self.scene, cur_ckpt)
            # print(fname[0])
            print("reload new head network ckpt:", ckpt_weight)
            self.regressor.load_head(ckpt_weight)
            self.regressor.heads.eval()
        return cur_ckpt

    def save_sc_to_file(self, sc, fname):
        """
        save scene coordinates to dataset
        """
        rgb_dirname, rgb_filename = osp.split(fname[0])
        map_query_dir = osp.dirname(rgb_dirname)
        save_dir = map_query_dir + "/scene_coord"
        if not osp.isdir(save_dir):
            os.makedirs(save_dir)
        rgb_filename_no_extension, ext = osp.splitext(rgb_filename)
        save_filename = osp.join(save_dir, rgb_filename_no_extension + ".npy")
        np.save(save_filename, sc)

    def save_sc_n_pose_to_file(self, sc, pose, mean, fname):
        """
        save scene coordinates and poses to dataset
        """
        rgb_dirname, rgb_filename = osp.split(fname[0])
        map_query_dir = osp.dirname(rgb_dirname)

        # save sc map
        save_dir1 = map_query_dir + "/scene_coord_sub_mean"
        if not osp.isdir(save_dir1):
            os.makedirs(save_dir1)
        rgb_filename_no_extension, ext = osp.splitext(rgb_filename)
        save_filename = osp.join(save_dir1, rgb_filename_no_extension + ".npy")
        np.save(save_filename, sc)

        # save gt pose
        save_dir2 = map_query_dir + "/poses_sub_mean"
        if not osp.isdir(save_dir2):
            os.makedirs(save_dir2)

        if self.options.not_mapfree:
            pose_filename_no_extension = "pose_"+rgb_filename_no_extension+".txt"

        else:
            pose_filename_no_extension = "pose_"+rgb_filename_no_extension[6:]+".txt" # ex: pose_00580.txt
        save_pose_filename = osp.join(save_dir2, pose_filename_no_extension)
        np.savetxt(save_pose_filename, pose)

        # save mean stats
        save_dir3 = map_query_dir + "/mean_stats"
        if not osp.isdir(save_dir3):
            os.makedirs(save_dir3)
        save_stats_filename = osp.join(save_dir3, rgb_filename_no_extension + ".txt")
        np.savetxt(save_stats_filename, mean)

    def save_aug_img_n_sc_n_pose_to_file(self, img, sc_mask, sc, pose, intrinsics, fname, gnum):
        """
        save augmented image and scene coordinates maps and poses to file
        img: [B,3,H,W]
        sc_mask: [B,3,H//8,W//8]
        sc: [B,3,H//8,W//8]
        pose: [4,4]
        intrinsics: [3,3]
        fname: filename
        gnum: group number
        """
        rgb_dirname, rgb_filename = osp.split(fname[0])
        map_query_dir = osp.dirname(rgb_dirname)

        # save augmented sc map
        save_dir0 = map_query_dir + f"/aug/rgb_{gnum:03d}"  # should be sth like this: map_free_training_scenes/train/mapfree_s00000/train/aug/sc_000
        if not osp.isdir(save_dir0):
            os.makedirs(save_dir0)
        rgb_filename_no_extension, ext = osp.splitext(rgb_filename)
        save_filename = osp.join(save_dir0, rgb_filename_no_extension + ".jpg")
        save_image(img, save_filename)

        # save augmented sc map
        save_dir1 = map_query_dir + f"/aug/sc_{gnum:03d}" # should be sth like this: map_free_training_scenes/train/mapfree_s00000/train/aug/sc_000
        if not osp.isdir(save_dir1):
            os.makedirs(save_dir1)
        rgb_filename_no_extension, ext = osp.splitext(rgb_filename)
        save_filename = osp.join(save_dir1, rgb_filename_no_extension + ".npy")
        np.save(save_filename, sc)

        # save augmented gt pose
        save_dir2 = map_query_dir + f"/aug/poses_{gnum:03d}" # should be sth like this: map_free_training_scenes/train/mapfree_s00000/train/aug/poses_000
        if not osp.isdir(save_dir2):
            os.makedirs(save_dir2)

        if self.options.not_mapfree:
            pose_filename_no_extension = "pose_"+rgb_filename_no_extension+".txt"
        else:
            pose_filename_no_extension = "pose_" + rgb_filename_no_extension[6:] + ".txt"  # ex: pose_00580.txt
        save_pose_filename = osp.join(save_dir2, pose_filename_no_extension)
        np.savetxt(save_pose_filename, pose)

        # save augmented sc mask
        save_dir3 = map_query_dir + f"/aug/sc_mask_{gnum:03d}"  # should be sth like this: map_free_training_scenes/train/mapfree_s00000/train/aug/sc_mask_000
        if not osp.isdir(save_dir3):
            os.makedirs(save_dir3)
        rgb_filename_no_extension, ext = osp.splitext(rgb_filename)
        save_filename = osp.join(save_dir3, rgb_filename_no_extension + ".npy")
        np.save(save_filename, sc_mask)

        # save augmented sc mask
        save_dir4 = map_query_dir + f"/aug/intrinsics_{gnum:03d}"  # should be sth like this: map_free_training_scenes/train/mapfree_s00000/train/aug/sc_mask_000
        if not osp.isdir(save_dir4):
            os.makedirs(save_dir4)

        if self.options.not_mapfree:
            intrinsic_filename_no_extension = "intrinsic_" + rgb_filename_no_extension+".txt"
            save_filename = osp.join(save_dir4, intrinsic_filename_no_extension)
            np.savetxt(save_filename, intrinsics)
        else:
            intrinsic_filename_no_extension = "intrinsic_" + rgb_filename_no_extension[6:] + ".txt"  # ex: intrinsic_00580.txt # Legacy: we previously saved bad suffix
            save_filename = osp.join(save_dir4, intrinsic_filename_no_extension)
            # np.save(save_filename, intrinsics) # Legacy: should've been savetxt
            np.savetxt(save_filename, intrinsics)

    def subtract_means_per_scene(self, scene_coordinates_B3HW, gt_pose_B44):
        # store sc maps that subtracts mean
        sc_to_save = (scene_coordinates_B3HW - self.regressor.heads.mean).float().cpu().numpy()
        # store gt pose that subtracts mean
        gt_pose_to_save = copy.deepcopy(gt_pose_B44)
        gt_pose_to_save[:, :3, 3] = gt_pose_to_save[:, :3, 3] - self.regressor.heads.mean[:, :, 0,
                                                                0].float().cpu()
        gt_pose_to_save = gt_pose_to_save[0].numpy()  # [4,4]
        # store mean for the record
        scene_mean_stats = self.regressor.heads.mean.cpu().numpy()[0, :, 0, 0]  # [3]
        return sc_to_save, gt_pose_to_save, scene_mean_stats

    def sc_map_generation_one_epoch(self, dl, split, group_num=0):

        cur_ckpt = ""  # current heads checkpoint
        self.regressor.eval()

        unnorm = UnNormalize([0.4], [0.25])
        with torch.no_grad():
            for batch_idx, batch in enumerate(dl):
                image_B1HW, image_mask_B1HW = batch['image'], batch['image_mask']
                gt_pose_B44, gt_pose_inv_B44 = batch['pose'], batch['pose_inv']
                intrinsics_B33, intrinsics_inv_B33 = batch['intrinsics'], batch['intrinsics_inv']
                scene_coords_B13HW = batch['scene_coords']
                sc_mask = batch['sc_mask'] if 'sc_mask' in batch.keys() else None
                fname = batch['rgb_files']

                ### for each new scene, we need to reload the ACE head checkpoint ###
                cur_ckpt = self.check_n_load_new_head(fname, cur_ckpt, split)

                if self.options.scheme3:
                    # random crop data to fixed size, add paddings if necessary
                    target_H = dl.dataset.default_img_H  # 480
                    target_W = dl.dataset.default_img_W  # 640

                    aug_H, aug_W = image_B1HW.shape[2], image_B1HW.shape[3]
                    if DEBUG:
                        print("target size: ", target_H, target_W, "aug data size: ", aug_H, aug_W)
                        print("intrinsics_B33[0]", intrinsics_B33[0])
                    # un-normalize image for backup reasons. Values are in [0,1.]
                    unnorm_image = unnorm(copy.deepcopy(image_B1HW))  # this is grey scale [1,1,480,640]

                    # debug: save image before random_crop
                    if DEBUG:
                        tmp = copy.deepcopy(unnorm_image)
                        tmp[:, :, int(intrinsics_B33[0,1, 2]):int(intrinsics_B33[0,1, 2]) + 1,
                        int(intrinsics_B33[0,0, 2]):int(intrinsics_B33[0,0, 2]) + 1] = 1.0
                        save_image(tmp, f'tmp/{batch_idx:03d}_before_crop.png')
                    # end debug

                    # This is to handle customized random_crop and shifts principle point,
                    random_crop = Custom_RandomCrop((target_H, target_W), (aug_H, aug_W), intrinsics_B33[0])
                    image_B1HW_random_crop = random_crop(image_B1HW) # this is the image to send to the ACE regressor
                    unorm_image_random_crop = unnorm(copy.deepcopy(image_B1HW_random_crop)) # this is the unormalized and cropped image to be saved to file

                    # debug: save image after random_crop
                    if DEBUG:
                        tmp = copy.deepcopy(unorm_image_random_crop)
                        tmp[:, :, int(random_crop.intrinsics[1, 2]):int(random_crop.intrinsics[1, 2]) + 1,
                        int(random_crop.intrinsics[0, 2]):int(random_crop.intrinsics[0, 2]) + 1] = 1.0
                        save_image(tmp, f'tmp/{batch_idx:03d}_after_crop.png')
                    # end debug

                    # record intrinsics
                    new_intrinsics = random_crop.intrinsics  # get updated intrinsics
                    # compute scene coordinates
                    scene_coordinates_B3HW = self.regressor(image_B1HW_random_crop.to(self.device)) # mod0 fix, make sure the input image is normalized like ACE

                else:
                    # compute scene coordinates
                    scene_coordinates_B3HW = self.regressor(image_B1HW.to(self.device))  # [1,3,60,80], image_B1HW is [1, 1, 480, 640] w/o augmentation

                # save scene coordinates to file
                if self.options.scheme2:
                    sc_to_save, gt_pose_to_save, scene_mean_stats = self.subtract_means_per_scene(scene_coordinates_B3HW, gt_pose_B44)
                    self.save_sc_n_pose_to_file(sc_to_save, gt_pose_to_save, scene_mean_stats, fname)
                elif self.options.scheme3:

                    # crop image_mask like image data
                    image_mask = random_crop(image_mask_B1HW)

                    # debug: save mask after random_crop
                    if DEBUG:
                        save_image(image_mask.float(), f'tmp/{batch_idx:03d}_after_crop_mask.png')
                    # end debug

                    # # The sc_mask needs to be downsampled to the actual output resolution and cast to bool.
                    sc_mask = TF.resize(image_mask, [scene_coordinates_B3HW.shape[2], scene_coordinates_B3HW.shape[3]], interpolation=TF.InterpolationMode.NEAREST) # for ACE, SC map is downsampled by 8
                    sc_mask = sc_mask.bool() # [1,3,60,80]

                    # sc_crop = random_crop.sc_map_crop(scene_coordinates_B3HW, df=8)
                    sc_crop_sub_mean, gt_pose_to_save, scene_mean_stats = self.subtract_means_per_scene(scene_coordinates_B3HW, gt_pose_B44)
                    sc_to_save = sc_mask*sc_crop_sub_mean # apply mask to ensure padding remains to be padding (0) # [1,3,60,80]

                    # debug:
                    if DEBUG:
                        save_image_saliancy(sc_to_save, f'tmp/{batch_idx:03d}_sc_map.png', True, True)
                        breakpoint()
                    # end debug

                    self.save_aug_img_n_sc_n_pose_to_file(unorm_image_random_crop, sc_mask, sc_to_save, gt_pose_to_save, new_intrinsics, fname, group_num)
                else:
                    scene_coordinates_B3HW = scene_coordinates_B3HW.float().cpu().numpy()
                    self.save_sc_to_file(scene_coordinates_B3HW, fname)

    def create_sc_maps(self, split="train"):
        """
        generate scene coordinate maps and save them into corresponding directories
        """
        # Disable benchmarking, since we have variable tensor sizes.
        torch.backends.cudnn.benchmark = False
        if split == "train":
            dl = self.train_dl
        elif split == "val":
            dl = self.val_dl
        elif split == "test":
            dl = self.test_dl
        else:
            print("unknown dataset split, please check if split is correct")
            sys.exit()

        if self.options.scheme3: # may generate data for multiple loops
            assert(split=="train" or split=="val") # should only use for training set
            for group_num in range(self.options.scheme3_aug_number[0], self.options.scheme3_aug_number[1]):
                self.sc_map_generation_one_epoch(dl, split, group_num)
        else: # assume rest mode only generate data for one loop
            self.sc_map_generation_one_epoch(dl, split)

    def check_to_save_max_scene_bound(self, pre_ckpt, cur_ckpt, prev_scene_dir, max_scene_pose_bounds, fname):
        # save the max_scene_pose_bounds for previous scene
        print("cur_ckpt", cur_ckpt)
        if pre_ckpt != "":
            os.makedirs(prev_scene_dir, exist_ok=True)
            np.savetxt(prev_scene_dir + 'scene_bound.txt', max_scene_pose_bounds)
            print("saving scene_bounds.txt to ", prev_scene_dir)

        # re-initialize stuff
        max_scene_pose_bounds = [0., 0., 0.]
        pre_ckpt = cur_ckpt

        # ['', 'home', 'shuaichen_nianticlabs_com', 'storage', 'map_free_training_scenes_aug_16', 'train', 'mapfree_s00001', 'train', 'rgb', 'frame_00420.jpg']
        file_name_splits = fname[0].split('/')
        prev_scene_dir = '/'.join(file_name_splits[-5:-3])  # 'train/mapfree_s00001'

        prev_scene_dir = self.options.support_set_path + '/' + prev_scene_dir + '/train/scene_bound/'
        return pre_ckpt, cur_ckpt, prev_scene_dir, max_scene_pose_bounds

    def generate_pose_bound_support_dataset(self, split="train"):
        '''
        here we only compute pose bound based on the mapping sequence, not query sequence?
        '''
        pre_ckpt = ""  # current heads checkpoint
        torch.backends.cudnn.benchmark = False
        if split == "train":
            dl = self.train_dl
        elif split == "val":
            dl = self.val_dl
        elif split == "test":
            dl = self.test_dl
        else:
            print("unknown dataset split, please check if split is correct")
            sys.exit()

        max_scene_pose_bounds=[0.,0.,0.]
        prev_scene_dir= ''
        with torch.no_grad():
            for batch_idx, batch in enumerate(dl):
                gt_pose_B44, gt_pose_inv_B44 = batch['pose'], batch['pose_inv']
                fname = batch['rgb_files']

                current_split = fname[0].split('/')[-3]
                if current_split == 'test':
                    # we skip the query set so that we only compute the scene bounds from the mapping sequence
                    continue

                ### for each new scene, we need to reload the ACE head checkpoint ###
                cur_ckpt = self.check_n_load_new_head(fname, pre_ckpt, split)
                if pre_ckpt != cur_ckpt:
                    pre_ckpt, cur_ckpt, prev_scene_dir, max_scene_pose_bounds = self.check_to_save_max_scene_bound(pre_ckpt,
                                                                                                                   cur_ckpt,
                                                                                                                   prev_scene_dir,
                                                                                                                   max_scene_pose_bounds,
                                                                                                                   fname)
                dummy=torch.zeros(1,3,1,1).to(self.device)
                sc_to_save, gt_pose_to_save, scene_mean_stats = self.subtract_means_per_scene(dummy, gt_pose_B44)
                scene_pose = gt_pose_to_save[:3,3]
                # print("scene_pose", scene_pose)
                for i in range(len(max_scene_pose_bounds)):
                    max_scene_pose_bounds[i] = max(max_scene_pose_bounds[i], abs(scene_pose[i]))
                # print("max_scene_pose_bounds", max_scene_pose_bounds)
            # save the last scene
            pre_ckpt, cur_ckpt, prev_scene_dir, max_scene_pose_bounds = self.check_to_save_max_scene_bound(pre_ckpt,
                                                                                                           cur_ckpt,
                                                                                                           prev_scene_dir,
                                                                                                           max_scene_pose_bounds,
                                                                                                           fname)

    def store_mapping_buffer_features(self, path_to_scene_prefix, scene_name, buffer_idx, num_store_feature_map=10):
        '''
        randomly sample mapping features and store mapping buffer features to dataset
        path_to_scene_prefix: PATH before the scene folder
        scene_name: name of the current scene
        buffer_idx: buffer_idx-1 is the end index of current self.training_buffer
        num_store_feature_map: number of feature maps to store, default 10
        '''
        inputNumbers = range(0, buffer_idx - 1)
        for i in range(num_store_feature_map):
            sample_indices = random.sample(inputNumbers, 4800)
            mapping_features = self.training_buffer['features'][sample_indices]
            # save mapping features to file
            save_dir = path_to_scene_prefix + '/' + scene_name + '/train/mapping_buffer'
                       #f"/rgb_{i:02d}"  # should be sth like this: map_free_training_scenes/train/mapfree_s00000/train/aug/sc_000
            if not osp.isdir(save_dir):
                os.makedirs(save_dir)

            save_filename = save_dir + f'/mapping_feature_{i:02d}' + ".npy"
            np.save(save_filename, mapping_features.cpu())


    def create_mapping_buffer(self, split="train"):
        '''
        create mapping buffer and randomly sample mapping features
        split: "train", or "val", or "test"
        '''
        # Disable benchmarking, since we have variable tensor sizes.
        torch.backends.cudnn.benchmark = False
        torch.multiprocessing.set_sharing_strategy('file_system')

        root_dir = Path(self.options.dataset_path)/split
        scene_dirs = sorted(root_dir.iterdir())

        for scene_idx, scene_dir in enumerate(scene_dirs):
            if scene_idx<0:
                #resume from runtime error
                continue
            dl, map_dataset = load_single_map_scene(self.options, scene_dir)

            training_buffer_size = 8000000
            samples_per_image=1024
            _logger.info("Starting creation of the training buffer.")

            # Create a training buffer that lives on the GPU.
            self.training_buffer = {
                'features': torch.empty((training_buffer_size, self.regressor.feature_dim), # training_buffer_size: 8000000, feature_dim: 512
                                        dtype=(torch.float32, torch.float16)[self.options.use_half], device=self.device)
            }

            # Features are computed in evaluation mode.
            self.regressor.eval()


            # The encoder is pretrained, so we don't compute any gradient.
            with torch.no_grad():
                # Iterate until the training buffer is full.
                buffer_idx = 0
                dataset_passes = 0
                buffer_idx_counter = 0  # counter*1million sampled feature points

                while buffer_idx < training_buffer_size:
                    dataset_passes += 1
                    for batch in dl:

                        image_B1HW, image_mask_B1HW = batch['image'], batch['image_mask']

                        # Copy to device.
                        image_B1HW = image_B1HW.to(self.device, non_blocking=True)
                        image_mask_B1HW = image_mask_B1HW.to(self.device, non_blocking=True)

                        # Compute image features.
                        with autocast(enabled=self.options.use_half):
                            features_BCHW = self.regressor.get_features(image_B1HW) # [1, 512, 87, 116]

                        # Dimensions after the network's downsampling.
                        B, C, H, W = features_BCHW.shape

                        # The image_mask needs to be downsampled to the actual output resolution and cast to bool.
                        image_mask_B1HW = TF.resize(image_mask_B1HW, [H, W], interpolation=TF.InterpolationMode.NEAREST)
                        image_mask_B1HW = image_mask_B1HW.bool()

                        # If the current mask has no valid pixels, continue.
                        if image_mask_B1HW.sum() == 0:
                            continue

                        def normalize_shape(tensor_in):
                            """Bring tensor from shape BxCxHxW to NxC"""
                            return tensor_in.transpose(0, 1).flatten(1).transpose(0, 1)

                        batch_data = {
                            'features': normalize_shape(features_BCHW), # [1, 512, 87, 116] -> [10092, 512]
                        }

                        # Turn image mask into sampling weights (all equal).
                        image_mask_B1HW = image_mask_B1HW.float()
                        image_mask_N1 = normalize_shape(image_mask_B1HW)

                        # Over-sample according to image mask.
                        features_to_select = samples_per_image * B
                        features_to_select = min(features_to_select, training_buffer_size - buffer_idx)

                        # Sample indices uniformly, with replacement.
                        sample_idxs = torch.multinomial(image_mask_N1.view(-1),
                                                        features_to_select,
                                                        replacement=True,
                                                        generator=self.sampling_generator)

                        # Select the data to put in the buffer.
                        for k in batch_data:
                            batch_data[k] = batch_data[k][sample_idxs]

                        # Write to training buffer. Start at buffer_idx and end at buffer_offset - 1.
                        buffer_offset = buffer_idx + features_to_select

                        for k in batch_data:
                            self.training_buffer[k][buffer_idx:buffer_offset] = batch_data[k]
                        buffer_idx = buffer_offset

                        # logging current loading status
                        if buffer_idx // 2000000 > buffer_idx_counter:
                            buffer_idx_counter = buffer_idx // 2000000
                            print("filling training buffers with {}/{} samples".format(buffer_idx_counter * 2000000,
                                                                                       training_buffer_size))
                        if buffer_idx >= training_buffer_size:
                            break

                # store mapping buffer features in the last batch
                filenames = batch['rgb_files'] # ex: 'PATH/mapfree_s00000/train/rgb/frame_00000.jpg'
                path_to_scene_prefix, scene = osp.split(osp.split(osp.split(osp.split(filenames[0])[0])[0])[0])
                # print("scene: ", scene)
                self.store_mapping_buffer_features(path_to_scene_prefix, scene, buffer_idx)

            del self.training_buffer
            del dl, map_dataset # avoid too many files open OS error
            torch.cuda.empty_cache()


    def preprocess_Ace_dataset(self):
        preprocess_time = 0.
        print("start preprocess...")
        preprocess_start_time = time.time()

        assert((self.options.scheme2 and self.options.scheme3) == False) # can't use scheme2 and scheme3 at same time

        if self.options.scheme2:
            self.create_sc_maps("train")
            if not self.options.scheme2_aug_train_only:
                self.create_sc_maps("val")
                self.create_sc_maps("test")
        elif self.options.scheme3:
            self.create_sc_maps("train")
            if not self.options.scheme3_aug_train_only:
                self.create_sc_maps("val")
        else:
            NotImplementedError
        preprocess_end_time = time.time()
        preprocess_time += preprocess_end_time - preprocess_start_time
        _logger.info(f"Finish scene coordinate generation in {preprocess_end_time - preprocess_start_time:.1f}s.")
        print("finished the sc generation preprocess, exiting for now...")
        sys.exit()

if __name__ == '__main__':

    # Setup logging levels.
    logging.basicConfig(level=logging.INFO)
    options = get_opts()
    ACE_Preprocessor = TrainerACE(options)

    # we first generate all the scene coordinates files using pretrained ACE networks
    ACE_Preprocessor.preprocess_Ace_dataset()
    sys.exit()
