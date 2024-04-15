import logging
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from skimage import color
from skimage import io
import imageio

from skimage.transform import rotate, resize
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from scipy.spatial.transform import Rotation as R
import os.path as osp

_logger = logging.getLogger(__name__)

SPIKE_NOISE=False
USE_AUG_16=True # if True, we use 16 aug + 1 un-aug training data. If False, default 8 aug + 1 un-aug data
# random.seed(2024) # try to be deterministic for unit tests
class CamLocDatasetAll(Dataset):
    """Camera localization dataset,
    unlike ACE's CamLocDataset, we load data of all scenes in the datasets.

    Access to image, calibration and ground truth data given a dataset directory.
    """

    def __init__(self,
                 root_dir,
                 mode=0,
                 sparse=False,
                 augment=False,
                 aug_rotation=15,
                 aug_scale_min=2 / 3,
                 aug_scale_max=3 / 2,
                 aug_black_white=0.1,
                 aug_color=0.3,
                 image_height=480,
                 use_half=True,
                 num_clusters=None,
                 cluster_idx=None,
                 split=["map"],
                 trainskip=1,
                 load_sc_map=True,
                 load_scheme2_sc_map=False,
                 load_scheme3_sc_map=False,
                 marepo_sc_augment=False,
                 jitter_trans=1.0,
                 jitter_rot=180,
                 load_mapping_buffer_features=False,
                 random_select_mapping_buffer=False,
                 all_mapping_buffers=False,
                 non_mapfree_dataset_naming=False,
                 center_crop=False
                 ):
        """Constructor.

        Parameters:
            root_dir: Folder of the data (training or test).
            mode:
                0 = RGB only, load no initialization targets. Default for the ACE paper.
                1 = RGB + ground truth scene coordinates, load or generate ground truth scene coordinate targets
                2 = RGB-D, load camera coordinates instead of scene coordinates
            sparse: for mode = 1 (RGB+GT SC), load sparse initialization targets when True, load dense depth maps and
                generate initialization targets when False
            augment: Use random data augmentation for preprocess, note: not supported for mode = 2 (RGB-D) since pre-generated eye
                coordinates cannot be augmented
            aug_rotation: Max 2D image rotation angle, sampled uniformly around 0, both directions, degrees.
            aug_scale_min: Lower limit of image scale factor for uniform sampling
            aug_scale_min: Upper limit of image scale factor for uniform sampling
            aug_black_white: Max relative scale factor for image brightness/contrast sampling, e.g. 0.1 -> [0.9,1.1]
            aug_color: Max relative scale factor for image saturation/hue sampling, e.g. 0.1 -> [0.9,1.1]
            image_height: RGB images are rescaled to this maximum height (if augmentation is disabled, and in the range
                [aug_scale_min * image_height, aug_scale_max * image_height] otherwise).
            use_half: Enabled if training with half-precision floats.
            num_clusters: split the input frames into disjoint clusters using hierarchical clustering in order to train
                an ensemble model. Clustering is deterministic, so multiple training calls with the same number of
                target clusters will result in the same split. See the paper for details of the approach. Disabled by
                default.
            cluster_idx: If num_clusters is not None, then use this parameter to choose the cluster used for training.
            split: choose between ["map"], ["query"], or ["map", "query"] dataset.
                "map" scenes is in the train/ folder of the scene and "query" scenes is the test/ folder
            trainskip: Uniformly subsample training data with 1/trainskip factor.
                It means we load data with every trainskip frames
            load_sc_map: loading precomputed scene coordinate maps of the scene
            load_scheme2_sc_map: if True, load sc map and GT subtracted with mean. if False, load original sc map and GT poses
            load_scheme2_sc_map: if True, load additional augmented sc map and GT subtracted with mean.
            marepo_sc_augment: if True, we apply additional data augmentation when training Marepo.
                                  Such as Gaussian noise to SC map, random shifts to SC maps.
            load_mapping_buffer_features: if True we load pre-stored mapping buffer features (4800,512) for each scene
            random_select_mapping_buffer: if True, we randomly load 1 of 10 pre-stored mapping buffer features.
                                          if False, we only load the 0th mapping buffer features
            all_mapping_buffers: if True, we load all 10 mapping buffer features (48000,512) for each scene.
            center_crop: a flag indicate that image needs to be center cropped so that the image size is proportional to mapfree training data
        """

        self.use_half = use_half

        self.init = (mode == 1)
        self.sparse = sparse
        self.eye = (mode == 2)

        self.image_height = image_height

        self.preprocess_augment = augment
        self.marepo_sc_augment = marepo_sc_augment
        if self.marepo_sc_augment:
            self.jitter_trans = jitter_trans
            self.jitter_rot = jitter_rot
            print(f"random jitter trans: {jitter_trans:.2f}m, random jitter rot: {jitter_rot:.1f}deg")

        self.aug_rotation = aug_rotation
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max
        self.aug_black_white = aug_black_white
        self.aug_color = aug_color

        self.num_clusters = num_clusters
        self.cluster_idx = cluster_idx

        self.split = split # ["map"], ["query"], ["map", "query"] assume we only parse the following arguments
        self.trainskip = trainskip
        self.load_sc_map = load_sc_map
        self.load_scheme2_sc_map=load_scheme2_sc_map
        self.load_scheme3_sc_map=load_scheme3_sc_map
        self.load_mapping_buffer_features=load_mapping_buffer_features
        self.random_select_mapping_buffer=random_select_mapping_buffer
        self.all_mapping_buffers=all_mapping_buffers
        self.non_mapfree_dataset_naming= non_mapfree_dataset_naming

        self.center_crop=center_crop

        assert((self.random_select_mapping_buffer and self.all_mapping_buffers) == False) # can't both to be true

        if self.num_clusters is not None:
            if self.num_clusters < 1:
                raise ValueError("num_clusters must be at least 1")

            if self.cluster_idx is None:
                raise ValueError("cluster_idx needs to be specified when num_clusters is set")

            if self.cluster_idx < 0 or self.cluster_idx >= self.num_clusters:
                raise ValueError(f"cluster_idx needs to be between 0 and {self.num_clusters - 1}")

        if self.eye and self.preprocess_augment and (self.aug_rotation > 0 or self.aug_scale_min != 1 or self.aug_scale_max != 1):
            # pre-generated eye coordinates cannot be augmented
            _logger.warning("WARNING: Check your augmentation settings. Camera coordinates will not be augmented.")

        # Setup data paths.
        root_dir = Path(root_dir)
        scene_dir = sorted(root_dir.iterdir())
        # scene_dir = sorted(os.listdir(root_dir))

        self.rgb_files = []
        self.pose_files = []
        self.calibration_files = []
        self.scene_coord_files = []
        self.support_scene_files=[]

        if self.load_scheme3_sc_map:
            self.aug_group=8 if USE_AUG_16==False else 16 # currently assume it is 8 groups of augmented data
        else:
            self.aug_group=0 # no preprocessed augmented data

        # Main folders.
        for data_split in self.split:
            if data_split == "map":
                print("loading mapping files...")
                split_folder = "train"
            elif data_split == "query":
                print("loading query files...")
                split_folder = "test"
            else:
                print("please check input split for dataset, should be ['map'] or ['query'] or ['map', 'query'] exit...")
                sys.exit()

            for scene in scene_dir:
                scene = scene / split_folder # ex: ~/storage/map_free_training_scenes/train/mapfree_s00225/train
                rgb_dir = scene / 'rgb'
                pose_dir = scene / 'poses'
                calibration_dir = scene / 'calibration'

                # Optional folders. Only used in marepo
                if self.load_sc_map:  # use sub mean data without augmentation
                    if self.load_scheme2_sc_map:
                        pose_dir = scene / 'poses_sub_mean'
                        sc_dir = scene / 'scene_coord_sub_mean'
                        self.scene_coord_files.extend(sorted(sc_dir.iterdir()))
                    else:  # use original data without augmentation
                        sc_dir = scene / 'scene_coord'
                        self.scene_coord_files.extend(sorted(sc_dir.iterdir()))
                else:
                    self.scene_coord_files = None

                # Find all images. The assumption is that it only contains image files.
                sorted_rgb_files = sorted(rgb_dir.iterdir())
                num_rgb_files = len(sorted_rgb_files)
                self.rgb_files.extend(sorted_rgb_files) # ex: /home/shuaichen_nianticlabs_com/storage/map_free_training_scenes/train/mapfree_s00000/train/rgb/frame_00569.jpg

                # Find all ground truth pose files. One per image.
                self.pose_files.extend(sorted(pose_dir.iterdir())) # ex: /home/shuaichen_nianticlabs_com/storage/map_free_training_scenes/train/mapfree_s00000/train/poses/pose_00577.txt

                # Load camera calibrations. One focal length per image.
                self.calibration_files.extend(sorted(calibration_dir.iterdir()))

        if self.trainskip > 1:
            self.rgb_files = self.rgb_files[::self.trainskip]
            self.pose_files = self.pose_files[::self.trainskip]
            self.calibration_files = self.calibration_files[::self.trainskip]
            if self.scene_coord_files:
                self.scene_coord_files = self.scene_coord_files[::self.trainskip]

        if len(self.rgb_files) != len(self.pose_files):
            raise RuntimeError('RGB file count does not match pose file count!')

        if len(self.rgb_files) != len(self.calibration_files):
            raise RuntimeError('RGB file count does not match calibration file count!')

        if self.scene_coord_files and len(self.rgb_files) != len(self.scene_coord_files):
            raise RuntimeError('RGB file count does not match scene coordinate file count!')

        print("total number of files found:", len(self.rgb_files))

        # Image transformations. Excluding scale since that can vary batch-by-batch.
        if self.preprocess_augment:
            if self.center_crop:
                self.image_transform = transforms.Compose([
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4],  # statistics calculated over 7scenes training set, should generalize fairly well
                        std=[0.25]
                    ),
                    transforms.CenterCrop((480,640)) # mapfree is using 480x640
                ])
            else:
                self.image_transform = transforms.Compose([
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4],  # statistics calculated over 7scenes training set, should generalize fairly well
                        std=[0.25]
                    ),
                ])
        else:
            if self.center_crop:
                self.image_transform = transforms.Compose([
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4],  # statistics calculated over 7scenes training set, should generalize fairly well
                        std=[0.25]
                    ),
                    transforms.CenterCrop((480, 640))  # mapfree is using 480x640
                ])
            else:
                self.image_transform = transforms.Compose([
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4],  # statistics calculated over 7scenes training set, should generalize fairly well
                        std=[0.25]
                    ),
                ])

        # We use this to iterate over all frames. If clustering is enabled this is used to filter them.
        self.valid_file_indices = np.arange(len(self.rgb_files))

        # If clustering is enabled.
        if self.num_clusters is not None:
            _logger.info(f"Clustering the {len(self.rgb_files)} into {num_clusters} clusters.")
            _, _, cluster_labels = self._cluster(num_clusters)

            self.valid_file_indices = np.flatnonzero(cluster_labels == cluster_idx)
            _logger.info(f"After clustering, chosen cluster: {cluster_idx}, Using {len(self.valid_file_indices)} images.")

        # if self.load_scheme2_sc_map ==False:
        #     # Calculate mean camera center (using the valid frames only). This is the most time consuming part in the dataloader
        #     self.mean_cam_center = self._compute_mean_camera_center()
        # else:
        #     self.mean_cam_center = torch.zeros((3,)) # for scheme2, we don't need this since mean is stored in ACE heads

        self.mean_cam_center = torch.zeros((3,))  # for scheme2, we don't need this since mean is stored in ACE heads

        if self.use_half and torch.cuda.is_available():
            self.mean_cam_center = self.mean_cam_center.half()

        # we store original loaded image size of the dataset
        if self.center_crop:
            self.default_img_H=480
            self.default_img_W=640
        else:
            image_0 = self._load_image(0)
            image_0 = self._resize_image(image_0, image_height)
            self.default_img_H = image_height
            self.default_img_W = image_0.size[0]

        # tmp = self.__getitem__([208,209,210])
        # breakpoint()
        # for i in range(120):
        #     tmp = self.__getitem__([i])
        # breakpoint()
        # tmp = self._get_single_item(0, self.image_height)
        # tmp = self.__getitem__([105,108])
        # tmp = self._get_aug_single_item(105, self.image_height)
        # breakpoint()

    @staticmethod
    def _resize_image(image, image_height):
        # Resize a numpy image as PIL. Works slightly better than resizing the tensor using torch's internal function.
        image = TF.to_pil_image(image)
        image = TF.resize(image, image_height, antialias=True)
        return image

    @staticmethod
    def _rotate_image(image, angle, order, mode='constant'):
        # Image is a torch tensor (CxHxW), convert it to numpy as HxWxC.
        image = image.permute(1, 2, 0).numpy()
        # Apply rotation.
        image = rotate(image, angle, order=order, mode=mode)
        # Back to torch tensor.
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image

    def _random_rotate_image(self, image, image_mask, pose):
        # Generate a random rotation angle.
        angle = random.uniform(-self.aug_rotation, self.aug_rotation)  # +/- 15 deg

        # Rotate input image and mask.
        image = self._rotate_image(image, angle, 1, 'reflect')
        image_mask = self._rotate_image(image_mask, angle, order=1, mode='constant')

        angle = angle * math.pi / 180.
        # Create a rotation matrix.
        pose_rot = torch.eye(4)
        pose_rot[0, 0] = math.cos(angle)
        pose_rot[0, 1] = -math.sin(angle)
        pose_rot[1, 0] = math.sin(angle)
        pose_rot[1, 1] = math.cos(angle)

        # Apply rotation matrix to the ground truth camera pose.
        pose = torch.matmul(pose, pose_rot)  # why multiplication on the right? https://zhuanlan.zhihu.com/p/128155013
        return image, image_mask, pose

    def _cluster(self, num_clusters):
        """
        Clusters the dataset using hierarchical kMeans.
        Initialization:
            Put all images in one cluster.
        Interate:
            Pick largest cluster.
            Split with kMeans and k=2.
            Input for kMeans is the 3D median scene coordiante per image.
        Terminate:
            When number of target clusters has been reached.
        Returns:
            cam_centers: For each cluster the mean (not median) scene coordinate
            labels: For each image the cluster ID
        """
        num_images = len(self.pose_files)
        _logger.info(f'Clustering a dataset with {num_images} frames into {num_clusters} clusters.')

        # A tensor holding all camera centers used for clustering.
        cam_centers = np.zeros((num_images, 3), dtype=np.float32)
        for i in range(num_images):
            pose = self._load_pose(i)
            cam_centers[i] = pose[:3, 3]

        # Setup kMEans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        flags = cv2.KMEANS_PP_CENTERS

        # Label of next cluster.
        label_counter = 0

        # Initialise list of clusters with all images.
        clusters = []
        clusters.append((cam_centers, label_counter, np.zeros(3)))

        # All images belong to cluster 0.
        labels = np.zeros(num_images)

        # iterate kMeans with k=2
        while len(clusters) < num_clusters:
            # Select largest cluster (list is sorted).
            cur_cluster = clusters.pop(0)
            label_counter += 1

            # Split cluster.
            cur_error, cur_labels, cur_centroids = cv2.kmeans(cur_cluster[0], 2, None, criteria, 10, flags)

            # Update cluster list.
            cur_mask = (cur_labels == 0)[:, 0]
            cur_cam_centers0 = cur_cluster[0][cur_mask, :]
            clusters.append((cur_cam_centers0, cur_cluster[1], cur_centroids[0]))

            cur_mask = (cur_labels == 1)[:, 0]
            cur_cam_centers1 = cur_cluster[0][cur_mask, :]
            clusters.append((cur_cam_centers1, label_counter, cur_centroids[1]))

            cluster_labels = labels[labels == cur_cluster[1]]
            cluster_labels[cur_mask] = label_counter
            labels[labels == cur_cluster[1]] = cluster_labels

            # Sort updated list.
            clusters = sorted(clusters, key=lambda cluster: cluster[0].shape[0], reverse=True)

        # clusters are sorted but cluster indices are random, remap cluster indices to sorted indices
        remapped_labels = np.zeros(num_images)
        remapped_clusters = []

        for cluster_idx_new, cluster in enumerate(clusters):
            cluster_idx_old = cluster[1]
            remapped_labels[labels == cluster_idx_old] = cluster_idx_new
            remapped_clusters.append((cluster[0], cluster_idx_new, cluster[2]))

        labels = remapped_labels
        clusters = remapped_clusters

        cluster_centers = np.zeros((num_clusters, 3))
        cluster_sizes = np.zeros((num_clusters, 1))

        for cluster in clusters:
            # Compute distance of each cam to the center of the cluster.
            cam_num = cluster[0].shape[0]
            cam_data = np.zeros((cam_num, 3))
            cam_count = 0

            # First compute the center of the cluster (mean).
            for i, cam_center in enumerate(cam_centers):
                if labels[i] == cluster[1]:
                    cam_data[cam_count] = cam_center
                    cam_count += 1

            cluster_centers[cluster[1]] = cam_data.mean(0)

            # Compute the distance of each cam from the cluster center. Then average and square.
            cam_dists = np.broadcast_to(cluster_centers[cluster[1]][np.newaxis, :], (cam_num, 3))
            cam_dists = cam_data - cam_dists
            cam_dists = np.linalg.norm(cam_dists, axis=1)
            cam_dists = cam_dists ** 2

            cluster_sizes[cluster[1]] = cam_dists.mean()

            _logger.info("Cluster %i: %.1fm, %.1fm, %.1fm, images: %i, mean squared dist: %f" % (
                cluster[1], cluster_centers[cluster[1]][0], cluster_centers[cluster[1]][1], cluster_centers[cluster[1]][2],
                cluster[0].shape[0], cluster_sizes[cluster[1]]))

        _logger.info('Clustering done.')

        return cluster_centers, cluster_sizes, labels

    def _compute_mean_camera_center(self):
        mean_cam_center = torch.zeros((3,))
        for idx in self.valid_file_indices:
            pose = self._load_pose(idx)

            # Get the translation component.
            mean_cam_center += pose[0:3, 3]

        # Avg.
        mean_cam_center /= len(self)
        return mean_cam_center


    def _load_image(self, idx):
        # print("idx", idx)
        # print("self.rgb_files[idx]", self.rgb_files[idx])
        # breakpoint()
        
        try:
            image = imageio.imread(self.rgb_files[idx])
        except:
            print("image truncated bug? check if image is corrupted")
            breakpoint()


        # image = io.imread(self.rgb_files[idx])

        if len(image.shape) < 3:
            # Convert to RGB if needed.
            image = color.gray2rgb(image)

        return image

    def _parse_aug_meta(self, rgb_files):
        '''
        parsing augmented rgb meta
        rgb_files: given an rgb filename
        return root scene dir, frame number
        '''
        rgb_stem = rgb_files.stem
        rgb_root = str(rgb_files.parent.parent)
        frame_number = rgb_stem[-5:]
        return rgb_root, rgb_stem, frame_number

    def _load_aug_image(self, idx, gnum):
        """
        load augmented image
        """

        rgb_root, rgb_stem, frame_number = self._parse_aug_meta(self.rgb_files[idx])
        rgb_file = rgb_root + f"/aug/rgb_{gnum:03d}/" + rgb_stem + ".jpg"
        rgb_file = Path(rgb_file)
        image = io.imread(rgb_file)

        if len(image.shape) < 3:
            # Convert to RGB if needed.
            image = color.gray2rgb(image)

        return image, rgb_file

    def _load_aug_intrinsic(self, idx, gnum):
        """
        load augmented intrinsic
        return (3,3)
        """

        if self.non_mapfree_dataset_naming:
            rgb_root, rgb_stem, frame_number = self._parse_aug_meta(self.rgb_files[idx])
            intrinsic_file = rgb_root + f"/aug/intrinsics_{gnum:03d}/" + "intrinsic_" + rgb_stem + ".txt"
            # load intrinsic file
            intrinsic = np.loadtxt(intrinsic_file)
        else: # handle bad suffix naming
            rgb_root, rgb_stem, frame_number = self._parse_aug_meta(self.rgb_files[idx])
            # try to load intrinsic file with bad suffix, otherwise load the file with correct suffix
            try:
                intrinsic_file = rgb_root + f"/aug/intrinsics_{gnum:03d}/" + "intrinsic_" + frame_number + ".txt.npy"  # This is to handle bad suffix
                # load intrinsic file
                intrinsic = np.load(intrinsic_file)
            except:
                intrinsic_file = rgb_root + f"/aug/intrinsics_{gnum:03d}/" + "intrinsic_" + frame_number + ".txt"
                # load intrinsic file
                intrinsic = np.loadtxt(intrinsic_file)
        return intrinsic

    def _load_aug_mask(self, idx, gnum, image_height, image_width):
        """
        load augmented mask. Here, we use upsampled sc mask to mimic image mask.

        This should be fixed later
        return (3,3)
        """

        rgb_root, rgb_stem, frame_number = self._parse_aug_meta(self.rgb_files[idx])
        sc_mask = rgb_root + f"/aug/sc_mask_{gnum:03d}/" + rgb_stem + ".npy"
        # load sc mask file
        sc_mask = np.load(sc_mask)
        sc_mask = torch.tensor(sc_mask)

        # upsample sc_mask to image_mask
        image_mask = TF.resize(sc_mask, (image_height, image_width), antialias=True)

        return image_mask[0], sc_mask[0]

    def _load_pose(self, idx):
        # Stored as a 4x4 matrix.
        pose = np.loadtxt(self.pose_files[idx])
        pose = torch.from_numpy(pose).float()

        return pose

    def _load_aug_pose(self, idx, gnum):
        if self.non_mapfree_dataset_naming:
            rgb_root, rgb_stem, frame_number = self._parse_aug_meta(self.rgb_files[idx])
            pose_file = rgb_root + f"/aug/poses_{gnum:03d}/" + "pose_" + rgb_stem + ".txt"
        else:
            rgb_root, rgb_stem, frame_number = self._parse_aug_meta(self.rgb_files[idx])
            pose_file = rgb_root + f"/aug/poses_{gnum:03d}/" + "pose_" + frame_number + ".txt"

        # Stored as a 4x4 matrix.
        pose = np.loadtxt(pose_file)
        pose = torch.from_numpy(pose).float()

        return pose

    def _load_aug_scene_coords(self, idx, gnum):
        """
        load augmented intrinsic
        return (3,3)
        """
        rgb_root, rgb_stem, frame_number = self._parse_aug_meta(self.rgb_files[idx])
        sc_file = rgb_root + f"/aug/sc_{gnum:03d}/" + rgb_stem + ".npy"
        # load scene coordinates file
        sc = np.load(sc_file)
        return sc

    def _random_jitter_global_coordinate_method(self, sc, sc_mask, pose):
        """
        Apply random rotation and random shift to global coordinate. This is method 2 implementation (slide 58)
        camera cooridnate (e) = H(pose).inv() @ y (scene coordinate)
        assume random transformation T =>
        H' = H @ T
        y' = H @ T @ H.inv() @ y
        sc: [N,C,H,W]
        sc_mask: [N,1,H,W], or [1,H,W]
        pose:[4,4]
        """

        random_rot = np.random.uniform(-self.jitter_rot, self.jitter_rot, 3)
        r_matrix = R.from_euler('zyx', random_rot, degrees=True).as_matrix()

        T = np.identity(4) # 4x4
        T[:3,:3] = r_matrix

        # add random translation
        t_shift = np.random.uniform(-self.jitter_trans, self.jitter_trans, 3) # +/- 1m

        T[:3,3] = t_shift

        T = torch.Tensor(T)

        # y' = T @ y
        # raise sc to homogeneous coordinate and apply transformation
        N, C, H, W = sc.shape
        sc_homo = sc.permute(0, 2, 3, 1).reshape(N * H * W, C)  # [4800, 3]
        ones = torch.ones((N * H * W, 1))  # [4800,1]
        sc_homo = torch.cat([sc_homo, ones], dim=1)  # [4800,4]
        sc_homo_new = (T @ sc_homo.T).T
        sc_new = sc_homo_new[:, :3].reshape(N, H, W, C).permute(0, 3, 1, 2)

        try:
            if sc_mask != None:
                sc_new = sc_new * sc_mask  # filter invalid pixel on sc map
        except:
            # create new sc_mask that is same size as sc_new
            n,c,h,w=sc_new.shape
            sc_mask_new = torch.zeros((n,h,w)).bool()
            nn,hh,ww = sc_mask.shape
            sc_mask_new[:nn,:hh,:ww] = sc_mask
            sc_new = sc_new * sc_mask_new
            # print(sc_new.shape, sc_mask.shape)
            # breakpoint()

        # H' = T @ H <- this is essentially (T@H).inv()
        pose_new = T @ pose
        return sc_new, pose_new

    def _gaussian_noise_to_sc(self, sc, sc_mask):
        '''
        Apply gaussian noise to scene coordinate map (only on valid pixels)
        '''
        mean = 0
        std = 0.05 # 10cm std
        noise = torch.Tensor(np.random.normal(mean, std, (1,3,60,80)))
        sc = sc+noise
        sc = sc*sc_mask # filter invalid pixel on sc map
        return sc

    def _spike_noise_to_sc(self, sc, sc_mask):
        '''
        Apply spike noise to scene coordinate map, randomly in patch of 5x5
        sc: [N,C,H,W]
        sc_mask: [N,1,H,W] or [1,H,W]
        return sc_new: [N,C,H,W]
        '''
        B,C,H,W = sc.shape
        patch_width = 5 # assuming 5x5 patches
        perturb_translation_range = 1 # perturb +/- 10m
        percent_pixel_samples = 0.05

        # sample roughly 10% of patches based on total pixel count
        sample_nums = int(percent_pixel_samples*H*W//(patch_width**2))
        sample_idx_h = np.random.randint(0, H-patch_width-1, sample_nums) # upper left corner idx vertically
        sample_idx_w = np.random.randint(0, W-patch_width-1, sample_nums) # upper left corner idx horizontally

        noise_map = np.full((B,C,H,W), 0, dtype=np.float32)

        for sn_idx in range(sample_nums):
            # get upper left corner position index of the random patch
            hi = sample_idx_h[sn_idx]
            wi = sample_idx_w[sn_idx]

            # for each patch, we assign 3D noise
            noise_map[:,:,hi:hi+patch_width,wi:wi+patch_width] = \
                np.random.uniform(-perturb_translation_range,perturb_translation_range,C)[None,:,None,None]

        noise_map = torch.Tensor(noise_map)

        # apply noise on sc map
        sc_new = sc+noise_map
        if sc_mask != None:
            sc_new = sc_new * sc_mask  # filter invalid pixel on sc map
        return sc_new

    def _parse_n_load_mapping_buffer_feature(self, rgb_file):
        """
        Parse mapping buffer feature location of the scene,
        and load the pre-processed features.
        rgb_file: (there are two types of rgb_file path)
                ex1: '/home/shuaichen_nianticlabs_com/storage/map_free_training_scenes/train/mapfree_s00018/test/rgb/frame_00120.jpg'
                ex2: '/home/shuaichen_nianticlabs_com/storage/map_free_training_scenes/train/mapfree_s00018/test/aug/rgb_005/frame_00120.jpg'
        """

        osp_split_left=rgb_file
        osp_split_right=None

        # parse scene folder to sth like this: /home/shuaichen_nianticlabs_com/storage/map_free_training_scenes/train/mapfree_s00018/
        while osp_split_right!='test': # here 'test' means the query folder
            osp_split_left, osp_split_right = osp.split(osp_split_left)

        # get mapping buffer feature filenames
        if self.random_select_mapping_buffer: # randomly select from 10 pre-stored buffers
            rand_idx = random.randint(0, 9)
            map_buffer_file_path = osp.join(osp_split_left, 'train', 'mapping_buffer', f'mapping_feature_{rand_idx:02d}.npy')
            map_buffer_feature = np.load(map_buffer_file_path)  # [4800,512]
        elif self.all_mapping_buffers:
            map_buffer_list = []
            for i in range(10):
                map_buffer_file_path = osp.join(osp_split_left, 'train', 'mapping_buffer', f'mapping_feature_{i:02d}.npy')
                map_buffer_feature = np.load(map_buffer_file_path)
                map_buffer_list.append(map_buffer_feature)
            map_buffer_feature = np.array(map_buffer_list)
            map_buffer_feature = map_buffer_feature.reshape((-1,512))
        else: # only select 00th buffer
            map_buffer_file_path = osp.join(osp_split_left, 'train', 'mapping_buffer', 'mapping_feature_00.npy')
            map_buffer_feature = np.load(map_buffer_file_path) # [4800,512]
        map_buffer_feature = torch.tensor(map_buffer_feature)
        return map_buffer_feature
    def _get_single_item(self, idx, image_height):
        '''
        Currently used for preprocessing
        '''
        # Apply index indirection.
        idx = self.valid_file_indices[idx]

        # Load image.
        image = self._load_image(idx)
        # print("image size1", image.shape)
        # print("image_height", image_height)

        # Load intrinsics.
        k = np.loadtxt(self.calibration_files[idx])

        if k.size == 1:
            focal_length = float(k)
            centre_point = None
        elif k.shape == (3, 3):
            k = k.tolist()
            focal_length = [k[0][0], k[1][1]]
            centre_point = [k[0][2], k[1][2]]
        else:
            raise Exception("Calibration file must contain either a 3x3 camera \
                        intrinsics matrix or a single float giving the focal length \
                        of the camera.")

        # The image will be scaled to image_height, adjust focal length as well.
        f_scale_factor = image_height / image.shape[0]  # 0.8888888888 default

        if centre_point:
            centre_point = [c * f_scale_factor for c in centre_point]
            focal_length = [f * f_scale_factor for f in focal_length]
        else:
            focal_length *= f_scale_factor

        if self.center_crop:
            image = TF.to_pil_image(image)
            # Apply remaining transforms. including the possible center crop
            image = self.image_transform(image)
            # print("image size3", image.shape)

            # Rescale image.
            image = TF.resize(image, image_height, antialias=True)

        else:
            # old implementation
            # Rescale image.
            image = self._resize_image(image, image_height)
            # print("image size2", image.size)

            # Apply remaining transforms. including the possible center crop
            image = self.image_transform(image)
            # print("image size3", image.shape)

        # Create mask of the same size as the resized image.
        image_mask = torch.ones((1, image.shape[1], image.shape[2]))

        # Load pose.
        pose = self._load_pose(idx)

        # Load preproceed scene coordinates, if needed.
        if self.scene_coord_files:
            scene_coords = np.load(self.scene_coord_files[idx])
            scene_coords = torch.Tensor(scene_coords)
        else:
            scene_coords = 0  # we don't need it at preprocessing stage

        # Apply data augmentation if necessary.
        if self.preprocess_augment:
            image, image_mask, pose = self._random_rotate_image(image, image_mask, pose)

        # Convert to half if needed.
        if self.use_half and torch.cuda.is_available():
            image = image.half()

        # Binarize the mask.
        image_mask = image_mask > 0

        # Invert the pose.
        pose_inv = pose.inverse()

        # Create the intrinsics matrix.
        intrinsics = torch.eye(3)

        # Hardcode the principal point to the centre of the image unless otherwise specified.
        if centre_point:
            intrinsics[0, 0] = focal_length[0]
            intrinsics[1, 1] = focal_length[1]
            intrinsics[0, 2] = centre_point[0]
            intrinsics[1, 2] = centre_point[1]
        else:
            intrinsics[0, 0] = focal_length
            intrinsics[1, 1] = focal_length
            intrinsics[0, 2] = image.shape[2] / 2  # shouldn't this be (image.shape[2]-1)/2 ?
            intrinsics[1, 2] = image.shape[1] / 2

        # Also need the inverse.
        intrinsics_inv = intrinsics.inverse()
        # print("image4 size1", image.shape)
        # print("intrinsics", intrinsics)

        out = {}
        out['image'] = image
        out['image_mask'] = image_mask
        out['pose'] = pose
        out['pose_inv'] = pose_inv
        out['intrinsics'] = intrinsics
        out['intrinsics_inv'] = intrinsics_inv
        out['scene_coords'] = scene_coords
        out['rgb_files'] = str(self.rgb_files[idx])
        if self.load_mapping_buffer_features:
            map_buffer_feature = self._parse_n_load_mapping_buffer_feature(str(self.rgb_files[idx]))
            out['map_buffer_feature'] = map_buffer_feature
        return out

    def _get_aug_single_item(self, idx, image_height):
        '''
        # currently this function used for marepo Training
        get augmented single item, this method uses preprocessed augmented data to support scheme3 training
        '''

        # select random group of augmented data, -1 being original data w/ augmented data, >=0 being preprocessed augmented data
        assert(image_height==480) # for safety

        gnum = random.randint(-1, self.aug_group-1)
        # print("gnum:", gnum)

        # Apply index indirection.
        idx = self.valid_file_indices[idx]

        # Load image.
        if gnum==-1:
            image = self._load_image(idx)
            if self.center_crop:
                image = TF.to_pil_image(image)
                # Apply remaining transforms. including the possible center crop
                image = self.image_transform(image)
                # print("image size3", image.shape)

            rgb_file = self.rgb_files[idx]
        else:
            image, rgb_file = self._load_aug_image(idx, gnum)
        # print("image size1", image.shape)
        # print("image_height", image_height)
        # breakpoint()

        if gnum==-1:
            k = np.loadtxt(self.calibration_files[idx])
        else:
            # Load preprocessed intrinsics.
            k = self._load_aug_intrinsic(idx, gnum)

        if k.size == 1:
            focal_length = float(k)
            centre_point = None
        elif k.shape == (3, 3):
            k = k.tolist()
            focal_length = [k[0][0], k[1][1]]
            centre_point = [k[0][2], k[1][2]]
        else:
            raise Exception("Calibration file must contain either a 3x3 camera \
                intrinsics matrix or a single float giving the focal length \
                of the camera.")

        # The image will be scaled to image_height, adjust focal length as well.
        f_scale_factor = image_height / image.shape[0] # 0.8888888888 default

        if centre_point:
            centre_point = [c * f_scale_factor for c in centre_point]
            focal_length = [f * f_scale_factor for f in focal_length]
        else:
            focal_length *= f_scale_factor

        # Rescale image.
        image = self._resize_image(image, image_height)
        # print("image size2", image.size)

        # Create mask of the same size as the resized image (it's a PIL image at this point).
        if gnum==-1:
            image_mask = torch.ones((1, image.size[1], image.size[0]))
            # sc_mask = torch.ones((1, round(image.size[1]/8), round(image.size[0]/8))) # assuming rounding here
            sc_mask = torch.ones((1, math.ceil(image.size[1] / 8), math.ceil(image.size[0] / 8)))
            sc_mask = sc_mask > 0
        else: # load preprocessed image_mask
            image_mask, sc_mask = self._load_aug_mask(idx, gnum, image.size[1], image.size[0]) # [1,480,640]

        # Apply remaining transforms.
        image = self.image_transform(image)
        # print("image size3", image.shape)

        if gnum == -1:
            pose = self._load_pose(idx)
        else: # load preprocessed pose
            pose = self._load_aug_pose(idx, gnum)

        if self.scene_coord_files:
            if gnum == -1:
                scene_coords = np.load(self.scene_coord_files[idx])
            else:
                scene_coords = self._load_aug_scene_coords(idx, gnum)
            scene_coords = torch.Tensor(scene_coords)
        else:
            scene_coords = None # we don't need it at preprocessing stage

        # Apply data augmentation for preprocess if necessary.
        if self.preprocess_augment:
            image, image_mask, pose = self._random_rotate_image(image, image_mask, pose)

        # Apply sc augmentation at marepo training
        if self.marepo_sc_augment:
            assert(scene_coords!=None)
            scene_coords, pose = self._random_jitter_global_coordinate_method(scene_coords, sc_mask, pose)

            if SPIKE_NOISE:
                scene_coords = self._spike_noise_to_sc(scene_coords, sc_mask)

        # Convert to half if needed.
        if self.use_half and torch.cuda.is_available():
            image = image.half()
            scene_coords = scene_coords.half()

        # Binarize the mask.
        image_mask = image_mask > 0

        # Invert the pose.
        pose_inv = pose.inverse()

        # Create the intrinsics matrix.
        intrinsics = torch.eye(3)

        # Hardcode the principal point to the centre of the image unless otherwise specified.
        if centre_point:
            intrinsics[0, 0] = focal_length[0]
            intrinsics[1, 1] = focal_length[1]
            intrinsics[0, 2] = centre_point[0]
            intrinsics[1, 2] = centre_point[1]
        else:
            intrinsics[0, 0] = focal_length
            intrinsics[1, 1] = focal_length
            intrinsics[0, 2] = image.shape[2] / 2
            intrinsics[1, 2] = image.shape[1] / 2

        # Also need the inverse.
        intrinsics_inv = intrinsics.inverse()
        # print("image4 size1", image.shape)

        out = {}
        out['image'] = image
        out['image_mask'] = image_mask
        out['pose'] = pose
        out['pose_inv'] = pose_inv
        out['intrinsics'] = intrinsics
        out['intrinsics_inv'] = intrinsics_inv
        out['scene_coords'] = scene_coords
        out['rgb_files'] = str(rgb_file)
        out['sc_mask'] = sc_mask
        if self.load_mapping_buffer_features:
            map_buffer_feature = self._parse_n_load_mapping_buffer_feature(str(rgb_file))
            out['map_buffer_feature'] = map_buffer_feature
        return out


    def __len__(self):
        return len(self.valid_file_indices)

    def __getitem__(self, idx):
        if self.preprocess_augment:
            scale_factor = random.uniform(self.aug_scale_min, self.aug_scale_max) # [0.6666, 1.5]
        else:
            scale_factor = 1

        # Target image height. We compute it here in case we are asked for a full batch of tensors because we need
        # to apply the same scale factor to all of them.
        image_height = int(self.image_height * scale_factor)

        if self.load_scheme3_sc_map: # randomly load additional preprcessed augmented data (scheme3)
            if type(idx) == list:
                # Whole batch.
                tensors = [self._get_aug_single_item(i, image_height) for i in idx]
                return default_collate(tensors)
            else:
                # Single element.
                return self._get_aug_single_item(idx, image_height)
        else:
            if type(idx) == list:
                # Whole batch.
                tensors = [self._get_single_item(i, image_height) for i in idx]
                return default_collate(tensors)
            else:
                # Single element.
                return self._get_single_item(idx, image_height)
