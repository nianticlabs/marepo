import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
class Custom_RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
        aug_size (tuple or int): Image size of augmented data
        intrinsics: [3,3]: camera intrinsics
    """

    def __init__(self, output_size, aug_size, intrinsics):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

        self.aug_size = aug_size
        self.intrinsics = intrinsics

        self.generate_random_top_left_loc()

        # update new principle point in camera intrinsic matrix
        self.update_principle_point()

    def update_new_focal(self):
        old_focal = (self.intrinsics[0,0], self.intrinsics[1,1])
        f_scale_factor = self.output_size[0] / self.aug_size[0]
        new_focal = (f_scale_factor * old_focal[0], f_scale_factor * old_focal[1])
        self.intrinsics[0,0] = new_focal[0]
        self.intrinsics[1,1] = new_focal[1]

    def generate_random_top_left_loc(self):
        if self.aug_size[0] > self.output_size[0]:
            assert (self.aug_size[1] > self.output_size[1])
            # crop target from input image
            h, w = self.aug_size[:2]
            self.new_h, self.new_w = self.output_size
            self.top = np.random.randint(0, h - self.new_h + 1)
            self.left = np.random.randint(0, w - self.new_w + 1)
            self.mode = "random_crop"
        elif self.aug_size[0] < self.output_size[0]:
            assert (self.aug_size[1] < self.output_size[1])
            # map input image to target with padding
            h, w = self.output_size
            self.new_h, self.new_w = self.aug_size[:2]
            self.top = np.random.randint(0, h - self.new_h + 1)
            self.left = np.random.randint(0, w - self.new_w + 1)
            self.mode = "random_padding"
        else:
            assert (self.aug_size[0] == self.output_size[0])
            assert (self.aug_size[1] == self.output_size[1])
            # probably do nothing since self.aug_size[0] == self.output_size[0]
            self.new_h, self.new_w = self.output_size
            self.top = 0
            self.left = 0
            self.mode = "none"

        # print("input aug_size: ", self.aug_size, "target size:", self.output_size, "top, left: ", self.top, self.left)

    def update_principle_point(self):
        # update self.intrinsics

        # print("intrinsic before:", self.intrinsics)
        if self.mode == "random_crop":
            self.intrinsics[0,2] = self.intrinsics[0,2] - self.left # horizontal
            self.intrinsics[1,2] = self.intrinsics[1,2] - self.top # vertical
        elif self.mode == "random_padding":
            self.intrinsics[0,2] = self.intrinsics[0,2] + self.left
            self.intrinsics[1, 2] = self.intrinsics[1, 2] + self.top
        # print("intrinsic after:", self.intrinsics)

    def crop(self, image, top, left, new_h, new_w):
        # crop target from input image
        return image[:,:,
                top:top + new_h,
                left:left + new_w]

    def pad(self, image, top, left, new_h, new_w, output_size_h, output_size_w):
        # map input image to target with padding
        N,C,H,W = image.shape
        new_image = torch.zeros((N,C,output_size_h,output_size_w), dtype=image.dtype, device=image.device)
        # print("image.shape", image.shape)
        # print("new_image.shape", new_image.shape)
        try:
            # TODO: new_h and new_w may not be the best option, due to self.top//df vs. self.new_h//df
            new_image[:,:,top:top+H,left:left+W] = image[:,:,:,:]
        except:
            print("image H,W:", H, W, "new_h, new_w:", new_h, new_w)
            breakpoint()
        return new_image

    def sc_map_crop(self, sc, df=8):
        '''
        This is a supplmentary crop function designed to perform same crop to scene coordinate maps,
        which the sc map typically smaller than the original image due to ACE network
        sc: torch.tensor [N,C,H,W] scene coordinate map
        df: float downsize factor, default is 8
        '''

        if self.mode == "random_crop":
            sc = self.crop(sc, self.top//df, self.left//df, self.new_h//df, self.new_w//df)
        elif self.mode == "random_padding":
            sc = self.pad(sc, self.top//df, self.left//df, self.new_h//df, self.new_w//df, self.output_size[0]//df, self.output_size[1]//df)
        else:
            # do nothing
            pass
        return sc

    def __call__(self, image):
        '''
        image: torch.Tensor [N,C,H,W]
        '''

        # crop or pad depends on mode
        if self.mode == "random_crop":
            image = self.crop(image, self.top, self.left, self.new_h, self.new_w)
        elif self.mode == "random_padding":
            image = self.pad(image, self.top, self.left, self.new_h, self.new_w, self.output_size[0], self.output_size[1])
        else:
            # do nothing
            return image
        return image

def fix_90deg_jitter_global_coordinate(sc_, sc_mask, pose_, y_deg=90):
    '''
    Apply fixed 90 deg rotation to global coordinate
    sc: [N,C,H,W]
    sc_mask: [N,1,H,W]
    psoe:[4,4]
    '''
    sc = sc_.clone()
    pose = pose_.clone()

    # print("inference: _fix_90deg_jitter_global_coordinate()")
    # fix 90 degrees exp. for verify jitter augmentation
    r_matrix = R.from_euler('y', y_deg, degrees=True).as_matrix() # assuming opencv coordiante (x (right), y (down), z (forward))
    T = np.identity(4)  # 4x4
    T[:3, :3] = r_matrix
    T = torch.Tensor(T)

    # apply transformation
    pose_new = pose @ T

    # raise sc to homogeneous coordinate and apply transformation
    N, C, H, W = sc.shape
    sc_homo = sc.permute(0, 2, 3, 1).reshape(N * H * W, C)  # [4800, 3]
    ones = torch.ones((N * H * W, 1))  # [4800,1]
    sc_homo = torch.cat([sc_homo, ones], dim=1)  # [4800,4]
    # y' = H @ T @ H.inv() @ y
    # we need to convert sc_homo [4800,4] -> [4,4800] -> sc_homo_new [4800,4]
    sc_homo_new = (pose_new @ torch.linalg.inv(pose) @ sc_homo.T).T
    sc_new = sc_homo_new[:, :3].reshape(N, H, W, C).permute(0, 3, 1, 2)

    if sc_mask != None:
        sc_new = sc_new * sc_mask  # filter invalid pixel on sc map
    return sc_new, pose_new

def random_jitter_global_coordinate(sc_, sc_mask, pose_):
    """
    Apply random rotation and random shift to global coordinate
    camera cooridnate (e) = H(pose).inv() @ y (scene coordinate)
    assume random transformation T =>
    H' = H @ T
    y' = H @ T @ H.inv() @ y
    sc: [N,C,H,W]
    sc_mask: [N,1,H,W]
    psoe:[4,4]
    """
    sc = sc_.clone()
    pose = pose_.clone()
    random_rot = R.random()
    r_matrix = random_rot.as_matrix()

    # print("random rot in deg:", random_rot.as_euler('xyz', degrees=True))
    T = np.identity(4) # 4x4
    T[:3,:3] = r_matrix

    # add random translation
    t_shift = np.random.uniform(-1, 1, 3) # +/- 1m
    # print("random t in m:",t_shift)

    # if JITTER_ROT_ONLY==False:
    T[:3,3] = t_shift

    T = torch.Tensor(T)

    # check T and pose are invertible matrices
    # print(torch.dist(pose @ torch.linalg.inv(pose), torch.eye(4)))
    # print(torch.dist(T @ torch.linalg.inv(T), torch.eye(4)))

    # apply transformation
    pose_new = pose @ T

    # raise sc to homogeneous coordinate and apply transformation
    N,C,H,W = sc.shape
    sc_homo = sc.permute(0,2,3,1).reshape(N*H*W, C) # [4800, 3]
    ones = torch.ones((N*H*W,1)) # [4800,1]
    sc_homo = torch.cat([sc_homo, ones], dim=1) # [4800,4]
    # y' = H @ T @ H.inv() @ y
    # we need to convert sc_homo [4800,4] -> [4,4800] -> sc_homo_new [4800,4]
    sc_homo_new = (pose_new @ torch.linalg.inv(pose) @ sc_homo.T).T
    sc_new = sc_homo_new[:,:3].reshape(N,H,W,C).permute(0,3,1,2)
    sc_new = sc_new * sc_mask  # filter invalid pixel on sc map
    return sc_new, pose_new

def add_uniform_noise(tensor, percentage=0.1, noise_level=0.1):
    '''
    tensor: [N,C,H,W]
    percentage: 0-1
    noise_level: in meter
    '''
    # Get the shape of the tensor
    B, C, H, W = tensor.shape

    # Calculate the number of pixels to be affected by noise
    num_pixels = int(percentage * H * W)

    # Generate random indices for pixels
    indices = torch.randperm(H * W)[:num_pixels]

    # Generate random noise values between +/- 0.1
    noise_values = torch.rand(num_pixels, device=tensor.device) * 2 * noise_level - noise_level

    # Reshape indices for 2D indexing
    row_indices = indices // W
    col_indices = indices % W

    # Add noise to the selected pixels
    tensor[:, :, row_indices, col_indices] += noise_values.view(1, 1, -1)

    return tensor