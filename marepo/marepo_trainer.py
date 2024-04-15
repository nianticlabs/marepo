# Copyright © Niantic, Inc. 2024.

import logging
import random
import os
import os.path as osp

import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import json
import sys
sys.path.append('../') # set system path to parent directory
from marepo.marepo_network import Regressor
from marepo.load_dataset_marepo import load_single_scene, load_multiple_scene
from marepo.metrics import compute_pose_error_new
from marepo.marepo_vis_util import plot_image_saliancy, plot_image_saliancy_with_blending, plot_image
from loss import PoseLossMapFree, PoseLossMapFree_C2F

# pytorch-lightning
from pytorch_lightning import LightningModule, Trainer

_logger = logging.getLogger(__name__)

MORE_THRESHOLDS=False #  if True: we further log the new threshold such as 0.25m/2deg, 0.5m/5deg, 5m/10deg, to tensorboard

def set_seed(seed):
    """
    Seed all sources of randomness.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class TrainerMarepoTransformer(LightningModule):
    '''
    Pytorch lightning trainer for Map-relative pos regression
    see https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#train-epoch-level-metrics for pseudocode of fit()
    '''
    def __init__(self, options):
        super().__init__()
        self.save_hyperparameters(options) # pl function, self.hparams

        # Setup randomness for reproducibility.
        self.base_seed = 2089
        set_seed(self.base_seed)

        # I use self.actual_epoch to keep track with trained epochs in case of resume training
        # This is due to I cannot reset the self.current_epoch variable
        self.actual_epoch=0

        # generate saving checkpoint path
        self.output_map_file = osp.join("../logs/", self.hparams.output_map, self.hparams.output_map + '.pt')

        # create save path if not exists
        save_dir = osp.split(self.output_map_file)[0]
        if not osp.isdir(save_dir):
            print("create save folder:", save_dir)
            os.makedirs(save_dir)

        self.setup_dataloader()

        # Create network using the state dict of the pretrained encoder.
        encoder_state_dict = torch.load(self.hparams.encoder_path, map_location="cpu")
        _logger.info(f"Loaded pretrained encoder from: {self.hparams.encoder_path}")
        head_state_dict = torch.load(self.hparams.head_network_path, map_location="cpu") # this model is not used in Marepo Training
        _logger.info(f"Loaded pretrained head weights from: {self.hparams.head_network_path}")

        # some configuration for the transformer
        f = open(self.hparams.transformer_json)
        config = json.load(f)
        f.close()
        config["transformer_pose_mean"] = self.train_dl.dataset.mean_cam_center
        _logger.info(f"Loaded transformer config from: {self.hparams.transformer_json}")
        self.default_img_H = self.train_dl.dataset.default_img_H # we get default image H and W for position encoding
        self.default_img_W = self.train_dl.dataset.default_img_W
        config["default_img_HW"] = [self.default_img_H, self.default_img_W]

        self.regressor = Regressor.create_from_split_state_dict(encoder_state_dict, head_state_dict, config)
        self.regressor = self.regressor.to(self.device)
        self.regressor.transformer_head.train() # for us, we only care about the transformer_head

        # create pose regression loss
        ''' section 3.3 of paper '''
        self.coarse_to_fine_prediction=False
        if 'c2f' in config:
            if config['c2f']=='V0' or config['c2f']=='V2':
                self.coarse_to_fine_prediction = True
                self.loss = PoseLossMapFree_C2F(soft_clamp=self.hparams.soft_clamping_l1_loss).to(self.device)
                self.num_predictions = len(config['layer_names'])//4
            else:
                self.loss = PoseLossMapFree(soft_clamp=self.hparams.soft_clamping_l1_loss).to(self.device)
        else:
            self.loss = PoseLossMapFree(soft_clamp=self.hparams.soft_clamping_l1_loss).to(self.device)

        log_dir, _ = osp.split(self.output_map_file)
        log_dir += '/runs'
        if self.global_rank == 0:
            self.writer = SummaryWriter(log_dir=log_dir)

        ''' Random rescale SC values, training for large scale scene i.e., Cambridge Landmarks '''
        self.random_rescale_sc=True if self.hparams.random_rescale_sc else False

    def setup_dataloader(self):

        # Create train/val/test dataset.
        if self.hparams.dataset_path == "":  # load single scene
            self.train_dl, self.val_dl, self.test_dl = load_single_scene(self.hparams)
        else:  # load all scenes
            self.train_dl, self.val_dl, self.test_dl = load_multiple_scene(self.hparams)

        _logger.info("Loaded training scan from: {} -- {} images, mean: {:.2f} {:.2f} {:.2f}".format(
            self.hparams.dataset_path,
            len(self.train_dl),
            self.train_dl.dataset.mean_cam_center[0],
            self.train_dl.dataset.mean_cam_center[1],
            self.train_dl.dataset.mean_cam_center[2]))

    def configure_optimizers(self):
        if self.hparams.oneCycleScheduler:
            # Setup optimization parameters. We set up this early
            self.optimizer = optim.AdamW(self.regressor.transformer_head.parameters(), lr=self.hparams.learning_rate)

            # Setup learning rate scheduler. For this policy, self.hparams.learning_rate is the learning_rate_min
            steps_per_epoch = len(self.train_dl)//self.hparams.num_gpus + 1 # the +1 is a Fixed typo for multi-gpu inference

            self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                           max_lr=self.hparams.learning_rate_max,
                                                           epochs=self.hparams.epochs,
                                                           steps_per_epoch=steps_per_epoch,
                                                           cycle_momentum=False,
                                                           ) # three_phase=True
        else:
            # Setup optimization parameters. We set up this early
            self.optimizer = optim.AdamW(self.regressor.transformer_head.parameters(), lr=self.hparams.learning_rate)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.95,
                                                                        patience=self.hparams.patience[1],
                                                                        verbose=True)

        # Check whether to resume training
        if self.hparams.resume_train:
            self.resume_model()  # we reduce the max epoch if we restore training
        elif self.hparams.resume_from_pretrain:
            self.resume_model_from_pretrain()

        return self.optimizer

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl

    def resume_model(self):
        # check if the model existed
        if not osp.isfile(self.output_map_file):
            print("Cannot find valid checkpoint, please check!")
            sys.exit()
        else:
            print("Resume model from ", self.output_map_file)

        checkpoint = torch.load(self.output_map_file)
        self.regressor.transformer_head.load_state_dict(checkpoint['aceformer_head'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.actual_epoch = checkpoint['training_epoch']+1 # update actual_epoch for resume training
        self.scheduler = checkpoint['scheduler']

        _logger.info("Resume training from epoch: {}. The max epochs is {}. There's {} epochs left".format(
            self.actual_epoch,
            self.hparams.epochs,
            self.hparams.epochs - self.actual_epoch))

    def resume_model_from_pretrain(self):
        # check if the model existed
        if not osp.isfile(self.hparams.pretrain_model_path):
            print("Cannot find valid checkpoint, please check!")
            sys.exit()
        else:
            print("Resume model from pretrained: ", self.hparams.pretrain_model_path)

        checkpoint = torch.load(self.hparams.pretrain_model_path)
        self.regressor.transformer_head.load_state_dict(checkpoint['aceformer_head'])
        # self.optimizer.load_state_dict(checkpoint['optimizer'])

        _logger.info("Resume training from epoch: {}. The max epochs is {}. There's {} epochs left".format(
            self.actual_epoch,
            self.hparams.epochs,
            self.hparams.epochs - self.actual_epoch))

    def save_model(self, ckpt_save_path=None):
        # NOTE: This would save the whole regressor (encoder weights included) in full precision floats (~30MB).
        # torch.save(self.regressor.state_dict(), self.options.output_map_file)

        # This saves just the transformer_head weights as half-precision floating point numbers.
        # The scene-agnostic encoder and scene coordinate head weights can then be loaded from the pretrained encoder file.
        transformer_head_state_dict = self.regressor.transformer_head.state_dict()
        for k, v in transformer_head_state_dict.items():
            if self.hparams.use_half:
                transformer_head_state_dict[k] = transformer_head_state_dict[k].half()
            else:
                transformer_head_state_dict[k] = transformer_head_state_dict[k]

        if ckpt_save_path==None:
            # we use default ckpt path: self.output_map_file
            save_file_name = self.output_map_file
        else:
            # we use designate ckpt path: ckpt_save_path
            save_file_name = ckpt_save_path

        torch.save({
            'training_epoch': self.actual_epoch,
            'optimizer': self.optimizer.state_dict(),
            'aceformer_head': transformer_head_state_dict,
            'scheduler': self.scheduler
        }, save_file_name)
        _logger.info(f"Saved trained head weights to: {save_file_name}")

    def on_train_epoch_start(self):
        # Enable benchmarking since all operations work on the same tensor size.
        self.regressor.transformer_head.train()
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

        # Keep track train loss of the epoch.
        self.train_loss_epoch = []
        # self.train_loss_trans_epoch = []
        # self.train_loss_rot_epoch = []

        # Keep track of rotation and translation errors for calculation of the median error.
        self.rErrs = []
        self.tErrs = []

        self.pct10_5 = 0 # 10cm/5deg
        self.pct5 = 0    # 5cm/5deg
        self.pct2 = 0    # 2cm/2deg
        self.pct1 = 0    # 1cm/1deg

        # re-initialize some of the buffers for saving intermediate prediction results if needed
        # currently only support coarse-to-fine scheme
        if self.coarse_to_fine_prediction:
            self.rErrs = [[] for i in range(self.num_predictions)]
            self.tErrs = [[] for i in range(self.num_predictions)]

            self.pct10_5 = [0 for i in range(self.num_predictions)]
            self.pct5    = [0 for i in range(self.num_predictions)]
            self.pct2    = [0 for i in range(self.num_predictions)]
            self.pct1    = [0 for i in range(self.num_predictions)]

    def compute_stats_on_errors(self, t_err, r_err, rErrs, tErrs, pct10_5, pct5, pct2, pct1, pct500_10=None, pct50_5=None, pct25_2=None):
        '''
        compute stats on errors
        t_err:  torch.tensor() [B] computed translation errors
        r_err:  torch.tensor() [B] computed rotation errors
        rErrs: list records on epoch
        tErrs: list records on epoch
        pct10_5: counters
        ...
        return
        '''
        for idx, (te, re) in enumerate(zip(t_err, r_err)):
            te = te.cpu().item()
            re = re.cpu().item()
            rErrs.append(re)
            tErrs.append(te*100)

            # check thresholds
            if re < 5 and te < 0.1: # 10cm/5deg
                pct10_5 += 1
            if re < 5 and te < 0.05:  # 5cm/5deg
                pct5 += 1
            if re < 2 and te < 0.02:  # 2cm/2deg
                pct2 += 1
            if re < 1 and te < 0.01:  # 1cm/1deg
                pct1 += 1

            if MORE_THRESHOLDS and pct500_10!=None:
                # more loose threshold
                if re < 10 and te < 5:  # 5m/10deg
                    pct500_10 += 1
                if re < 5 and te < 0.5:  # 50cm/5deg
                    pct50_5 += 1
                if re < 2 and te < 0.25:  # 25cm/2deg
                    pct25_2 += 1
                return rErrs, tErrs, pct10_5, pct5, pct2, pct1, pct500_10, pct50_5, pct25_2

        return rErrs, tErrs, pct10_5, pct5, pct2, pct1

    def training_step(self, batch, batch_idx):
        """
        Run one iteration of training, computing the pose error and minimising it.
        """
        self.regressor.transformer_head.train()
        gt_pose_B44 = batch['pose']
        intrinsics_B33 = batch['intrinsics']
        scene_coords_B13HW = batch['scene_coords']
        sc_mask = batch['sc_mask'] if 'sc_mask' in batch.keys() else None

        scene_coords_B13HW = scene_coords_B13HW.contiguous()
        gt_pose_B44.contiguous()

        B, _, C, H, W = scene_coords_B13HW.shape
        # Reshape to B3HW
        scene_coords_B3HW = scene_coords_B13HW.view(B, C, H, W)

        with autocast(enabled=self.hparams.use_half):
            pred_pose = self.regressor.get_pose(scene_coords_B3HW, intrinsics_B33,
                                                sc_mask=sc_mask, random_rescale_sc=self.random_rescale_sc)

            loss, trans_loss, rot_loss = self.loss(pred_pose, gt_pose_B44)

            with torch.no_grad():
                if self.coarse_to_fine_prediction:
                    for i in range(self.num_predictions):
                        t_err, r_err = compute_pose_error_new(pred_pose[i][:, :3, :4], gt_pose_B44[:, :3, :4])
                        self.rErrs[i], self.tErrs[i], \
                            self.pct10_5[i], self.pct5[i], \
                            self.pct2[i], self.pct1[i] = self.compute_stats_on_errors(t_err, r_err, self.rErrs[i], self.tErrs[i],
                                                                                      self.pct10_5[i], self.pct5[i], self.pct2[i],
                                                                                      self.pct1[i])
                else:
                    t_err, r_err = compute_pose_error_new(pred_pose[:, :3, :4], gt_pose_B44[:, :3, :4])
                    self.rErrs, self.tErrs, \
                        self.pct10_5, self.pct5, \
                        self.pct2, self.pct1 = self.compute_stats_on_errors(t_err, r_err, self.rErrs, self.tErrs,
                                                                            self.pct10_5, self.pct5, self.pct2, self.pct1)

        self.log('train/loss', loss, True, sync_dist=True)
        self.train_loss_epoch.append(loss.item())
        # self.train_loss_trans_epoch.append(trans_loss.item())
        # self.train_loss_rot_epoch.append(rot_loss.item())

        if self.hparams.oneCycleScheduler:
            self.scheduler.step()
            # _logger.info(f"lr: {self.optimizer.param_groups[0]['lr']:.8f}" )
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        return loss

    def on_validation_model_eval(self):
        self.regressor.eval()
        self.regressor.transformer_head.eval()

    def on_validation_epoch_start(self):
        self.rErrs_val = []
        self.tErrs_val = []

        self.pct10_5_val = 0  # 10cm/5deg
        self.pct5_val = 0  # 5cm/5deg
        self.pct2_val = 0  # 2cm/2deg
        self.pct1_val = 0  # 1cm/1deg
        self.val_loss_epoch = []
        # self.val_loss_trans_epoch = []
        # self.val_loss_rot_epoch = []

        if MORE_THRESHOLDS:
            # more loose thresholds
            self.pct500_10_val = 0
            self.pct50_5_val = 0
            self.pct25_2_val = 0


        # re-initialize some of the buffers for saving intermediate prediction results if needed
        # currently only support coarse-to-fine scheme
        if self.coarse_to_fine_prediction:
            self.rErrs_val = [[] for i in range(self.num_predictions)]
            self.tErrs_val = [[] for i in range(self.num_predictions)]

            self.pct10_5_val = [0 for i in range(self.num_predictions)]
            self.pct5_val = [0 for i in range(self.num_predictions)]
            self.pct2_val = [0 for i in range(self.num_predictions)]
            self.pct1_val = [0 for i in range(self.num_predictions)]

    def validation_step(self, batch_val, batch_idx):
        self.regressor.transformer_head.eval()
        gt_pose_B44 = batch_val['pose']
        intrinsics_B33 = batch_val['intrinsics']
        scene_coords_B13HW = batch_val['scene_coords']

        with torch.no_grad():
            B, _, C, H, W = scene_coords_B13HW.shape
            scene_coords_B3HW = scene_coords_B13HW.view(B, C, H, W).to(self.device)
            with autocast(enabled=self.hparams.use_half):
                pred_pose = self.regressor.get_pose(scene_coords_B3HW, intrinsics_B33)
            # compute validation pose
            loss, trans_loss, rot_loss = self.loss(pred_pose, gt_pose_B44.to(self.device))

        self.val_loss_epoch.append(loss.item())
        # self.val_loss_trans_epoch.append(trans_loss.item())
        # self.val_loss_rot_epoch.append(rot_loss.item())

        # compute pose error
        with torch.no_grad():
            if self.coarse_to_fine_prediction:
                for i in range(self.num_predictions):
                    t_err, r_err = compute_pose_error_new(pred_pose[i][:, :3, :4], gt_pose_B44[:, :3, :4])
                    self.rErrs_val[i], self.tErrs_val[i], \
                        self.pct10_5_val[i], self.pct5_val[i], \
                        self.pct2_val[i], self.pct1_val[i] = self.compute_stats_on_errors(t_err, r_err,
                                                                                          self.rErrs_val[i], self.tErrs_val[i],
                                                                                          self.pct10_5_val[i], self.pct5_val[i],
                                                                                          self.pct2_val[i],
                                                                                          self.pct1_val[i])
            else:
                t_err, r_err = compute_pose_error_new(pred_pose[:, :3, :4], gt_pose_B44[:, :3, :4])
                if MORE_THRESHOLDS:
                    self.rErrs_val, self.tErrs_val, \
                        self.pct10_5_val, self.pct5_val, \
                        self.pct2_val, self.pct1_val, \
                        self.pct500_10_val, self.pct50_5_val, \
                        self.pct25_2_val = self.compute_stats_on_errors(t_err, r_err, self.rErrs_val, self.tErrs_val,
                                                                        self.pct10_5_val, self.pct5_val, self.pct2_val,
                                                                        self.pct1_val,
                                                                        self.pct500_10_val, self.pct50_5_val,
                                                                        self.pct25_2_val)
                else:
                    self.rErrs_val, self.tErrs_val, \
                        self.pct10_5_val, self.pct5_val, \
                        self.pct2_val, self.pct1_val = self.compute_stats_on_errors(t_err, r_err, self.rErrs_val, self.tErrs_val,
                                                                                    self.pct10_5_val, self.pct5_val, self.pct2_val,
                                                                                    self.pct1_val)
        return loss

    def on_validation_epoch_end(self):
        val_loss_mean = self.log_validation_stats()

        # # update scheduler for reduce on plateau
        if self.hparams.oneCycleScheduler==False:
            self.scheduler.step(val_loss_mean)

        # Ideally, the lr should've been the same for different GPU
        lr_to_save = self.all_gather(self.optimizer.param_groups[0]['lr']).mean()

        if self.global_rank == 0:
            self.writer.add_scalar("lr", lr_to_save,
                                   self.actual_epoch)

        self.regressor.transformer_head.train()

    def on_validation_model_train(self):
        self.regressor.transformer_head.train() # we only train transformer_head

    def on_train_epoch_end(self):
        '''on_train_epoch_end() is after on_validation_end() in pl'''
        if self.actual_epoch % 1 == 0:
            if self.global_rank == 0:
                self.save_model()

        self.log_training_stats()

        if self.actual_epoch == self.hparams.epochs:
            self.trainer.should_stop=True
            _logger.info("The max training epoch has been reached: {}. Current actual epoch {}. Training terminates soon".format(
                self.hparams.epochs,
                self.actual_epoch))

        self.actual_epoch += 1

    def log_training_stats(self):
        train_loss_epoch_mean = self.all_gather(np.mean(self.train_loss_epoch)).mean()
        # train_loss_trans_epoch_mean = self.all_gather(np.mean(self.train_loss_trans_epoch)).mean()
        # train_loss_rot_epoch_mean = self.all_gather(np.mean(self.train_loss_rot_epoch)).mean()

        if self.global_rank == 0:
            # Losses
            self.writer.add_scalar("Loss/train", train_loss_epoch_mean, self.actual_epoch)  # all_gather() to collect all threads results
            # self.writer.add_scalar("Loss_Partial/train_trans", train_loss_trans_epoch_mean, self.actual_epoch)
            # self.writer.add_scalar("Loss_Partial/train_rot", train_loss_rot_epoch_mean, self.actual_epoch)

        if self.coarse_to_fine_prediction:
            total_frames = len(self.rErrs[0])
            # Compute median errors.
            tErrs_train = np.array(self.tErrs)
            rErrs_train = np.array(self.rErrs)
            median_tErrs = self.all_gather(np.median(tErrs_train, axis=1))
            mean_tErrs = self.all_gather(np.mean(tErrs_train, axis=1))
            median_rErrs = self.all_gather(np.median(rErrs_train, axis=1))
            mean_rErrs = self.all_gather(np.mean(rErrs_train, axis=1))

            total_frames = self.all_gather(total_frames).float()
            pct10_5 = self.all_gather(np.array(self.pct10_5)).float()
            pct5 = self.all_gather(np.array(self.pct5)).float()
            pct2 = self.all_gather(np.array(self.pct2)).float()
            pct1 = self.all_gather(np.array(self.pct1)).float()

            if self.global_rank == 0:
                if median_rErrs.dim() > 1:  # handle multi-gpu training
                    median_tErrs = median_tErrs.mean(dim=0)
                    mean_tErrs = mean_tErrs.mean(dim=0)
                    median_rErrs = median_rErrs.mean(dim=0)
                    mean_rErrs = mean_rErrs.mean(dim=0)

                    total_frames = total_frames.mean(dim=0)
                    pct10_5 = pct10_5.mean(dim=0)
                    pct5 = pct5.mean(dim=0)
                    pct2 = pct2.mean(dim=0)
                    pct1 = pct1.mean(dim=0)

                # Errors
                self.writer.add_scalar("Train_Error/train_median_rot/deg", median_rErrs[self.num_predictions-1], self.actual_epoch)
                self.writer.add_scalar("Train_Error/train_median_t/cm", median_tErrs[self.num_predictions-1], self.actual_epoch)
                self.writer.add_scalar("Train_Error/train_mean_rot/deg", mean_rErrs[self.num_predictions-1], self.actual_epoch)
                self.writer.add_scalar("Train_Error/train_mean_t/cm", mean_tErrs[self.num_predictions-1], self.actual_epoch)
                # Accuracies
                self.writer.add_scalar("Train_Accuracy/train_10cm_5deg_%", pct10_5[self.num_predictions-1] / total_frames * 100, self.actual_epoch)
                self.writer.add_scalar("Train_Accuracy/train_5cm_5deg_%", pct5[self.num_predictions-1] / total_frames * 100, self.actual_epoch)
                self.writer.add_scalar("Train_Accuracy/train_2cm_2deg_%", pct2[self.num_predictions-1] / total_frames * 100, self.actual_epoch)
                self.writer.add_scalar("Train_Accuracy/train_1cm_1deg_%", pct1[self.num_predictions-1] / total_frames * 100, self.actual_epoch)

                # log intermediate results
                for i in range(self.num_predictions - 1):
                    self.writer.add_scalar(f"{i:01d}_Train_Error/train_median_rot/deg", median_rErrs[i], self.actual_epoch)
                    self.writer.add_scalar(f"{i:01d}_Train_Error/train_median_t/cm", median_tErrs[i], self.actual_epoch)
                    self.writer.add_scalar(f"{i:01d}_Train_Error/train_mean_rot/deg", mean_rErrs[i], self.actual_epoch)
                    self.writer.add_scalar(f"{i:01d}_Train_Error/train_mean_t/cm", mean_tErrs[i], self.actual_epoch)

                    self.writer.add_scalar(f"{i:01d}_Train_Accuracy/train_10cm_5deg_%", pct10_5[i] / total_frames * 100, self.actual_epoch)
                    self.writer.add_scalar(f"{i:01d}_Train_Accuracy/train_5cm_5deg_%", pct5[i] / total_frames * 100, self.actual_epoch)
                    self.writer.add_scalar(f"{i:01d}_Train_Accuracy/train_2cm_2deg_%", pct2[i] / total_frames * 100, self.actual_epoch)
                    self.writer.add_scalar(f"{i:01d}_Train_Accuracy/train_1cm_1deg_%", pct1[i] / total_frames * 100, self.actual_epoch)

        else:
            total_frames = len(self.rErrs)
            # Compute median errors.
            tErrs_train = np.array(self.tErrs)
            rErrs_train = np.array(self.rErrs)
            median_tErrs = self.all_gather(np.median(tErrs_train)).mean()
            mean_tErrs = self.all_gather(np.mean(tErrs_train)).mean()
            median_rErrs = self.all_gather(np.median(rErrs_train)).mean()
            mean_rErrs = self.all_gather(np.mean(rErrs_train)).mean()

            total_frames = self.all_gather(total_frames).float().mean()
            pct10_5 = self.all_gather(self.pct10_5).float().mean()
            pct5 = self.all_gather(self.pct5).float().mean()
            pct2 = self.all_gather(self.pct2).float().mean()
            pct1 = self.all_gather(self.pct1).float().mean()

            if self.global_rank == 0:
                # Errors
                self.writer.add_scalar("Train_Error/train_median_rot/deg", median_rErrs, self.actual_epoch)
                self.writer.add_scalar("Train_Error/train_median_t/cm", median_tErrs, self.actual_epoch)
                self.writer.add_scalar("Train_Error/train_mean_rot/deg", mean_rErrs, self.actual_epoch)
                self.writer.add_scalar("Train_Error/train_mean_t/cm", mean_tErrs, self.actual_epoch)
                # Accuracies
                self.writer.add_scalar("Train_Accuracy/train_10cm_5deg_%", pct10_5 / total_frames * 100, self.actual_epoch)
                self.writer.add_scalar("Train_Accuracy/train_5cm_5deg_%", pct5 / total_frames * 100, self.actual_epoch)
                self.writer.add_scalar("Train_Accuracy/train_2cm_2deg_%", pct2 / total_frames * 100, self.actual_epoch)
                self.writer.add_scalar("Train_Accuracy/train_1cm_1deg_%", pct1 / total_frames * 100, self.actual_epoch)
        return

    def log_validation_stats(self):

        val_loss_mean = self.all_gather(np.mean(np.array(self.val_loss_epoch))).mean()
        # val_loss_trans_mean = self.all_gather(np.mean(np.array(self.val_loss_trans_epoch))).mean()
        # val_loss_rot_mean = self.all_gather(np.mean(np.array(self.val_loss_rot_epoch))).mean()

        if self.global_rank == 0:
            # Losses
            self.writer.add_scalar("Loss/val", val_loss_mean, self.actual_epoch)
            # self.writer.add_scalar("Loss_Partial/val_trans", val_loss_trans_mean, self.actual_epoch)
            # self.writer.add_scalar("Loss_Partial/val_rot", val_loss_rot_mean, self.actual_epoch)

        if self.coarse_to_fine_prediction:
            total_frames = len(self.rErrs_val[0])
            tErrs_val = np.array(self.tErrs_val) # [3,B]
            rErrs_val = np.array(self.rErrs_val) # [3,B]
            median_tErrs = self.all_gather(np.median(tErrs_val,axis=1)) # [3]
            mean_tErrs = self.all_gather(np.mean(tErrs_val,axis=1)) # [3]
            median_rErrs = self.all_gather(np.median(rErrs_val,axis=1)) # [3]
            mean_rErrs = self.all_gather(np.mean(rErrs_val,axis=1)) # [3]

            total_frames = self.all_gather(total_frames).float()
            pct10_5_val = self.all_gather(np.array(self.pct10_5_val)).float()
            pct5_val = self.all_gather(np.array(self.pct5_val)).float()
            pct2_val = self.all_gather(np.array(self.pct2_val)).float()
            pct1_val = self.all_gather(np.array(self.pct1_val)).float()

            if self.global_rank == 0:

                if median_rErrs.dim()>1: # handle multi-gpu training
                    median_tErrs = median_tErrs.mean(dim=0)
                    mean_tErrs = mean_tErrs.mean(dim=0)
                    median_rErrs = median_rErrs.mean(dim=0)
                    mean_rErrs = mean_rErrs.mean(dim=0)

                    total_frames = total_frames.mean(dim=0)
                    pct10_5_val = pct10_5_val.mean(dim=0)
                    pct5_val = pct5_val.mean(dim=0)
                    pct2_val = pct2_val.mean(dim=0)
                    pct1_val = pct1_val.mean(dim=0)

                _logger.info(
                    f"Median Rot Err: {median_rErrs[self.num_predictions-1]:.2f}deg, Median Trans Err: {median_tErrs[self.num_predictions-1]:.1f}cm, "
                    f"Mean Rot Err: {mean_rErrs[self.num_predictions-1]:.2f}deg, Mean Trans Err: {mean_tErrs[self.num_predictions-1]:.1f}cm")

                # Errors
                self.writer.add_scalar("Error/val_median_rot/deg", median_rErrs[self.num_predictions-1], self.actual_epoch)
                self.writer.add_scalar("Error/val_median_t/cm", median_tErrs[self.num_predictions-1], self.actual_epoch)
                self.writer.add_scalar("Error/val_mean_rot/deg", mean_rErrs[self.num_predictions-1], self.actual_epoch)
                self.writer.add_scalar("Error/val_mean_t/cm", mean_tErrs[self.num_predictions-1], self.actual_epoch)
                # Accuracies
                self.writer.add_scalar("Val_Accuracy/val_10cm_5deg_%", pct10_5_val[self.num_predictions-1] / total_frames * 100, self.actual_epoch)
                self.writer.add_scalar("Val_Accuracy/val_5cm_5deg_%", pct5_val[self.num_predictions-1] / total_frames * 100, self.actual_epoch)
                self.writer.add_scalar("Val_Accuracy/val_2cm_2deg_%", pct2_val[self.num_predictions-1] / total_frames * 100, self.actual_epoch)
                self.writer.add_scalar("Val_Accuracy/val_1cm_1deg_%", pct1_val[self.num_predictions-1] / total_frames * 100, self.actual_epoch)

                # log intermediate results
                for i in range(self.num_predictions-1):
                    self.writer.add_scalar(f"{i:01d}_Error/val_median_rot/deg", median_rErrs[i], self.actual_epoch)
                    self.writer.add_scalar(f"{i:01d}_Error/val_median_t/cm", median_tErrs[i], self.actual_epoch)
                    self.writer.add_scalar(f"{i:01d}_Error/val_mean_rot/deg", mean_rErrs[i], self.actual_epoch)
                    self.writer.add_scalar(f"{i:01d}_Error/val_mean_t/cm", mean_tErrs[i], self.actual_epoch)

                    self.writer.add_scalar(f"{i:01d}_Val_Accuracy/val_10cm_5deg_%", pct10_5_val[i] / total_frames * 100, self.actual_epoch)
                    self.writer.add_scalar(f"{i:01d}_Val_Accuracy/val_5cm_5deg_%", pct5_val[i] / total_frames * 100, self.actual_epoch)
                    self.writer.add_scalar(f"{i:01d}_Val_Accuracy/val_2cm_2deg_%", pct2_val[i] / total_frames * 100, self.actual_epoch)
                    self.writer.add_scalar(f"{i:01d}_Val_Accuracy/val_1cm_1deg_%", pct1_val[i] / total_frames * 100, self.actual_epoch)

        else:
            total_frames = len(self.rErrs_val)
            tErrs_val = np.array(self.tErrs_val)
            rErrs_val = np.array(self.rErrs_val)
            median_tErrs = self.all_gather(np.median(tErrs_val)).mean()
            mean_tErrs = self.all_gather(np.mean(tErrs_val)).mean()
            median_rErrs = self.all_gather(np.median(rErrs_val)).mean()
            mean_rErrs = self.all_gather(np.mean(rErrs_val)).mean()

            total_frames = self.all_gather(total_frames).float().mean()
            pct10_5_val = self.all_gather(self.pct10_5_val).float().mean()
            pct5_val = self.all_gather(self.pct5_val).float().mean()
            pct2_val = self.all_gather(self.pct2_val).float().mean()
            pct1_val = self.all_gather(self.pct1_val).float().mean()

            if MORE_THRESHOLDS:
                pct500_10_val = self.all_gather(self.pct500_10_val).float().mean()
                pct50_5_val = self.all_gather(self.pct50_5_val).float().mean()
                pct25_2_val = self.all_gather(self.pct25_2_val).float().mean()

            # if self.trainer.is_global_zero:
            if self.global_rank == 0:
                _logger.info(
                    f"Median Rot Err: {median_rErrs:.2f}deg, Median Trans Err: {median_tErrs:.1f}cm, "
                    f"Mean Rot Err: {mean_rErrs:.2f}deg, Mean Trans Err: {mean_tErrs:.1f}cm")

                _logger.info(
                    f"val_10cm_5deg_%: {pct10_5_val / total_frames * 100:.2f}, "
                    f"val_5cm_5deg_%: {pct5_val / total_frames * 100:.2f}, "
                    f"val_2cm_2deg_%: {pct2_val / total_frames * 100:.2f}, "
                    f"val_1cm_1deg_%: {pct1_val / total_frames * 100:.2f}"
                )

                # Errors
                self.writer.add_scalar("Error/val_median_rot/deg", median_rErrs.mean(), self.actual_epoch)
                self.writer.add_scalar("Error/val_median_t/cm", median_tErrs.mean(), self.actual_epoch)
                self.writer.add_scalar("Error/val_mean_rot/deg", mean_rErrs.mean(), self.actual_epoch)
                self.writer.add_scalar("Error/val_mean_t/cm", mean_tErrs.mean(), self.actual_epoch)
                # Accuracies
                self.writer.add_scalar("Val_Accuracy/val_10cm_5deg_%", pct10_5_val / total_frames * 100, self.actual_epoch)
                self.writer.add_scalar("Val_Accuracy/val_5cm_5deg_%", pct5_val / total_frames * 100, self.actual_epoch)
                self.writer.add_scalar("Val_Accuracy/val_2cm_2deg_%", pct2_val / total_frames * 100, self.actual_epoch)
                self.writer.add_scalar("Val_Accuracy/val_1cm_1deg_%", pct1_val / total_frames * 100, self.actual_epoch)

                if MORE_THRESHOLDS:
                    self.writer.add_scalar("Val2_Accuracy/val_500cm_10deg_%", pct500_10_val / total_frames * 100, self.actual_epoch)
                    self.writer.add_scalar("Val2_Accuracy/val_50cm_5deg_%", pct50_5_val / total_frames * 100, self.actual_epoch)
                    self.writer.add_scalar("Val2_Accuracy/val_25cm_2deg_%", pct25_2_val / total_frames * 100, self.actual_epoch)

        return val_loss_mean

    def log_tb_vis(self, image_B1HW, scene_coords_B3HW_input, pred_sc, pred_ng, split=''):
        '''
        saving visual image, SC before, SC after, and neural guidance map to tensorboard
        '''

        image_batch = [plot_image(image_B1HW[frame_idx], True, False) for frame_idx in range(image_B1HW.shape[0])]
        self.writer.add_figure(f"{split}_image_batch", image_batch, self.actual_epoch)

        sc_maps_ACE_head = [plot_image_saliancy(scene_coords_B3HW_input[frame_idx], True, True) for frame_idx in range(image_B1HW.shape[0])]
        self.writer.add_figure(f"{split}_sc_maps_ACE_head", sc_maps_ACE_head, self.actual_epoch)

        sc_maps_AceFormer = [plot_image_saliancy(pred_sc[frame_idx], True, True) for frame_idx in range(image_B1HW.shape[0])]
        self.writer.add_figure(f"{split}_sc_maps_AceFormer", sc_maps_AceFormer, self.actual_epoch)

        ng_maps_AceFormer = [plot_image_saliancy(pred_ng[frame_idx], True, True) for frame_idx in range(image_B1HW.shape[0])]
        self.writer.add_figure(f"{split}_ng_maps_AceFormer", ng_maps_AceFormer, self.actual_epoch)

        alpha_blending_img_ng_maps = [plot_image_saliancy_with_blending(image_B1HW[frame_idx], pred_ng[frame_idx], True, True) for frame_idx in range(image_B1HW.shape[0])]
        self.writer.add_figure(f"{split}_alpha_blending_img_ng_maps_AceFormer", alpha_blending_img_ng_maps, self.actual_epoch)

    def test_step(self):
        # TO be implemented
        NotImplementedError


