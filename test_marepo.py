#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2024.

import argparse
import logging
import time
from distutils.util import strtobool
from pathlib import Path
import json
import random
import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from marepo.marepo_network import Regressor
from dataset import CamLocDataset
from test_marepo_util import batch_frame_test_time_error_computation

_logger = logging.getLogger(__name__)

def _strtobool(x):
    return bool(strtobool(x))

if __name__ == '__main__':
    # Setup logging.
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Test a trained network on a specific scene.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('scene', type=Path,
                        help='path to a scene in the dataset folder, e.g. "datasets/Cambridge_GreatCourt"')
    parser.add_argument('network', type=Path, help='path to marepo network (just the transformer weights)')
    parser.add_argument('--dataset_path', type=Path,
                        default="",
                        help='path to the dataset folder, e.g. "~/storage/map_free_training_scenes/". '
                             'When this config is set, it means we are training with every scenes in the dataset')
    parser.add_argument('--encoder_path', type=Path, default=Path(__file__).parent / "ace_encoder_pretrained.pt",
                        help='file containing pre-trained encoder weights')
    parser.add_argument('--head_network_path', type=Path,
                        default=Path(__file__).parent / "logs/wayspots_bears/wayspots_bears.pt",
                        help='file containing pre-trained ACE head weights')
    parser.add_argument('--dataset_head_network_path', type=Path,
                        default="",
                        help='path to the pre-trained ACE head weights of entire dataset, e.g. "logs/mapfree/". '
                             'When this config is set, it means we are training with every scenes in the dataset')
    parser.add_argument('--preprocessing', type=_strtobool, default=False,
                        help='use pretrained ACE networks to generate scene coordinate maps (Not used in testing)')
    parser.add_argument('--session', '-sid', default='',
                        help='custom session name appended to output files, '
                             'useful to separate different runs of a script')

    parser.add_argument('--image_resolution', type=int, default=480, help='base image resolution')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='number of train set batch size')
    parser.add_argument('--val_batch_size', type=int, default=1,
                        help='number of val set batch size')
    parser.add_argument('--test_batch_size', type=int, default=64,
                        help='number of test set batch size')
    parser.add_argument('--use_half', type=_strtobool, default=False,
                        help='train with half precision')
    parser.add_argument('--trainskip', type=int, default=1,
                        help='uniformly subsample train set by 1/trainskip')
    parser.add_argument('--testskip', type=int, default=1,
                        help='uniformly subsample val/test set by 1/testskip')
    parser.add_argument('--transformer_json', type=str, default="../transformer/config/default.json",
                        help='file contain transformer config')
    parser.add_argument('--load_scheme2_sc_map', type=_strtobool, default=False,
                        help='use saved SC maps (subtract mean) and GT pose (subtract mean)'
                             'instead of use original SC map and GT pose')
    parser.add_argument('--datatype', type=str, default="test", choices=['train', 'val', 'test'],
                        help='dataset type: train means mapping data, test means query data')

    parser.add_argument('--center_crop', type=_strtobool, default=False,
                        help='Flag for datasetloader indicating images need center crop to make them proportional in size to MapFree data')

    parser.add_argument('--load_rgb', type=_strtobool, default=False,
                        help='Use 3 rgb channel images instead of using 1 channel gray image.')
    parser.add_argument('--data_split', type=str, default='test',
                        choices=('train', 'test'),help='data split')

    # noise jitter experiment to SC Map
    parser.add_argument('--noise_ratio', type=float, default=0.0,
                        help='noise ratio added to the sc map, in percentage 0-1')
    parser.add_argument('--noise_level', type=float, default=0.0,
                        help='noise level added to the sc map, in cm, i.e.0.1m, 0.5m')

    opt = parser.parse_args()
    device = torch.device("cuda")
    scene_path = Path(opt.scene)
    encoder_path = Path(opt.encoder_path)
    head_network_path = Path(opt.head_network_path)
    transformer_path = Path(opt.network)
    session = opt.session
    num_data_loader_workers=6
    torch.manual_seed(2089)
    np.random.seed(2089)
    random.seed(2089)

    test_batch_size = opt.test_batch_size

    # print("warning: change it back to test after debugging!")
    # Create test dataset.
    test_dataset = CamLocDataset(
        root_dir=opt.scene / "test",
        mode=0,  # Default for marepo, we don't need scene coordinates/RGB-D.
        image_height=opt.image_resolution,
        center_crop=opt.center_crop,
        load_rgb=opt.load_rgb
    )

    test_dl = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=test_batch_size,
        num_workers=num_data_loader_workers)
    _logger.info(f'Test images found: {len(test_dl.dataset)}')

    # Setup dataloader. Batch size 1 by default.
    testset_loader = DataLoader(test_dataset, shuffle=False, num_workers=6)

    # Load network weights.
    encoder_state_dict = torch.load(encoder_path, map_location="cpu")
    _logger.info(f"Loaded encoder from: {encoder_path}")
    head_state_dict = torch.load(head_network_path, map_location="cpu")
    _logger.info(f"Loaded head weights from: {head_network_path}")
    transformer_state_dict = torch.load(transformer_path, map_location="cpu")
    _logger.info(f"Loaded transformer weights from: {transformer_path}")

    # some configuration for the transformer
    f = open(opt.transformer_json)
    config = json.load(f)
    f.close()
    config["transformer_pose_mean"] = torch.Tensor([0., 0., 0.])  # placeholder, load actual numbers later
    _logger.info(f"Loaded transformer config from: {opt.transformer_json}")
    default_img_H = test_dl.dataset.default_img_H  # we get default image H and W for position encoding
    default_img_W = test_dl.dataset.default_img_W
    config["default_img_HW"] = [default_img_H, default_img_W]

    # Create regressor.
    network = Regressor.load_marepo_from_state_dict(encoder_state_dict, head_state_dict, transformer_state_dict, config)
    network.eval()

    # if network is trained by scheme 2, we should load the Ace Head mean stats
    print("Warning: we load mean again because we need to shift SC mean at test")
    if opt.load_scheme2_sc_map:
        network.transformer_head.transformer_pose_mean = network.heads.mean

    # Setup for evaluation.
    network = network.to(device)
    network.eval()

    # Save the outputs in the same folder as the network being evaluated.
    output_dir = transformer_path.parent
    scene_name = scene_path.name

    # This will contain aggregate scene stats (median translation/rotation errors, and avg processing time per frame).
    test_log_file = output_dir / f'test_{scene_name}_{opt.session}.txt'
    _logger.info(f"Saving test aggregate statistics to: {test_log_file}")
    # This will contain each frame's pose (stored as quaternion + translation) and errors.
    pose_log_file = output_dir / f'poses_{scene_name}_{opt.session}.txt'
    _logger.info(f"Saving per-frame poses and errors to: {pose_log_file}")

    # Setup output files.
    test_log = open(test_log_file, 'w', 1)
    pose_log = open(pose_log_file, 'w', 1)

    # Metrics of interest.
    avg_batch_time = 0
    num_batches = 0

    # Keep track of rotation and translation errors for calculation of the median error.
    rErrs = []
    tErrs = []

    # Percentage of frames predicted within certain thresholds from their GT pose.
    pct10_5 = 0
    pct5 = 0
    pct2 = 0
    pct1 = 0

    # more loose thresholds
    pct500_10 = 0
    pct50_5 = 0
    pct25_2 = 0

    # Testing loop.
    testing_start_time = time.time()
    cur_batch_frame_index = 0
    with torch.no_grad():
        for batch in test_dl:
            image_B1HW, image_mask_B1HW = batch['image'], batch['image_mask']
            gt_pose_B44, gt_pose_inv_B44 = batch['pose'], batch['pose_inv']
            intrinsics_B33, intrinsics_inv_B33 = batch['intrinsics'], batch['intrinsics_inv']
            scene_coords_B13HW = batch['scene_coords']
            sc_mask = batch['sc_mask'] if 'sc_mask' in batch.keys() else None
            filenames = batch['rgb_files']

            batch_start_time = time.time()

            batch_size = image_B1HW.shape[0]
            image_B1HW = image_B1HW.to(device, non_blocking=True)

            # Predict scene coordinates.
            with autocast(enabled=True):
                features = network.get_features(image_B1HW)
                sc = network.get_scene_coordinates(features).float() # [N,3,H,W]

            # Predict pose
            with autocast(enabled=False):
                predict_pose = network.get_pose(sc, intrinsics_B33.to(device))
                predict_pose = predict_pose.float().cpu()

            rErrs, tErrs, avg_batch_time, num_batches, \
                pct10_5, pct5, pct2, pct1, \
                pct500_10, pct50_5, pct25_2 \
                = batch_frame_test_time_error_computation(predict_pose, gt_pose_B44, intrinsics_B33, filenames, rErrs, tErrs,
                                                           pct10_5, pct5, pct2, pct1,
                                                           pct500_10, pct50_5, pct25_2,
                                                           pose_log, avg_batch_time, batch_start_time, num_batches, _logger)

    total_frames = len(rErrs)
    assert total_frames == len(test_dl.dataset.rgb_files)

    # Compute median errors.
    median_rErr = np.median(rErrs)
    median_tErr = np.median(tErrs)
    mean_rErr = np.mean(rErrs)
    mean_tErr = np.mean(tErrs)

    # Compute average time.
    avg_time = avg_batch_time / total_frames

    # Compute final metrics.
    pct10_5 = pct10_5 / total_frames * 100
    pct5 = pct5 / total_frames * 100
    pct2 = pct2 / total_frames * 100
    pct1 = pct1 / total_frames * 100

    pct500_10 = pct500_10 / total_frames * 100
    pct50_5 = pct50_5 / total_frames * 100
    pct25_2 = pct25_2 / total_frames * 100

    _logger.info("===================================================")
    _logger.info("Test complete.")

    _logger.info('Accuracy:')
    _logger.info(f'\t5m/10deg: {pct500_10:.2f}%')
    _logger.info(f'\t0.5m/5deg: {pct50_5:.2f}%')
    _logger.info(f'\t0.25m/2deg: {pct25_2:.2f}%')

    _logger.info(f'\t10cm/5deg: {pct10_5:.2f}%')
    _logger.info(f'\t5cm/5deg: {pct5:.2f}%')
    _logger.info(f'\t2cm/2deg: {pct2:.2f}%')
    _logger.info(f'\t1cm/1deg: {pct1:.2f}%')

    _logger.info(f"Median Error: {median_rErr:.2f} deg, {median_tErr:.2f} cm")
    _logger.info(f"Mean Error: {mean_rErr:.2f} deg, {mean_tErr:.2f} cm")
    _logger.info(f"Avg. processing time: {avg_time * 1000:4.2f} ms")

    # Write to the test log file as well.
    test_log.write(f"{median_rErr} {median_tErr} {avg_time}\n")

    test_log.close()
    pose_log.close()

