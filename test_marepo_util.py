#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2024.

import argparse
import logging
import math
import time
from distutils.util import strtobool
from pathlib import Path
import json
import random

import cv2
import numpy as np
import torch
from marepo.metrics import compute_pose_error_new
import matplotlib.pyplot as plt

def single_frame_test_time_error_computation(predict_pose, gt_pose_B44, intrinsics_B33, filenames, rErrs, tErrs,
                                             pct10_5, pct5, pct2, pct1,
                                             pct500_10, pct50_5, pct25_2,
                                             pose_log, avg_batch_time, batch_start_time, num_batches, _logger):
    '''
    moved the previous test time functions here so that code is less ugly
    '''
    for frame_idx, (out_pose, gt_pose_44, intrinsics_33, frame_path) in enumerate(
            zip(predict_pose, gt_pose_B44, intrinsics_B33, filenames)):

        # Remove path from file name
        frame_name = Path(frame_path).name

        # Calculate translation error.
        t_err = float(torch.norm(gt_pose_44[0:3, 3] - out_pose[0:3, 3]))

        # Rotation error.
        gt_R = gt_pose_44[0:3, 0:3].numpy()
        out_R = out_pose[0:3, 0:3].numpy()

        r_err = np.matmul(out_R, np.transpose(gt_R))
        # Compute angle-axis representation.
        r_err = cv2.Rodrigues(r_err)[0]
        # Extract the angle.
        r_err = np.linalg.norm(r_err) * 180 / math.pi
        # _logger.info(f"Rotation Error: {r_err:.2f}deg, Translation Error: {t_err * 100:.1f}cm")

        # Save the errors.
        rErrs.append(r_err)
        tErrs.append(t_err * 100)

        # Check various thresholds.
        if r_err < 5 and t_err < 0.1:  # 10cm/5deg
            pct10_5 += 1
        if r_err < 5 and t_err < 0.05:  # 5cm/5deg
            pct5 += 1
        if r_err < 2 and t_err < 0.02:  # 2cm/2deg
            pct2 += 1
        if r_err < 1 and t_err < 0.01:  # 1cm/1deg
            pct1 += 1

        # more loose threshold
        if r_err < 10 and t_err < 5:  # 5m/10deg
            pct500_10 += 1
        if r_err < 5 and t_err < 0.5:  # 50cm/5deg
            pct50_5 += 1
        if r_err < 2 and t_err < 0.25:  # 25cm/2deg
            pct25_2 += 1

        # Write estimated pose to pose file (inverse).
        out_pose = out_pose.inverse()

        # Translation.
        t = out_pose[0:3, 3]

        # Rotation to axis angle.
        rot, _ = cv2.Rodrigues(out_pose[0:3, 0:3].numpy())
        angle = np.linalg.norm(rot)
        axis = rot / angle

        # Axis angle to quaternion.
        q_w = math.cos(angle * 0.5)
        q_xyz = math.sin(angle * 0.5) * axis

        # Write to output file. All in a single line.
        pose_log.write(f"{frame_name} "
                       f"{q_w} {q_xyz[0].item()} {q_xyz[1].item()} {q_xyz[2].item()} "
                       f"{t[0]} {t[1]} {t[2]} "
                       f"{r_err} {t_err}\n")

    avg_batch_time += time.time() - batch_start_time
    num_batches += 1
    return rErrs, tErrs, avg_batch_time, num_batches, \
                    pct10_5, pct5, pct2, pct1, \
                    pct500_10, pct50_5, pct25_2

def compute_stats_on_errors(t_err, r_err, rErrs, tErrs, pct10_5, pct5, pct2, pct1, pct500_10, pct50_5, pct25_2):
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
        tErrs.append(te * 100)

        # check thresholds
        if re < 5 and te < 0.1: # 10cm/5deg
            pct10_5 += 1
        if re < 5 and te < 0.05:  # 5cm/5deg
            pct5 += 1
        if re < 2 and te < 0.02:  # 2cm/2deg
            pct2 += 1
        if re < 1 and te < 0.01:  # 1cm/1deg
            pct1 += 1

        # more loose thresholds
        if re < 10 and te < 5:  # 5m/10deg
            pct500_10 += 1
        if re < 5 and te < 0.5:  # 50cm/5deg
            pct50_5 += 1
        if re < 2 and te < 0.25:  # 25cm/2deg
            pct25_2 += 1
    return rErrs, tErrs, pct10_5, pct5, pct2, pct1, pct500_10, pct50_5, pct25_2

def batch_frame_test_time_error_computation(predict_pose, gt_pose_B44, intrinsics_B33, filenames, rErrs, tErrs,
                                             pct10_5, pct5, pct2, pct1,
                                             pct500_10, pct50_5, pct25_2,
                                             pose_log, avg_batch_time, batch_start_time, num_batches, _logger):
    '''
    moved the previous test time functions here so that code is less ugly
    '''

    # here the t_err is in meters
    t_err, r_err = compute_pose_error_new(predict_pose[:, :3, :4], gt_pose_B44[:, :3, :4])

    # the tErrs is in centimeters because of te * 100 in compute_stats_on_errors()
    rErrs, tErrs, pct10_5, pct5, pct2, pct1, pct500_10, pct50_5, pct25_2 = \
        compute_stats_on_errors(t_err, r_err, rErrs, tErrs, pct10_5, pct5, pct2, pct1, pct500_10, pct50_5, pct25_2)

    # rest are logging, which could be comment out if needed
    for frame_idx, (out_pose, gt_pose_44, frame_path) in enumerate(
            zip(predict_pose, gt_pose_B44, filenames)):

        # Remove path from file name
        frame_name = Path(frame_path).name
        _logger.info(f"Rotation Error: {r_err[frame_idx]:.2f}deg, Translation Error: {t_err[frame_idx] * 100:.1f}cm")

        # Write estimated pose to pose file (inverse).
        out_pose = out_pose.inverse()

        # Translation.
        t = out_pose[0:3, 3]

        # Rotation to axis angle.
        rot, _ = cv2.Rodrigues(out_pose[0:3, 0:3].numpy())
        angle = np.linalg.norm(rot)
        axis = rot / angle

        # Axis angle to quaternion.
        q_w = math.cos(angle * 0.5)
        q_xyz = math.sin(angle * 0.5) * axis

        # Write to output file. All in a single line.
        pose_log.write(f"{frame_name} "
                       f"{q_w} {q_xyz[0].item()} {q_xyz[1].item()} {q_xyz[2].item()} "
                       f"{t[0]} {t[1]} {t[2]} "
                       f"{r_err} {t_err}\n")


    avg_batch_time += time.time() - batch_start_time
    num_batches += 1

    return rErrs, tErrs, avg_batch_time, num_batches, \
        pct10_5, pct5, pct2, pct1, \
        pct500_10, pct50_5, pct25_2

def vis_pose(vis_info, scene=None):
    '''
    visualize predicted pose result vs. gt pose
    '''

    pose = vis_info['pose']
    pose_gt = vis_info['pose_gt']

    # plot translation traj.
    fig = plt.figure(figsize = (8,6))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax1 = fig.add_axes([0, 0.2, 0.9, 0.85], projection='3d')
    ax1.scatter(pose[10:,0],pose[10:,1],zs=pose[10:,2], c='r', s=3**2,depthshade=0) # predict
    ax1.scatter(pose_gt[:,0], pose_gt[:,1], zs=pose_gt[:,2], c='g', s=3**2,depthshade=0) # GT
    ax1.scatter(pose[0:10,0],pose[0:10,1],zs=pose[0:10,2], c='k', s=3**2,depthshade=0) # predict
    ax1.view_init(30, 120)
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_zlabel('z (m)')

    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_zlim(-1, 1)

    if scene==None:
        fname = 'tmp/vis_pose.png'
    else:
        fname=f'tmp/vis_pose_{scene}.png'
    plt.savefig(fname, dpi=100)

