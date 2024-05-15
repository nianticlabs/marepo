#!/bin/bash

############################## training the marepo V100_3 # CVPR Paper Model
# Notice the ace_head_dummy.pt is a dummy ACE head checkpoint for constructing the marepo network.
# In our marepo training scheme, we do not update this part of the network
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ../train_marepo.py \
marepo_12T1R256_paper_model \
--dataset_path ../../map_free_training_scenes_aug_16_all \
--dataset_head_network_path ../logs/mapfree_all \
--transformer_APR_head True \
--head_network_path ../ace_head_dummy.pt \
--use_half=False \
--batch_size 8 \
--trainskip 1 \
--testskip 1 \
--learning_rate 0.0003 \
--epochs 150 \
--transformer_json ../transformer/config/nerf_focal_12T1R_256_homo_c2f.json \
--oneCycleScheduler True \
--marepo_sc_augment True \
--jitter_rot 180.0 \
--num_gpus 8 \
# 2>&1 | tee "tmp/marepo.txt"
