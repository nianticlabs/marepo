#!/bin/bash
SCRIPT_PATH=$(dirname $(realpath -s "$0"))
UPPER_REPO_PATH=$(realpath -s "${SCRIPT_PATH}/../..") # the path to the upper directory of the root of the marepo repo

############# preprocess mapfree data to generate scene coordinate files
# Notice the ace_head_dummy.pt is a dummy ACE head checkpoint for constructing the ACE Network.
# In preprocessing, we will actually dynamically load the correct ACE heads on-the-fly.

CUDA_VISIBLE_DEVICES=0 python ../preprocess_marepo.py \
--dataset_path ${UPPER_REPO_PATH}/map_free_training_scenes_aug_16 \
--dataset_head_network_path ${UPPER_REPO_PATH}/marepo/logs/mapfree \
--head_network_path ../ace_head_dummy.pt \
--use_half=False \
--preprocessing True \
--trainskip 1 \
--testskip 1 \
--scheme2 True

CUDA_VISIBLE_DEVICES=1 python ../preprocess_marepo.py \
--dataset_path ${UPPER_REPO_PATH}/map_free_training_scenes_aug_16_flip \
--dataset_head_network_path ${UPPER_REPO_PATH}/marepo/logs/mapfree_flip \
--head_network_path ../ace_head_dummy.pt \
--use_half=False \
--preprocessing True \
--trainskip 1 \
--testskip 1 \
--scheme2 True

############## Generate scheme 3 data augmentation files
OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=2 python ../preprocess_marepo.py \
--dataset_path ${UPPER_REPO_PATH}/map_free_training_scenes_aug_16 \
--dataset_head_network_path ${UPPER_REPO_PATH}/marepo/logs/mapfree \
--head_network_path ../ace_head_dummy.pt \
--use_half=False \
--preprocessing True \
--trainskip 1 \
--testskip 1 \
--scheme3 True \
--scheme3_aug_number 0 16

OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=3 python ../preprocess_marepo.py \
--dataset_path ${UPPER_REPO_PATH}/map_free_training_scenes_aug_16_flip \
--dataset_head_network_path ${UPPER_REPO_PATH}/marepo/logs/mapfree_flip \
--head_network_path ../ace_head_dummy.pt \
--use_half=False \
--preprocessing True \
--trainskip 1 \
--testskip 1 \
--scheme3 True \
--scheme3_aug_number 0 16 # Hint: instead of serial generation, we could do parallel generation, i.e. 0-4, 4-8, 8-12, 12-16 to accelerate the process

############### Use Softlinks to generate flip + no flip combined dataset
new_data_dir="${UPPER_REPO_PATH}/map_free_training_scenes_aug_16_all"
mkdir ${new_data_dir}

no_flip_dir="${UPPER_REPO_PATH}/map_free_training_scenes_aug_16"
flip_dir="${UPPER_REPO_PATH}/map_free_training_scenes_aug_16_flip"

mkdir "${new_data_dir}/train"
mkdir "${new_data_dir}/val"
mkdir "${new_data_dir}/test"

ln -s "${no_flip_dir}/train"/* "${new_data_dir}/train"
ln -s "${no_flip_dir}/val"/* "${new_data_dir}/val"
ln -s "${no_flip_dir}/val"/* "${new_data_dir}/train" # put val to train for final model
ln -s "${no_flip_dir}/test"/* "${new_data_dir}/test"

ln -s "${flip_dir}/train"/* "${new_data_dir}/train"
ln -s "${flip_dir}/val"/* "${new_data_dir}/val"
ln -s "${flip_dir}/val"/* "${new_data_dir}/train" # put val to train for final model

############### Use Softlinks to generate flip + no flip combined ace head logs
new_ace_head_dir="${UPPER_REPO_PATH}/marepo/logs/mapfree_all"
mkdir ${new_ace_head_dir}

no_flip_dir="${UPPER_REPO_PATH}/marepo/logs/mapfree"
flip_dir="${UPPER_REPO_PATH}/marepo/logs/mapfree_flip"

mkdir "${new_ace_head_dir}/train"
mkdir "${new_ace_head_dir}/val"
mkdir "${new_ace_head_dir}/test"

ln -s "${no_flip_dir}/train"/* "${new_ace_head_dir}/train"
ln -s "${no_flip_dir}/val"/* "${new_ace_head_dir}/val"
ln -s "${no_flip_dir}/test"/* "${new_ace_head_dir}/test"

ln -s "${flip_dir}/train"/* "${new_ace_head_dir}/train"
ln -s "${flip_dir}/val"/* "${new_ace_head_dir}/val"
