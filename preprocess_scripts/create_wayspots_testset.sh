#!/bin/bash

# In this script, we copy the wayspots test sets to the new folder
# so that we could generate scheme3 data augmentation for the mapping sequences, and finetune our baseline models
SCRIPT_PATH=$(dirname $(realpath -s "$0"))
UPPER_REPO_PATH=$(realpath -s "${SCRIPT_PATH}/../..") # the path to the upper directory of the root of the marepo repo

new_dir="${UPPER_REPO_PATH}/wayspots_finetune_dataset"
testset_path="${UPPER_REPO_PATH}/marepo/datasets"

mkdir ${new_dir}
cd ${testset_path}

for scene in "wayspots_"*
do
  echo "${scene}"
  mkdir "${new_dir}/${scene}"
  mkdir "${new_dir}/${scene}/train"
  mkdir "${new_dir}/${scene}/val"
  mkdir "${new_dir}/${scene}/test"
  cp -d -r "$scene" "${new_dir}/${scene}/train"
  ln -s "${new_dir}/${scene}/train/${scene}" "${new_dir}/${scene}/val" # Note: this could be softlinks to save disk space
  ln -s "${new_dir}/${scene}/train/${scene}" "${new_dir}/${scene}/test" # Note: this could be softlinks to save disk space
done

############## Use Softlinks to generate ace head logs
# new_ace_head_dir="${UPPER_REPO_PATH}/marepo/logs/wayspots_pretrain"
# mkdir ${new_ace_head_dir}

# no_flip_dir="${UPPER_REPO_PATH}/marepo/logs/wayspots"

# mkdir "${new_ace_head_dir}/train"
# mkdir "${new_ace_head_dir}/val"
# mkdir "${new_ace_head_dir}/test"

# ln -s "${no_flip_dir}"/* "${new_ace_head_dir}/train"
# ln -s "${no_flip_dir}"/* "${new_ace_head_dir}/val"
# ln -s "${no_flip_dir}"/* "${new_ace_head_dir}/test"

## We first run sh create_wayspots_testset.sh. Then run this part.
# Notice the ace_head_dummy.pt is a dummy ACE head checkpoint for constructing the ACE Network.
# In preprocessing, we will actually dynamically load the correct ACE heads on-the-fly.

head_network_path="${UPPER_REPO_PATH}/marepo/logs/wayspots_pretrain"

for scene_data in "$new_dir"/*
do
  echo "${scene_data##*/}"
  CUDA_VISIBLE_DEVICES=0 python ../preprocess_marepo.py \
  --dataset_path "${new_dir}/${scene_data##*/}" \
  --dataset_head_network_path ${head_network_path} \
  --head_network_path ../ace_head_dummy.pt \
  --use_half=False \
  --preprocessing True \
  --trainskip 1 \
  --testskip 1 \
  --scheme2 True \
  --scheme2_aug_train_only True

  CUDA_VISIBLE_DEVICES=0 python ../preprocess_marepo.py \
  --dataset_path "${new_dir}/${scene_data##*/}" \
  --dataset_head_network_path ${head_network_path} \
  --head_network_path ../ace_head_dummy.pt \
  --use_half=False \
  --preprocessing True \
  --trainskip 1 \
  --testskip 1 \
  --scheme3 True \
  --scheme3_aug_train_only True
done
