#!/bin/bash

# In this script, we copy the 7scenes test sets to the new folder
# so that we could generate scheme3 data augmentation for the mapping sequences, and finetune our baseline models

SCRIPT_PATH=$(dirname $(realpath -s "$0"))
UPPER_REPO_PATH=$(realpath -s "${SCRIPT_PATH}/../..") # the path to the upper directory of the root of the marepo repo

############################## Build finetune dataset like mapfree structure ##########################
new_dir="${UPPER_REPO_PATH}/7scenes_finetune_dataset"
testset_path="${UPPER_REPO_PATH}/marepo/datasets"


mkdir ${new_dir}
cd ${testset_path}

for scene in "7scenes_"*
do
  echo "${scene}"
  if [ ${scene} = "7scenes_source" ]
  then
    echo "skip 7scenes_source".
  else
    mkdir "${new_dir}/${scene}"
    mkdir "${new_dir}/${scene}/train"
    mkdir "${new_dir}/${scene}/val"
    mkdir "${new_dir}/${scene}/test"
    cp -d -rL "$scene" "${new_dir}/${scene}/train" # copy the file to new directory
    ln -s "${new_dir}/${scene}/train/${scene}" "${new_dir}/${scene}/val" # Note: this could be softlinks to save disk space
    ln -s "${new_dir}/${scene}/train/${scene}" "${new_dir}/${scene}/test" # Note: this could be softlinks to save disk space
  fi
done

############################## build a new pretrained checkpoint directory like mapfree dataset structure ##########################
pretrain_ckpt_dir="${UPPER_REPO_PATH}/marepo/logs/pretrain/ace_models/7Scenes"
new_ckpt_dir="${UPPER_REPO_PATH}/marepo/logs/pretrain/ace_models/7Scenes_pretrain"
testset_path="${UPPER_REPO_PATH}/marepo/datasets"

mkdir ${new_ckpt_dir}
mkdir "${new_ckpt_dir}/train/"
mkdir "${new_ckpt_dir}/val/"
mkdir "${new_ckpt_dir}/test/"

cd ${testset_path}

for scene in "7scenes_"*
do
  echo "${scene}"
  if [ ${scene} = "7scenes_source" ]
  then
    echo "skip 7scenes_source".
  else

    mkdir "${new_ckpt_dir}/train/${scene}"
    mkdir "${new_ckpt_dir}/val/${scene}"
    mkdir "${new_ckpt_dir}/test/${scene}"
    ln -s "${pretrain_ckpt_dir}/${scene}.pt" "${new_ckpt_dir}/train/${scene}" # copy the softlink pointed file to new directory
    ln -s "${pretrain_ckpt_dir}/${scene}.pt" "${new_ckpt_dir}/val/${scene}"
    ln -s "${pretrain_ckpt_dir}/${scene}.pt" "${new_ckpt_dir}/test/${scene}"
  fi
done

#########################################
# generate augmented data for the 7Scene mapping sequences,
# The augmentation helps 7Scenes adaptation for ACEFormer and generates necessary files like in mapfree training
# Notice the ace_head_dummy.pt is a dummy ACE head checkpoint for constructing the ACE Network.
# In preprocessing, we will actually dynamically load the correct ACE heads on-the-fly.
########################################

testset_dir="${UPPER_REPO_PATH}/7scenes_finetune_dataset"
head_network_path="${UPPER_REPO_PATH}/marepo/logs/pretrain/ace_models/7Scenes_pretrain" # same as ${new_ckpt_dir}

cd "${UPPER_REPO_PATH}/marepo/preprocess_scripts"
for scene_data in "$testset_dir"/*
do
  echo "${scene_data##*/}"
  # perform scheme2 data augmentation for 7Scenes
  CUDA_VISIBLE_DEVICES=0 python ../preprocess_marepo.py \
  --dataset_path "${testset_dir}/${scene_data##*/}" \
  --dataset_head_network_path ${head_network_path} \
  --head_network_path ../ace_head_dummy.pt \
  --use_half=False \
  --preprocessing True \
  --trainskip 1 \
  --testskip 1 \
  --scheme2 True \
  --scheme2_aug_train_only True \
  --not_mapfree True

  # perform scheme3 data augmentation for 7Scenes
  CUDA_VISIBLE_DEVICES=0 python ../preprocess_marepo.py \
  --dataset_path "${testset_dir}/${scene_data##*/}" \
  --dataset_head_network_path ${head_network_path} \
  --head_network_path ../ace_head_dummy.pt \
  --use_half=False \
  --preprocessing True \
  --trainskip 1 \
  --testskip 1 \
  --scheme3 True \
  --scheme3_aug_number 0 4 \
  --scheme3_aug_train_only True \
  --not_mapfree True

  # perform scheme3 data augmentation for 7Scenes
  CUDA_VISIBLE_DEVICES=0 python ../preprocess_marepo.py \
  --dataset_path "${testset_dir}/${scene_data##*/}" \
  --dataset_head_network_path ${head_network_path} \
  --head_network_path ../ace_head_dummy.pt \
  --use_half=False \
  --preprocessing True \
  --trainskip 1 \
  --testskip 1 \
  --scheme3 True \
  --scheme3_aug_number 4 8 \
  --scheme3_aug_train_only True \
  --not_mapfree True

#  # perform scheme3 data augmentation for 7Scenes
  CUDA_VISIBLE_DEVICES=0 python ../preprocess_marepo.py \
  --dataset_path "${testset_dir}/${scene_data##*/}" \
  --dataset_head_network_path ${head_network_path} \
  --head_network_path ../ace_head_dummy.pt \
  --use_half=False \
  --preprocessing True \
  --trainskip 1 \
  --testskip 1 \
  --scheme3 True \
  --scheme3_aug_number 8 12 \
  --scheme3_aug_train_only True \
  --not_mapfree True

  # perform scheme3 data augmentation for 7Scenes
  CUDA_VISIBLE_DEVICES=0 python ../preprocess_marepo.py \
  --dataset_path "${testset_dir}/${scene_data##*/}" \
  --dataset_head_network_path ${head_network_path} \
  --head_network_path ../ace_head_dummy.pt \
  --use_half=False \
  --preprocessing True \
  --trainskip 1 \
  --testskip 1 \
  --scheme3 True \
  --scheme3_aug_number 12 16 \
  --scheme3_aug_train_only True \
  --not_mapfree True

done
