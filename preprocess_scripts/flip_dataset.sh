#!/bin/bash

SCRIPT_PATH=$(dirname $(realpath -s "$0"))
UPPER_REPO_PATH=$(realpath -s "${SCRIPT_PATH}/../..") # the path to the upper directory of the root of the marepo repo

############### preprocess mapfree data to flip mapping and query data
train_dir="${UPPER_REPO_PATH}/map_free_training_scenes_aug_16_flip/train"
cd ${train_dir}
for scene_data in *
do
  echo "${scene_data}"
  mv "${scene_data}/train" "${scene_data}/train_tmp"
  mv "${scene_data}/test" "${scene_data}/train"
done

# just to be safe therefore we seperate this step from the upper ones
for scene_data in *
do
  echo "${scene_data}"
  mv "${scene_data}/train_tmp" "${scene_data}/test"
done

for scene_data in *
do
  echo "${scene_data}"
  mv "${scene_data}" "${scene_data}_flip"
done


val_dir="${UPPER_REPO_PATH}/map_free_training_scenes_aug_16_flip/val"
cd ${val_dir}
for scene_data in *
do
  echo "${scene_data}"
  mv "${scene_data}/train" "${scene_data}/train_tmp"
  mv "${scene_data}/test" "${scene_data}/train"
done

# just to be safe therefore we seperate this step from the upper ones
for scene_data in *
do
  echo "${scene_data}"
  mv "${scene_data}/train_tmp" "${scene_data}/test"
done

for scene_data in *
do
  echo "${scene_data}"
  mv "${scene_data}" "${scene_data}_flip"
done