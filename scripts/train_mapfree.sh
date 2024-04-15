#!/usr/bin/env bash

# Find the path to the root of the repo.
SCRIPT_PATH=$(dirname $(realpath -s "$0"))
REPO_PATH=$(realpath -s "${SCRIPT_PATH}/../")

DATASET_PATH_TRAIN=$(realpath -s "${SCRIPT_PATH}/../../map_free_training_scenes_aug_16/train/*")
DATASET_PATH_VAL=$(realpath -s "${SCRIPT_PATH}/../../map_free_training_scenes_aug_16/val/*")
DATASET_PATH_TEST=$(realpath -s "${SCRIPT_PATH}/../../map_free_training_scenes_aug_16/test/*")

training_exe="${REPO_PATH}/train_ace.py"
testing_exe="${REPO_PATH}/test_ace.py"

################################################ Train Train ACE Heads #######################################
out_dir="${REPO_PATH}/logs/mapfree/train"
mkdir -p "$out_dir"

for scene in ${DATASET_PATH_TRAIN}; do
  echo "${scene}" # whole path
  echo "${scene##*/}" # base file name
  python $training_exe "${scene}" "$out_dir/${scene##*/}/${scene##*/}.pt"
  python $testing_exe "${scene}" "$out_dir/${scene##*/}/${scene##*/}.pt" 2>&1 | tee "$out_dir/${scene##*/}/log_${scene##*/}.txt"
done

for scene in ${DATASET_PATH_TRAIN}; do
  echo "${scene##*/}: $(cat "$out_dir/${scene##*/}/log_${scene##*/}.txt" | taiql -6 | head -1)"
done

################################################ Train Val ACE Heads #######################################
out_dir="${REPO_PATH}/logs/mapfree/val"
mkdir -p "$out_dir"

for scene in ${DATASET_PATH_VAL}; do
  echo "${scene}" # whole path
  echo "${scene##*/}" # base file name
  python $training_exe "${scene}" "$out_dir/${scene##*/}/${scene##*/}.pt"
  python $testing_exe "${scene}" "$out_dir/${scene##*/}/${scene##*/}.pt" 2>&1 | tee "$out_dir/${scene##*/}/log_${scene##*/}.txt"
done

for scene in ${DATASET_PATH_VAL}; do
  echo "${scene##*/}: $(cat "$out_dir/${scene##*/}/log_${scene##*/}.txt" | tail -6 | head -1)"
done

############################################### Train Test ACE Heads #######################################
out_dir="${REPO_PATH}/logs/mapfree/test"
mkdir -p "$out_dir"

for scene in ${DATASET_PATH_TEST}; do
  echo "${scene}" # whole path
  echo "${scene##*/}" # base file name
  python $training_exe "${scene}" "$out_dir/${scene##*/}/${scene##*/}.pt"
  python $testing_exe "${scene}" "$out_dir/${scene##*/}/${scene##*/}.pt" 2>&1 | tee "$out_dir/${scene##*/}/log_${scene##*/}.txt"
done

for scene in ${DATASET_PATH_TEST}; do
  echo "${scene##*/}: $(cat "$out_dir/${scene##*/}/log_${scene##*/}.txt" | tail -6 | head -1)"
done









###################################################################### Script for training additional ACE heads on flipped dataset ###################################################
# Find the path to the root of the repo.
SCRIPT_PATH=$(dirname $(realpath -s "$0"))
REPO_PATH=$(realpath -s "${SCRIPT_PATH}/../")

DATASET_PATH_TRAIN=$(realpath -s "${SCRIPT_PATH}/../../map_free_training_scenes_aug_16_flip/train/*")
DATASET_PATH_VAL=$(realpath -s "${SCRIPT_PATH}/../../map_free_training_scenes_aug_16_flip/val/*")

training_exe="${REPO_PATH}/train_ace.py"
testing_exe="${REPO_PATH}/test_ace.py"

################################################ Train Train ACE Heads #######################################
out_dir="${REPO_PATH}/logs/mapfree_flip/train"
mkdir -p "$out_dir"

for scene in ${DATASET_PATH_TRAIN}; do
  echo "${scene}" # whole path
  echo "${scene##*/}" # base file name
  OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python $training_exe "${scene}" "$out_dir/${scene##*/}/${scene##*/}.pt"
  OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python $testing_exe "${scene}" "$out_dir/${scene##*/}/${scene##*/}.pt" 2>&1 | tee "$out_dir/${scene##*/}/log_${scene##*/}.txt"
done
#
for scene in ${DATASET_PATH_TRAIN}; do
  echo "${scene##*/}: $(cat "$out_dir/${scene##*/}/log_${scene##*/}.txt" | taiql -6 | head -1)"
done

################################################ Train Val ACE Heads #######################################
out_dir="${REPO_PATH}/logs/mapfree_flip/val"
mkdir -p "$out_dir"

for scene in ${DATASET_PATH_VAL}; do
  echo "${scene}" # whole path
  echo "${scene##*/}" # base file name
  OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=1 python $training_exe "${scene}" "$out_dir/${scene##*/}/${scene##*/}.pt"
  OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=1 python $testing_exe "${scene}" "$out_dir/${scene##*/}/${scene##*/}.pt" 2>&1 | tee "$out_dir/${scene##*/}/log_${scene##*/}.txt"
done

for scene in ${DATASET_PATH_VAL}; do
  echo "${scene##*/}: $(cat "$out_dir/${scene##*/}/log_${scene##*/}.txt" | tail -6 | head -1)"
done
