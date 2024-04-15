#!/bin/bash

### Find the path to the root of the repo.
SCRIPT_PATH=$(dirname $(realpath -s "$0"))
REPO_PATH=$(realpath -s "${SCRIPT_PATH}/../")

DATASET_PATH_TEST=$(realpath -s "${REPO_PATH}/datasets/wayspots_*")


datatype="test"
testing_exe="${REPO_PATH}/test_marepo.py" # test script
read_log_Marepo="${REPO_PATH}/read_log_marepo.py" # for computing scene average stats

model_name="marepo" # Paper Model
out_dir="${REPO_PATH}/logs/paper_model/${model_name}"

############## benchmark on Marepo Paper Model ###############
for scene in ${DATASET_PATH_TEST}; do
  echo "${scene}" # whole path
  echo "${scene##*/}" # base file name
  ace_head_path="${REPO_PATH}/logs/mapfree/${datatype}/${scene##*/}/${scene##*/}.pt"
  ### For Big Model
  OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python $testing_exe "${scene}" "$out_dir/${model_name}.pt" --head_network_path ${ace_head_path} \
  --transformer_json ../transformer/config/nerf_focal_12T1R_256_homo.json --load_scheme2_sc_map True \
  2>&1 | tee "$out_dir/log_Marepo_${scene##*/}_${datatype}.txt"
done
python $read_log_Marepo "Wayspots" "$out_dir" "${datatype}"

############ benchmark on Finetuned MarepoS ###############
#### TEST
for scene in ${DATASET_PATH_TEST}; do
  echo "${scene}" # whole path
  echo "${scene##*/}" # base file name
  ace_head_path="${REPO_PATH}/logs/mapfree/${datatype}/${scene##*/}/${scene##*/}.pt"
  marepo_head_path="${REPO_PATH}/logs/paper_model/marepo_s_${scene##*/}/marepo_s_${scene##*/}.pt"
  CUDA_VISIBLE_DEVICES=0 python $testing_exe "${scene}" "$marepo_head_path" --head_network_path ${ace_head_path} \
  --transformer_json ../transformer/config/nerf_focal_12T1R_256_homo.json --load_scheme2_sc_map True \
  2>&1 | tee "$out_dir/log_Finetune_Marepo_${scene##*/}_${datatype}.txt"
done
python $read_log_Marepo "Wayspots" "$out_dir" "${datatype}" --finetune True

#### For Marepo 9D model ####
#model_name="marepo_9D" # 9D
#out_dir="${REPO_PATH}/logs/paper_model/${model_name}"
#for scene in ${DATASET_PATH_TEST}; do
#  echo "${scene}" # whole path
#  echo "${scene##*/}" # base file name
#  mkdir -p "$out_dir/${scene##*/}"
#  ace_head_path="${REPO_PATH}/logs/mapfree/${datatype}/${scene##*/}/${scene##*/}.pt"
#  OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=1 python $testing_exe "${scene}" "$out_dir/${model_name}.pt" --head_network_path ${ace_head_path} \
#  --transformer_json ../transformer/config/nerf_focal_12T1R_256_homo_9D.json --load_scheme2_sc_map True \
#  2>&1 | tee "$out_dir/log_Marepo_${scene##*/}_${datatype}.txt"
#done
#python $read_log_Marepo "Wayspots" "$out_dir" "${datatype}"


