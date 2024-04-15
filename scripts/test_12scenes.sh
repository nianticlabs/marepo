#!/bin/bash

# Find the path to the root of the repo.
SCRIPT_PATH=$(dirname $(realpath -s "$0"))
REPO_PATH=$(realpath -s "${SCRIPT_PATH}/..")

DATASET_PATH_TEST=$(realpath -s "${REPO_PATH}/datasets/12scenes_*")

datasets_folder="${REPO_PATH}/datasets"
datatype="test" # query image
testing_exe="${REPO_PATH}/test_marepo.py"
read_log_Marepo="${REPO_PATH}/read_log_marepo.py"
model_name="marepo" # paper model

model_file="${REPO_PATH}/logs/paper_model/${model_name}/${model_name}.pt"

out_dir="${REPO_PATH}/logs/paper_model/${model_name}"
ace_head_dir="${REPO_PATH}/logs/pretrain/ace_models/12Scenes"

################ benchmark on Marepo ###############
for scene in ${DATASET_PATH_TEST}; do
  if [ ${scene##*/} = "12scenes_source" ]
  then
    echo "skip 12scenes_source".
  else
    echo "${scene}" # whole path
    echo "${scene##*/}" # base file name
    ace_head_path="${ace_head_dir}/${scene##*/}.pt"
    OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=1 python $testing_exe "${scene}" $model_file --head_network_path ${ace_head_path} \
    --transformer_json ../transformer/config/nerf_focal_12T1R_256_homo.json --load_scheme2_sc_map True 2>&1 | tee "$out_dir/log_Marepo_${scene##*/}_${datatype}.txt"
  fi
done
python $read_log_Marepo "12Scenes" "${out_dir}" "${datatype}"





