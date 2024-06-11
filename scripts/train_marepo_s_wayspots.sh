#!/bin/bash
### Find the path to the root of the repo.
SCRIPT_PATH=$(dirname $(realpath -s "$0"))
REPO_PATH=$(realpath -s "${SCRIPT_PATH}/../")

model_name="marepo" # Paper Model
out_dir="${REPO_PATH}/logs/${model_name}"
# mkdir "${out_dir}"

########### example to finetuned marepo_s ###############
### TRAIN ###
############################# training the marepo finetune ace head on test set
testset_dir=$(realpath -s "${REPO_PATH}/../wayspots_finetune_dataset")
head_network_path="${REPO_PATH}/logs/wayspots_pretrain"
for scene_data in "$testset_dir"/*
do
  # echo "${scene_data##*/}"
  OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 python ../train_marepo.py \
  "${model_name}_s_${scene_data##*/}_240405" \
  --dataset_path "${testset_dir}/${scene_data##*/}" \
  --dataset_head_network_path ${head_network_path} \
  --transformer_APR_head True \
  --head_network_path "../logs/wayspots_pretrain/test/${scene_data##*/}/${scene_data##*/}.pt" \
  --use_half=False \
  --batch_size 8 \
  --trainskip 1 \
  --testskip 1 \
  --learning_rate 0.00001 \
  --epochs 2 \
  --transformer_json ../transformer/config/nerf_focal_12T1R_256_homo_c2f.json \
  --marepo_sc_augment True \
  --jitter_rot 180.0 \
  --resume_from_pretrain True \
  --pretrain_model_path ../logs/paper_model/marepo/marepo.pt \
  --finetune True \
  2>&1 | tee "tmp/finetune_${scene_data##*/}.txt"
done

############ benchmark on Finetuned MarepoS ###############
#### TEST ###
testing_exe="${REPO_PATH}/test_marepo.py"
read_log_Marepo="${REPO_PATH}/read_log_marepo.py" # for computing scene average stats

DATASET_PATH_TEST=$(realpath -s "${REPO_PATH}/datasets/wayspots_*")
datatype="test"

for scene in ${DATASET_PATH_TEST}; do
  echo "${scene}" # whole path
  echo "${scene##*/}" # base file name
  ace_head_path="${REPO_PATH}/logs/wayspots_pretrain/${datatype}/${scene##*/}/${scene##*/}.pt"
  marepo_head_path="${REPO_PATH}/logs/marepo_s_${scene##*/}_240405/marepo_s_${scene##*/}_240405.pt"
  CUDA_VISIBLE_DEVICES=0 python $testing_exe "${scene}" "$marepo_head_path" --head_network_path ${ace_head_path} \
  --transformer_json ../transformer/config/nerf_focal_12T1R_256_homo.json --load_scheme2_sc_map True \
  2>&1 | tee "$out_dir/log_Finetune_Marepo_${scene##*/}_${datatype}.txt"
done
python $read_log_Marepo "Wayspots" "$out_dir" "${datatype}" --finetune True
