REPO=$PWD
MODEL=${1:-bert-base-uncased}
GPU=${2:-0}
DATA_DIR=${3:-"/data/wjk/PABEE-glue/data/sst-2"}
OUT_DIR=${4:-"$REPO/outputs_stage1/"}
SEED=${5:-42}
STAGE2_DIR=${6:-"$REPO/outputs_stage2/"}
TASK='sst-2'
STAGE1_LR=2e-5
STAGE2_LR=8e-3
STAGE1_OUT_DIR="${OUT_DIR}${TASK}/LR${STAGE1_LR}_SEED${SEED}/"
STAGE2_OUT_DIR="${STAGE2_DIR}${TASK}/LR${STAGE2_LR}_SEED${SEED}/"

export CUDA_VISIBLE_DEVICES=$GPU


mkdir -p $STAGE1_OUT_DIR
python ./run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --task_name $TASK \
  --train_stage 1 \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $DATA_DIR \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --learning_rate $STAGE1_LR \
  --save_steps 100 \
  --num_train_epochs 5 \
  --output_dir $STAGE1_OUT_DIR \
  --overwrite_output_dir \
  --overwrite_cache \
  --evaluate_during_training \
  --seed $SEED \
  --patience 0

mkdir -p $STAGE2_OUT_DIR
python ./run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --task_name $TASK \
  --train_stage 2 \
  --stage1_dir $STAGE1_OUT_DIR\
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $DATA_DIR \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 1 \
  --learning_rate $STAGE2_LR \
  --save_steps 10 \
  --limit 9000 \
  --num_train_epochs 1 \
  --output_dir $STAGE2_OUT_DIR \
  --overwrite_output_dir \
  --overwrite_cache \
  --evaluate_during_training \
  --seed $SEED \
  --patience 6