REPO=$PWD
MODEL=${1:-albert-base-v2}
GPU=${2:-0}
DATA_DIR=${3:-"/data/wjk/PABEE-glue/data/mrpc"}
OUT_DIR=${4:-"$REPO/albert_outputs_stage1/"}
SEED=${5:-42}
STAGE2_DIR=${6:-"$REPO/albert_outputs_stage2/"}
TASK='mrpc'
STAGE1_LR=3e-5
STAGE2_LR=1e-3
export CUDA_VISIBLE_DEVICES=$GPU
STAGE1_OUT_DIR="${OUT_DIR}${TASK}/LR${STAGE1_LR}_SEED${SEED}/"
STAGE2_OUT_DIR="${STAGE2_DIR}${TASK}/LR${STAGE2_LR}_SEED${SEED}/"


mkdir -p $STAGE1_OUT_DIR
python ./run_glue.py \
  --model_type albert \
  --model_name_or_path albert-base-v2 \
  --task_name $TASK \
  --train_stage 1 \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $DATA_DIR  \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --learning_rate $STAGE1_LR \
  --save_steps 100 \
  --num_train_epochs 20 \
  --output_dir $STAGE1_OUT_DIR \
  --overwrite_output_dir \
  --overwrite_cache \
  --seed $SEED \
  --evaluate_during_training \
  --patience 0

mkdir -p $STAGE2_OUT_DIR
python ./run_glue.py \
  --model_type albert \
  --model_name_or_path albert-base-v2 \
  --task_name $TASK \
  --train_stage 2 \
  --stage1_dir $STAGE1_OUT_DIR \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $DATA_DIR  \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 1 \
  --learning_rate $STAGE2_LR \
  --save_steps 10 \
  --num_train_epochs 5 \
  --limit 8000 \
  --output_dir $STAGE2_OUT_DIR \
  --overwrite_output_dir \
  --overwrite_cache \
  --seed $SEED \
  --evaluate_during_training \
  --patience 6
