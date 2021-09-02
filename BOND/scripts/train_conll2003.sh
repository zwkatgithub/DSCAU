#!/bin/bash

if [[ $# -ne 1 ]]; then
  GPUID=0
else
  GPUID=$1
fi

echo "Run on GPU $GPUID"

# data
PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..
DATA_ROOT=$PROJECT_ROOT/../dataset/conll03/

# model
MODEL_TYPE=roberta
MODEL_NAME=roberta-base

# params
LR=1e-5
WEIGHT_DECAY=1e-4
EPOCH=50
SEED=0

ADAM_EPS=1e-8
ADAM_BETA1=0.9
ADAM_BETA2=0.98
WARMUP=200

TRAIN_BATCH=16
EVAL_BATCH=32

# self-training parameters
REINIT=0
BEGIN_STEP=900
LABEL_MODE=soft
PERIOD=1756
HP_LABEL=5.9

LAMB=0.001

# output
SAVED_DIR=$PROJECT_ROOT/models/conll03/self_training/${MODEL_TYPE}_reinit${REINIT}_begin${BEGIN_STEP}_period_${PERIOD}_${LABEL_MODE}_hp${HP_LABEL}_${EPOCH}_${LR}/
LOGS_DIR=$PROJECT_ROOT/train_logs

[ -e $SAVED_DIR/script  ] || mkdir -p $SAVED_DIR/script
cp -f $(readlink -f "$0") $SAVED_DIR/script

if [ ! -d $LOGS_DIR ]; then
    mkdir -p $LOGS_DIR
fi

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUID nohup python3 run_self_training_ner_subdict.py --data_dir $DATA_ROOT \
  --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME \
  --learning_rate $LR \
  --weight_decay $WEIGHT_DECAY \
  --adam_epsilon $ADAM_EPS \
  --adam_beta1 $ADAM_BETA1 \
  --adam_beta2 $ADAM_BETA2 \
  --num_train_epochs $EPOCH \
  --warmup_steps $WARMUP \
  --per_gpu_train_batch_size $TRAIN_BATCH \
  --per_gpu_eval_batch_size $EVAL_BATCH \
  --logging_steps 100 \
  --save_steps 100000 \
  --do_train \
  --do_eval \
  --do_predict \
  --evaluate_during_training \
  --saved_dir $SAVED_DIR \
  --cache_dir $PROJECT_ROOT/pretrained_model \
  --seed $SEED \
  --max_seq_length 128 \
  --overwrite_saved_dir \
  --self_training_reinit $REINIT --self_training_begin_step $BEGIN_STEP \
  --self_training_label_mode $LABEL_MODE --self_training_period $PERIOD \
  --self_training_hp_label $HP_LABEL \
  --lamb $LAMB --num 0 > $LOGS_DIR/conll2003.log &
