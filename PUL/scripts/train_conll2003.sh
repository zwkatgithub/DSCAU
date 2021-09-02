#!/bin/bash

if [[ $# -ne 1 ]]; then
  DEVICE=0
else
  DEVICE=$1
fi

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..

DATASET=conll2003
ENS=none
LAMB=1.0
SUF=bond
HEAD=5
REINIT=1

P1=1.0
P2=0.5
N1=0.01
N2=1.0

RESULT_DIR=$PROJECT_ROOT/results/result_${DATASET}_${LAMB}_${SUF}_${HEAD}
LOGS_DIR=$PROJECT_ROOT/train_logs
SAVED_DIR=$PROJECT_ROOT/models/saved_models_${DATASET}_${LAMB}_${SUF}_${HEAD}

if [ ! -d $RESULT_DIR ]; then
    mkdir -p $RESULT_DIR
fi

if [ ! -d $SAVED_DIR ]; then
    mkdir -p $SAVED_DIR
fi

if [ ! -d $LOGS_DIR ]; then
    mkdir -p $LOGS_DIR
fi


CUDA_VISIBLE_DEVICES=$DEVICE nohup python feature_pu_model_subdict.py --reinit ${REINIT} --dataset $DATASET --num_head $HEAD --suf $SUF --save-dir $SAVED_DIR --flag PER --neg --lamb ${LAMB} --reg all --m 0.3 --result-dir $RESULT_DIR --ens $ENS --p1 $P1 --p2 $P2 --n1 $N1 --n2 $N2 \
                                > ${LOGS_DIR}/${SUF}_${DATASET}_${ENS}_PER_${LAMB}_${HEAD}.log &

CUDA_VISIBLE_DEVICES=$DEVICE nohup python feature_pu_model_subdict.py --reinit ${REINIT} --dataset $DATASET --num_head $HEAD --suf $SUF --save-dir $SAVED_DIR --flag ORG --neg --lamb ${LAMB} --reg all --m 0.3 --result-dir $RESULT_DIR --ens $ENS --p1 $P1 --p2 $P2 --n1 $N1 --n2 $N2 \
                                > ${LOGS_DIR}/${SUF}_${DATASET}_${ENS}_ORG_${LAMB}_${HEAD}.log &

CUDA_VISIBLE_DEVICES=$DEVICE nohup python feature_pu_model_subdict.py --reinit ${REINIT} --dataset $DATASET --num_head $HEAD --suf $SUF --save-dir $SAVED_DIR --flag LOC --neg --lamb ${LAMB} --reg all --m 0.3 --result-dir $RESULT_DIR --ens $ENS --p1 $P1 --p2 $P2 --n1 $N1 --n2 $N2 \
                                > ${LOGS_DIR}/${SUF}_${DATASET}_${ENS}_LOC_${LAMB}_${HEAD}.log &

CUDA_VISIBLE_DEVICES=$DEVICE nohup python feature_pu_model_subdict.py --reinit ${REINIT} --dataset $DATASET --num_head $HEAD --suf $SUF --save-dir $SAVED_DIR --flag MISC --neg --lamb ${LAMB} --reg all --m 0.3 --result-dir $RESULT_DIR --ens $ENS --p1 $P1 --p2 $P2 --n1 $N1 --n2 $N2 \
                                > ${LOGS_DIR}/${SUF}_${DATASET}_${ENS}_MISC_${LAMB}_${HEAD}.log &