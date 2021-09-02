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
REINIT=0

P1=1.0
P2=0.5
N1=0.01
N2=1.0

RESULT_DIR=$PROJECT_ROOT/result_tmp
LOGS_DIR=$PROJECT_ROOT/eval_logs

# You can replace this dir with downloaded models.
SAVED_DIR=/home1/zhangwenkai/projects/distant_ner/PUL_bond_conll2003
# $PROJECT_ROOT/models/saved_models_${DATASET}_${LAMB}_${SUF}_${HEAD}

if [ ! -d $RESULT_DIR ]; then
    mkdir -p $RESULT_DIR
fi

if [ ! -d $LOGS_DIR ]; then
    mkdir -p $LOGS_DIR
fi

CUDA_VISIBLE_DEVICES=${DEVICE} python feature_pu_model_subdict.py --reinit ${REINIT} --eval --dataset $DATASET --num_head $HEAD --suf $SUF --save-dir $SAVED_DIR --flag PER --neg --lamb ${LAMB} --reg all --m 0.3 --result-dir $RESULT_DIR --ens $ENS --p1 $P1 --p2 $P2 --n1 $N1 --n2 $N2 \
                                > ${LOGS_DIR}/${SUF}_${DATASET}_${ENS}_PER_${LAMB}_${HEAD}_${DEVICE}.log 

CUDA_VISIBLE_DEVICES=${DEVICE} python feature_pu_model_subdict.py --reinit ${REINIT} --eval --dataset $DATASET --num_head $HEAD --suf $SUF --save-dir $SAVED_DIR --flag ORG --neg --lamb ${LAMB} --reg all --m 0.3 --result-dir $RESULT_DIR --ens $ENS --p1 $P1 --p2 $P2 --n1 $N1 --n2 $N2 \
                                > ${LOGS_DIR}/${SUF}_${DATASET}_${ENS}_ORG_${LAMB}_${HEAD}_${DEVICE}.log 

CUDA_VISIBLE_DEVICES=${DEVICE} python feature_pu_model_subdict.py --reinit ${REINIT} --eval --dataset $DATASET --num_head $HEAD --suf $SUF --save-dir $SAVED_DIR --flag LOC --neg --lamb ${LAMB} --reg all --m 0.3 --result-dir $RESULT_DIR --ens $ENS --p1 $P1 --p2 $P2 --n1 $N1 --n2 $N2 \
                                > ${LOGS_DIR}/${SUF}_${DATASET}_${ENS}_LOC_${LAMB}_${HEAD}_${DEVICE}.log 

CUDA_VISIBLE_DEVICES=${DEVICE} python feature_pu_model_subdict.py --reinit ${REINIT} --eval --dataset $DATASET --num_head $HEAD --suf $SUF --save-dir $SAVED_DIR --flag MISC --neg --lamb ${LAMB} --reg all --m 0.3 --result-dir $RESULT_DIR --ens $ENS --p1 $P1 --p2 $P2 --n1 $N1 --n2 $N2 \
                                > ${LOGS_DIR}/${SUF}_${DATASET}_${ENS}_MISC_${LAMB}_${HEAD}_${DEVICE}.log 

python final_evl_subdict.py --result-dir ${RESULT_DIR} --dataset ${DATASET}