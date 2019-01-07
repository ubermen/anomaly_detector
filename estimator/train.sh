#!/bin/bash

GAMECODE=$1
COLNAME=$2
DATE=$3
EXEC_TIME="$(date +%s)"

REGION=us-central1
GS_ROOT=gs://bigus/lqad
MODEL_NAME=lqad_ia
TYPE=train

JOB_NAME=${MODEL_NAME}_${TYPE}_${GAMECODE}_${COLNAME}_${DATE}_${EXEC_TIME}

DATA=$GS_ROOT/data/$GAMECODE/$COLNAME/$DATE
MODEL=$GS_ROOT/models/$MODEL_NAME/$GAMECODE/$COLNAME/$DATE

gcloud ml-engine jobs submit training $JOB_NAME --job-dir $MODEL --runtime-version 1.8 --config config.yaml --module-name trainer.simple_vae_on_estimator --package-path trainer/ --region $REGION -- --data-dir $DATA
gcloud ml-engine jobs stream-logs $JOB_NAME
