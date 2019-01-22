#!/bin/bash

export PROJECT_ROOT=/home/web_admin/log-quality
export GOOGLE_APPLICATION_CREDENTIALS=$PROJECT_ROOT/credentials/bi-service.json

GAMECODE=$1
COLNAME=$2
DATE=$3
MODULE=$4
EXEC_TIME="$(date +%s)"

REGION=us-central1
GS_ROOT=gs://bigus/lqad
MODEL_NAME=lqad_ia
TYPE=train

JOB_NAME=${MODEL_NAME}_${TYPE}_${GAMECODE}_${COLNAME}_${DATE}_${EXEC_TIME}

DATA=$GS_ROOT/data/$GAMECODE/$COLNAME/$DATE
MODEL=$GS_ROOT/models/$MODEL_NAME/$GAMECODE/$COLNAME/$DATE

python3 $PROJECT_ROOT/trainer/gcs_cleaner.py --dir $MODEL
gcloud ml-engine jobs submit training $JOB_NAME --job-dir $MODEL --runtime-version 1.8 --config $PROJECT_ROOT/config.yaml --module-name trainer.$MODULE --package-path $PROJECT_ROOT/trainer/ --region $REGION -- --data-dir $DATA
gcloud ml-engine jobs stream-logs $JOB_NAME
