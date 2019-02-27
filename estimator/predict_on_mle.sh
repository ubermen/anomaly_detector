#!/bin/bash

export PROJECT_ROOT=/home/web_admin/log-quality
export GOOGLE_APPLICATION_CREDENTIALS=$PROJECT_ROOT/credentials/bi-service.json

GAMECODE=$1
COLNAME=$2
DATE=$3
EXEC_TIME="$(date +%s)"
MODEL_DATE=$(date -d "$DATE 1 days ago" +%Y%m%d)

REGION=us-central1
GS_ROOT=gs://bigus/lqad
MODEL_NAME=lqad_ia
TYPE=test

VERSION=_${GAMECODE}_${COLNAME}_${MODEL_DATE}
VERSION_LATEST="$(gcloud ml-engine versions list --model=lqad_ia | grep $VERSION | tail -n 1 | cut -d' ' -f1)"
JOB_NAME=${MODEL_NAME}_${TYPE}_${GAMECODE}_${COLNAME}_${DATE}_${EXEC_TIME}

INPUT=$GS_ROOT/data/$GAMECODE/$COLNAME/$DATE/$TYPE/*
OUTPUT=$GS_ROOT/results/$MODEL_NAME/$GAMECODE/$COLNAME/$DATE

python3 $PROJECT_ROOT/trainer/gcs_cleaner.py --dir $OUTPUT

gcloud ml-engine jobs submit prediction $JOB_NAME --model $MODEL_NAME --version $VERSION_LATEST --input-paths $INPUT --output-path $OUTPUT --region $REGION --data-format TEXT
gcloud ml-engine jobs stream-logs $JOB_NAME
