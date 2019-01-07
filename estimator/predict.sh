#!/bin/bash

GAMECODE=$1
COLNAME=$2
DATE=$3
EXEC_TIME="$(date +%s)"

REGION=us-central1
GS_ROOT=gs://bigus/lqad
MODEL_NAME=lqad_ia
TYPE=test

VERSION=_${GAMECODE}_${COLNAME}_${DATE}
JOB_NAME=${MODEL_NAME}_${TYPE}_${GAMECODE}_${COLNAME}_${DATE}_${EXEC_TIME}

INPUT=$GS_ROOT/data/$GAMECODE/$COLNAME/$DATE/$TYPE
OUTPUT=$GS_ROOT/results/$MODEL_NAME/$GAMECODE/$COLNAME/$DATE

gcloud ml-engine jobs submit prediction $JOB_NAME --model $MODEL_NAME --version $VERSION --input-paths $INPUT --output-path $OUTPUT --region $REGION --data-format TEXT
gcloud ml-engine jobs stream-logs $JOB_NAME
