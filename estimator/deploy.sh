#!/bin/bash

GAMECODE=$1
COLNAME=$2
DATE=$3
EXEC_TIME="$(date +%s)"

REGION=us-central1
GS_ROOT=gs://bigus/lqad
MODEL_NAME=lqad_ia
TYPE=deploy

VERSION=_${GAMECODE}_${COLNAME}_${DATE}_${EXEC_TIME}

MODEL=$GS_ROOT/models/$MODEL_NAME/$GAMECODE/$COLNAME/$DATE
MODEL_EXPORTER=$MODEL/export/exporter/
MODEL_BINARIES="$(gsutil ls $MODEL_EXPORTER | tail -n 1)"

echo $MODEL_EXPORTER
echo gcloud ml-engine versions create $VERSION --model $MODEL_NAME --origin $MODEL_BINARIES --runtime-version 1.8

gcloud ml-engine versions create $VERSION --model $MODEL_NAME --origin $MODEL_BINARIES --runtime-version 1.8
