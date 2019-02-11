#!/bin/bash

export PROJECT_ROOT=/home/web_admin/log-quality
export GOOGLE_APPLICATION_CREDENTIALS=$PROJECT_ROOT/credentials/bi-service.json

GAMECODE=$1
COLNAME=$2
DATE=$3
TYPE=$4
SAMPLE_SIZE=$5
GEN_MD5=$6

GS_ROOT=gs://bigus/lqad

SRC_DATASET=bigpi_${GAMECODE}
SRC_TABLE=ParsingGameLog_${DATE}
TMP_DATASET=bigpi_test
TMP_TABLE=lqad_${GAMECODE}_${COLNAME}_${DATE}
DST_URI=$GS_ROOT/data/$GAMECODE/$COLNAME/$DATE/$TYPE

python3 $PROJECT_ROOT/exporter/exporter.py \
--src-dataset $SRC_DATASET \
--src-table $SRC_TABLE \
--tmp-dataset $TMP_DATASET \
--tmp-table $TMP_TABLE \
--column $COLNAME \
--dst-uri $DST_URI \
--sample-size $SAMPLE_SIZE \
--gen-md5 $GEN_MD5 \
--src-type bigquery \