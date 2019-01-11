#!/bin/bash

GAMECODE=$1
COLNAME=$2
DATE=$3
TYPE=$4

GS_ROOT=gs://bigus/lqad

SRC_DATASET=bigpi_${GAMECODE}
SRC_TABLE=ParsingGameLog_${DATE}
TMP_DATASET=bigpi_test
TMP_TABLE=lqad_${GAMECDE}_${COLNAME}_${DATE}
DST_URI=$GS_ROOT/data/$GAMECODE/$COLNAME/$DATE/$TYPE

python exporter.py \
--src-dataset $SRC_DATASET \
--src-table $SRC_TABLE \
--tmp-dataset $TMP_DATASET \
--tmp-table $TMP_TABLE \
--column $COLNAME \
--dst-uri $DST_URI \