#!/bin/bash

export PROJECT_ROOT=/home/web_admin/log-quality

GAMECODE=$1
COLNAME=$2
DATE=$3
TYPE=$4
SAMPLE_SIZE=$5
GEN_MD5=$6

HDFS_ROOT=hdfs://datalake/lqad

SRC_DATASET=${GAMECODE}_log
SRC_TABLE=tb_parsinggamelog_${DATE}
TMP_TABLE=lqad_${GAMECODE}_${COLNAME}_${TYPE}_${DATE}
DST_URI=$HDFS_ROOT/data/$GAMECODE/$COLNAME/$DATE/$TYPE

hdfs dfs -rm -r -skipTrash $DST_URI

python $PROJECT_ROOT/exporter/exporter.py \
--src-dataset $SRC_DATASET \
--src-table $SRC_TABLE \
--tmp-table $TMP_TABLE \
--column $COLNAME \
--dst-uri $DST_URI \
--sample-size $SAMPLE_SIZE \
--gen-md5 $GEN_MD5 \
--src-type hive \
--data-type $TYPE \
