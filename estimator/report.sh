#!/bin/bash

export PROJECT_ROOT=/home/web_admin/log-quality

GS_ROOT=gs://bigus/lqad
MODEL_NAME=lqad_ia

GAMECODE=$1
COLNAME=$2
DATE=$3
SAMPLE_COUNT=$4
SAMPLE_LENGTH=$5

SRC_DIR=$GS_ROOT/results/$MODEL_NAME/$GAMECODE/$COLNAME/$DATE
DST_TABLE=inherent_anomaly_summary

spark-submit \
--jars \
$PROJECT_ROOT/reporter/jars/gcs-connector-latest-hadoop2.jar,\
$PROJECT_ROOT/reporter/jars/mysql-connector-java-5.1.40.jar \
$PROJECT_ROOT/reporter/reporter.py \
--src-dir $SRC_DIR \
--dst-table $DST_TABLE \
--gamecode $GAMECODE \
--column $COLNAME \
--yyyymmdd $DATE \
--sample_count $SAMPLE_COUNT \
--sample_length $SAMPLE_LENGTH