#!/bin/bash

export PROJECT_ROOT=/home/web_admin/log-quality

MODEL_NAME=lqad_ia

ROOT=$1
GAMECODE=$2
COLNAME=$3
DATE=$4
SAMPLE_COUNT=$5
SAMPLE_LENGTH=$6

SRC_DIR=$ROOT/results/$MODEL_NAME/$GAMECODE/$COLNAME/$DATE
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