#!/bin/bash

export PROJECT_ROOT=/home/web_admin/log-quality

MODULE=$1
CMD=$2
ROOT=$3
GAMECODE=$4
COLNAME=$5
DATE=$6

SRC_DIR_BEF_DATE=$ROOT/data/$GAMECODE/$COLNAME

cd $PROJECT_ROOT &&
zip -r reporter.zip reporter/*.py
spark-submit \
--master yarn \
--conf spark.executor.cores=20 \
--conf spark.executor.memory=1g \
--num-executors 5 \
--jars \
$PROJECT_ROOT/reporter/jars/gcs-connector-latest-hadoop2.jar,\
$PROJECT_ROOT/reporter/jars/mysql-connector-java-5.1.40.jar \
$PROJECT_ROOT/reporter/$MODULE.py \
--src-dir $SRC_DIR_BEF_DATE \
--gamecode $GAMECODE \
--column $COLNAME \
--yyyymmdd $DATE \
--cmd $CMD