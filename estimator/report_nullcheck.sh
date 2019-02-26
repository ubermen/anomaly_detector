#!/bin/bash

export PROJECT_ROOT=/home/web_admin/log-quality

CMD=$1
ROOT=$2
GAMECODE=$3
COLNAME=$4
DATE=$5

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
$PROJECT_ROOT/reporter/null_checker.py \
--src-dir $SRC_DIR_BEF_DATE \
--gamecode $GAMECODE \
--column $COLNAME \
--yyyymmdd $DATE \
--cmd $CMD