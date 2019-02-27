#!/bin/bash

export PROJECT_ROOT=/home/web_admin/log-quality

GAMECODE=$1
COLNAME=$2
DATE=$3
EXEC_ALLOCATION=$4
EXEC_TIME="$(date +%s)"
MODEL_DATE=$(date -d "$DATE 1 days ago" +%Y%m%d)

HDFS_ROOT=hdfs://datalake/lqad
MODEL_NAME=lqad_ia
TYPE=test

JOB_NAME=${MODEL_NAME}_${TYPE}_${GAMECODE}_${COLNAME}_${DATE}_${EXEC_TIME}

MODEL=$HDFS_ROOT/models/$MODEL_NAME/$GAMECODE/$COLNAME/$MODEL_DATE
MODEL_EXPORTER=$MODEL/export/exporter/
MODEL_BINARIES="$(hdfs dfs -ls -C $MODEL_EXPORTER | tail -n 1)"

INPUT=$HDFS_ROOT/data/$GAMECODE/$COLNAME/$DATE/$TYPE/*
OUTPUT=$HDFS_ROOT/results/$MODEL_NAME/$GAMECODE/$COLNAME/$DATE

HADOOP_HDFS_HOME=/usr/hdp/2.6.4.0-91
CLASSPATH=$($HADOOP_HDFS_HOME/hadoop/bin/hadoop classpath --glob)

# static setting of LD_LIBRARY_PATH for remote call
LD_LIBRARY_PATH=/lib:/usr/lib:/usr/local/lib:/usr/lib/oracle/11.2/client/lib:.:/usr/local/java/jre/lib/amd64/server:/usr/lib/ams-hbase/lib/hadoop-native
#LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$JAVA_HOME/jre/lib/amd64/server
#LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/ams-hbase/lib/hadoop-native

hdfs dfs -rm -r -skipTrash $OUTPUT

cd $PROJECT_ROOT &&
rm trainer/*.pyc
zip -r trainer.zip trainer
PYSPARK_PYTHON=./environment/bin/python \
spark-submit \
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./environment/bin/python \
--archives environment.tar.gz#environment \
--master yarn \
--num-executors $EXEC_ALLOCATION \
--executor-memory 10G \
--conf spark.executorEnv.HADOOP_HDFS_HOME=$HADOOP_HDFS_HOME \
--conf spark.executorEnv.LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
--conf spark.executor.extraClassPath=$CLASSPATH \
--py-files trainer.zip \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.maxAppAttempts=1 \
--conf spark.yarn.executor.memoryOverhead=4G \
$PROJECT_ROOT/trainer/infer_on_spark.py \
--cluster-size $EXEC_ALLOCATION \
--model-dir $MODEL_BINARIES \
--input-dir $INPUT \
--output-dir $OUTPUT \
--app-name $JOB_NAME \