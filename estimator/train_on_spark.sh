#!/bin/bash

export PROJECT_ROOT=/home/web_admin/log-quality

GAMECODE=$1
COLNAME=$2
DATE=$3
MODULE=$4
EXEC_ALLOCATION=$5
EXEC_TIME="$(date +%s)"

HDFS_ROOT=hdfs://datalake/lqad
MODEL_NAME=lqad_ia
TYPE=train

JOB_NAME=${MODEL_NAME}_${TYPE}_${GAMECODE}_${COLNAME}_${DATE}_${EXEC_TIME}

DATA=$HDFS_ROOT/data/$GAMECODE/$COLNAME/$DATE
MODEL=$HDFS_ROOT/models/$MODEL_NAME/$GAMECODE/$COLNAME/$DATE

HADOOP_HDFS_HOME=/usr/hdp/2.6.4.0-91
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$JAVA_HOME/jre/lib/amd64/server
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/ams-hbase/lib/hadoop-native
CLASSPATH=$($HADOOP_HDFS_HOME/hadoop/bin/hadoop classpath --glob)

hdfs dfs -rm -r -skipTrash $MODEL

cd $PROJECT_ROOT &&
rm trainer/*.pyc
zip -r trainer.zip trainer
PYSPARK_PYTHON=./environment/bin/python \
spark-submit \
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./environment/bin/python \
--archives environment.tar.gz#environment \
--master yarn \
--conf spark.cores.max=$EXEC_ALLOCATION \
--conf spark.task.cpus=1 \
--conf spark.executor.memory=10g \
--conf spark.executorEnv.HADOOP_HDFS_HOME=$HADOOP_HDFS_HOME \
--conf spark.executorEnv.LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
--conf spark.executor.extraClassPath=$CLASSPATH \
--num-executors $EXEC_ALLOCATION \
--py-files trainer.zip \
$PROJECT_ROOT/trainer/$MODULE.py \
--cluster_size $EXEC_ALLOCATION \
--job-dir $MODEL \
--data-dir $DATA \
--engine spark \
--app-name $JOB_NAME \