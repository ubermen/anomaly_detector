#!/bin/bash

export PROJECT_ROOT=/home/web_admin/log-quality
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/trainer
export GOOGLE_APPLICATION_CREDENTIALS=$PROJECT_ROOT/credentials/bi-service.json

GAMECODE=$1
COLNAME=$2
DATE=$3
EXEC_TIME="$(date +%s)"

REGION=us-central1
GS_ROOT=gs://bigus/lqad
MODEL_NAME=lqad_ia
TYPE=test

MODEL=$GS_ROOT/models/$MODEL_NAME/$GAMECODE/$COLNAME/$DATE
MODEL_EXPORTER=$MODEL/export/exporter/
MODEL_BINARIES="$(gsutil ls $MODEL_EXPORTER | tail -n 1)"

INPUT=$GS_ROOT/data/$GAMECODE/$COLNAME/$DATE/$TYPE
OUTPUT=$GS_ROOT/results/$MODEL_NAME/$GAMECODE/$COLNAME/$DATE

python3 $PROJECT_ROOT/trainer/gcs_cleaner.py --dir $OUTPUT
zip -r $PROJECT_ROOT/trainer.zip $PROJECT_ROOT/trainer
PYSPARK_PYTHON=./environment/bin/python \
spark-submit \
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./environment/bin/python \
--archives environment.tar.gz#environment \
--master yarn \
--conf spark.cores.max=10 \
--conf spark.task.cpus=1 \
--num-executors 10 \
--py-files $PROJECT_ROOT/trainer.zip,trainer/utils.py,trainer/models.py \
$PROJECT_ROOT/trainer/infer_on_spark.py \
--cluster_size 10 \
--model-dir $MODEL_BINARIES \
--input-dir $INPUT \
--output-dir $OUTPUT \