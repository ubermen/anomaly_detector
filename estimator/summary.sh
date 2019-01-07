#!/bin/bash

GAMECODE=$1
COLNAME=$2
DATE=$3
EXEC_TIME="$(date +%s)"

REGION=us-central1
GS_ROOT=gs://bigus/lqad
MODEL_NAME=lqad_ia
TYPE=summary

OUTPUT=$GS_ROOT/results/$MODEL_NAME/$GAMECODE/$COLNAME/$DATE

gsutil cp $OUTPUT/prediction.results* .
cat prediction.results-0000* > total
sed -i -- 's/{"_1": //g' total
sed -i -- 's/"}//g' total
sed -i -- 's/, "_0": "/\t/g' total
sed -i -- 's/, "_0": /\t/g' total
sort -n total > sorted
