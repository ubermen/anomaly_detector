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

rm -rf summary
mkdir summary
gsutil cp $OUTPUT/prediction.results* ./summary/
cat ./summary/prediction.results-0000* > ./summary/total
sed -i -- 's/{"_1": //g' ./summary/total
sed -i -- 's/"}//g' ./summary/total
sed -i -- 's/, "_0": "/\t/g' ./summary/total
sed -i -- 's/, "_0": /\t/g' ./summary/total
sort -n ./summary/total > ./summary/sorted
