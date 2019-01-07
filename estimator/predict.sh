#!/bin/bash
REGION=us-central1
JOB_NAME=$1
MODEL_NAME=$2
VERSION=$3
INPUT_PATH=$4
OUTPUT_PATH=$5
gcloud ml-engine jobs submit prediction $JOB_NAME --model $MODEL_NAME --version $VERSION --input-paths $INPUT_PATH --output-path $OUTPUT_PATH --region $REGION --data-format TEXT
gcloud ml-engine jobs stream-logs $JOB_NAME
