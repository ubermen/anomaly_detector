#!/bin/bash
REGION=us-central1
JOB_NAME=$1
OUTPUT_PATH=gs://bigus/$JOB_NAME
MODEL_NAME=$2
VERSION=$3
EXPORT_PATH=$OUTPUT_PATH/export/exporter/
MODEL_BINARIES="$(gsutil ls $EXPORT_PATH | tail -n 1)"
gcloud ml-engine versions create $VERSION --model $MODEL_NAME --origin $MODEL_BINARIES --runtime-version 1.8
