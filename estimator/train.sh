#!/bin/bash
REGION=us-central1
JOB_NAME=$1
OUTPUT_PATH=gs://bigus/$JOB_NAME
gcloud ml-engine jobs submit training $JOB_NAME --job-dir $OUTPUT_PATH --runtime-version 1.8 --config config.yaml --module-name trainer.simple_vae_on_estimator --package-path trainer/ --region $REGION
gcloud ml-engine jobs stream-logs $JOB_NAME