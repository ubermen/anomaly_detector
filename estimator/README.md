# How to train, deploy and predict

__set_variables__
```shell
REGION=us-central1
```
__train__ 
```shell
JOB_NAME=lqad_train_32
OUTPUT_PATH=gs://bigus/$JOB_NAME
gcloud ml-engine jobs submit training $JOB_NAME --job-dir $OUTPUT_PATH --runtime-version 1.8 --config config.yaml --module-name trainer.simple_vae_on_estimator --package-path trainer/ --region $REGION
gcloud ml-engine jobs stream-logs $JOB_NAME
```
__deploy__ 
```shell
MODEL_NAME=lqad_test_00
MODEL_BINARIES=$OUTPUT_PATH/export/exporter/1546588689/
gcloud ml-engine versions create v1 --model $MODEL_NAME --origin $MODEL_BINARIES --runtime-version 1.8
```
__predict__
```shell
JOB_NAME=lqad_prediction_44
TEST_PATH=gs://bigus/data/globalsignin_devicemodel_test
PREDICTION_PATH=$OUTPUT_PATH/predictions
gcloud ml-engine jobs submit prediction $JOB_NAME -model $MODEL_NAME --version v1 --region $REGION --input-paths $TEST_PATH --output-path $PREDICTION_PATH --data-format TEXT
```
__summary__
```shell
gsutil cp $PREDICTION_PATH/prediction.results* .
cat prediction.results-0000* > total
sed -i -- 's/{"_1": //g' total
sed -i -- 's/"}//g' total
sed -i -- 's/, "_0": "/\t/g' total
sed -i -- 's/, "_0": /\t/g' total
sort -n total > sorted
```