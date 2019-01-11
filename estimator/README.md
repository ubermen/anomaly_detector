# How to train, deploy and predict

__scripts__
```shell
./export.sh globalsignin devicemodel 20181004 train 100000
./export.sh globalsignin devicemodel 20181004 eval 1000
./train.sh globalsignin devicemodel 20181004 simple_vae_on_estimator
./deploy.sh globalsignin devicemodel 20181004
./predict.sh globalsignin devicemodel 20181004
./summary.sh globalsignin devicemodel 20181004
```
