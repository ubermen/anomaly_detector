# How to train, deploy and predict

__scripts__
```shell
./export.sh globalsignin devicemodel 20181004 train 100000 true
./export.sh globalsignin devicemodel 20181004 eval 1000 true
./export.sh globalsignin devicemodel 20181004 test -1 true
./train.sh globalsignin devicemodel 20181004 simple_vae_on_estimator
./deploy.sh globalsignin devicemodel 20181004
./predict.sh globalsignin devicemodel 20181004
./summary.sh globalsignin devicemodel 20181004
./report.sh globalsignin devicemodel 20181004 10 16
```
