# How to train, deploy and predict

__scripts__
```shell
./export.sh globalsignin devicemodel 20181004 train 100000 true
./export.sh globalsignin devicemodel 20181004 eval 1000 true
./export.sh globalsignin devicemodel 20181004 test -1 true
./train_on_mle.sh globalsignin devicemodel 20181004 simple_vae_app
./train_on_spark.sh globalsignin devicemodel 20181004 simple_vae_app 10
./deploy.sh globalsignin devicemodel 20181004
./predict_on_mle.sh globalsignin devicemodel 20181004
./predict_on_spark.sh globalsignin devicemodel 20181004 10
./summary.sh globalsignin devicemodel 20181004
./report.sh hdfs://datalake/lqad globalsignin devicemodel 20181004 10 16
```

__virtualenv_settings__
```
sudo pip install venv-pack
cd /home/web_admin/log-quality
virtualenv -p /usr/bin/python environment
source environment/bin/activate
pip install -r requirements.txt 
venv-pack -o environment.tar.gz
```
