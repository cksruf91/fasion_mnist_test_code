MLflow model tracking example code for pytorch
------

## 환경 세팅
* python version : 3.11.1
* ```pip install -r requirement.txt```

# 1. create directory for dataset
데이터셋은 FashionMNIST 사용
```shell
mkdir ./dataset
```

# [optional] 2. Run MLflowUI
모델 학습 전에 MLflow ui를 먼저 띄운다.
```shell
# create mlflow server
export MLRUNS={directory}/mlruns
mlflow ui --host=127.0.0.1 --port=6200 --backend-store-uri=$MLRUNS

# create mlflow server(docker version)
sh mlflowui/build.sh
```
ui url : ```http://localhost:6200/#/```

위에서 띄운 MLflow tracking 서버를 활용하기 위해 환경변수(MLFLOW_TRACKING_URI)를 설정한다.
```shell
# MLFlow
export MLFLOW_TRACKING_URI=http://localhost:6200/
```

# 3. training model
모델 학습 
```shell
python main.py -m cnn -t train -e 10
```

# 4. inference
```shell
python main.py -m cnn -t inference -c result/{checkpoint}.zip
```


