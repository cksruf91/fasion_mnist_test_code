MLflow model tracking example code for pytorch
------

# create directory for dataset
```shell
mkdir ./dataset
```
데이터셋은 FashionMNIST 사용

# Run MLflow ui
모델 학습 전에 MLflow ui를 먼저 띄운다.
```shell
export MLRUNS={directory}/mlruns
mlflow ui --host=127.0.0.1 --port=5000 --backend-store-uri=$MLRUNS
```
Access ui : ```http://localhost:5000/#/```

# training model
위에서 띄운 MLflow tracking 서버를 활용하기 위해 환경변수(MLFLOW_TRACKING_URI)를 설정한다.
```shell
# MLFlow
export MLFLOW_TRACKING_URI=http://localhost:5000/
```
모델 학습 
```shell
python train.py
```

# inference
MLflow에 모델이 저장되어 있을경우 그 모델을 사용하여 inference 하는 예시
```shell
python inference.py
```


