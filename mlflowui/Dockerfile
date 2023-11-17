FROM ghcr.io/mlflow/mlflow:v2.5.0

# 시간 동기화
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "6200"]