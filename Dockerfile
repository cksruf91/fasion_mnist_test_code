FROM python:3.9-slim

# 시간 동기화
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# requirements.txt 복사
COPY requirement.txt requirement.txt

# 라이브러리 설치
RUN apt-get update
RUN apt-get install gcc=4:10.2.1-1 -y
RUN pip install -U pip
RUN pip install --no-cache-dir -r requirement.txt

# build 명령어 : docker build -f Dockerfile -t interpreter:latest .