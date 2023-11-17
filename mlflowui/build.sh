#!/bin/bash

imageName=mlflow-ui
imageTag=latest

for containerId in $(docker ps -a | grep $imageName:$imageTag | awk '{print $1}')
do
  echo "remove exist container" $containerId
  docker kill $containerId > /dev/null
  docker rm $containerId > /dev/null
done
docker build -f mlflowui/Dockerfile . -t $imageName:$imageTag
docker run -d -p 6200:6200 -v $HOME/mlflows/mlruns:/mlruns $imageName:$imageTag

imageId=$(docker images | grep none | awk '{print $3}')
if [ -n "$imageId" ]; then
  echo "remove untagged docker images" $imageId
  docker rmi $imageId > /dev/null
fi