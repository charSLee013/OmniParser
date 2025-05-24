#!/usr/bin/env bash

# 定义变量
IMAGE_NAME="omniparser-omniparser-api"
CONTAINER_NAME="omniparser-api"
PORT="8000:8000"  # 主机端口:容器端口

# 打印步骤信息的函数
function step() {
  echo -e "\n\033[1;36m>>> $1\033[0m"
}

# 1. 停止并删除已存在的容器
step "检查并删除已存在的容器..."
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
  step "找到容器 $CONTAINER_NAME，正在删除..."
  docker rm -f $CONTAINER_NAME
else
  step "没有找到名为 $CONTAINER_NAME 的容器"
fi

# 2. 构建新镜像
step "构建新的 Docker 镜像..."
docker build -t $IMAGE_NAME .

# 3. 启动新容器
step "启动新容器..."
docker run -d --name $CONTAINER_NAME -p $PORT --gpus all --cpus=8 --memory=12g $IMAGE_NAME

# 4. 显示容器状态
step "新容器已启动:"
docker ps -a | grep $CONTAINER_NAME

# 5. 显示日志选项
step "查看日志命令: docker logs -f $CONTAINER_NAME"

# 如果你想自动查看日志，取消下面这行的注释
docker logs -f $CONTAINER_NAME 