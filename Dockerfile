# 使用 Ubuntu 22.04 最小镜像作为基础镜像
FROM m.daocloud.io/docker.io/ubuntu:22.04

# 设置时间
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
# 设置工作目录
WORKDIR /app

# 先安装ca-certificates以解决证书问题，使用http源
RUN apt-get update && apt-get install -y ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 替换 apt 源为清华源，使用HTTP而不是HTTPS来避免证书问题
RUN echo "# 默认注释了源码镜像以提高 apt update 速度，如有需要可自行取消注释" > /etc/apt/sources.list && \
    echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list

# 安装系统依赖和 Python 3
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    build-essential \
    cmake \
    curl \
    libopencv-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 设置 Python 别名
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# 设置 pip 配置使用阿里云源
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# 首先只复制 requirements.txt 文件，这样如果 requirements.txt 没有改变，
# 后续的 pip install 命令就会使用缓存
COPY requirements.txt /app/

# 安装 Python 依赖
RUN pip install uv && uv pip install --no-cache-dir -r requirements.txt --system && \
    rm -rf ~/.cache/pip

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 现在复制其余的应用代码
# 注意: 这一步会在代码变更时触发重建，但前面的层都会使用缓存
COPY . /app/

# 暴露端口
EXPOSE 8800

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8800/health || exit 1

# 启动 API 服务器
CMD ["python", "gradio_demo.py"] 