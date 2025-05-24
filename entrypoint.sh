#!/bin/bash

echo "Starting entrypoint script..."

while true; do
    echo "Searching for existing api_server.py processes..."
    # 强制杀死所有名为 python 且参数包含 api_server.py 的进程
    # -f 选项强制匹配完整命令行，-o 选项显示最老的匹配进程
    # pkill -f "python api_server.py"
    # 使用更精确的方式避免误杀其他python进程
    pgrep -f "python api_server.py" | xargs -r kill -9
    echo "Existing processes, if any, have been terminated."

    echo "Starting api_server.py..."
    python api_server.py

    echo "api_server.py exited with status $?. Restarting in 5 seconds..."
    sleep 5
done 