#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试 OmniParser API 的内存使用情况
发送多个连续请求并监控内存使用
"""

import os
import time
import requests
import base64
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

# API端点配置
API_URL = "http://localhost:8000/api/parse"
HEALTH_URL = "http://localhost:8000/api/health"

def encode_image(image_path):
    """将图像编码为base64字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def check_api_health():
    """检查API健康状态"""
    try:
        response = requests.get(HEALTH_URL, timeout=10)
        if response.status_code == 200:
            return response.json(), True
        else:
            print(f"API健康检查失败: {response.status_code}, {response.text}")
            return response.json(), False
    except Exception as e:
        print(f"API健康检查异常: {str(e)}")
        return None, False

def send_api_request(image_base64, params=None):
    """发送API请求并返回结果"""
    if params is None:
        params = {
            "box_threshold": 0.05,
            "iou_threshold": 0.1,
            "use_paddleocr": True,
            "imgsz": 640
        }
    
    data = {
        "image": image_base64,
        **params
    }
    
    try:
        response = requests.post(API_URL, json=data, timeout=300)
        if response.status_code == 200:
            return response.json(), True
        else:
            print(f"API请求失败: {response.status_code}, {response.text}")
            return response.json(), False
    except Exception as e:
        print(f"API请求异常: {str(e)}")
        return None, False

def plot_memory_usage(memory_data):
    """绘制内存使用图表"""
    timestamps = [item["timestamp"] for item in memory_data]
    cuda_allocated = [item.get("memory_info", {}).get("cuda_allocated_mb", 0) for item in memory_data]
    process_memory = [item.get("memory_info", {}).get("process_memory_mb", 0) for item in memory_data]
    
    # 转换timestamps为可读格式
    readable_timestamps = [time.strftime('%H:%M:%S', time.localtime(ts)) for ts in timestamps]
    
    plt.figure(figsize=(12, 6))
    
    # CUDA内存使用
    plt.subplot(1, 2, 1)
    plt.plot(readable_timestamps, cuda_allocated, 'r-o', label='CUDA内存 (MB)')
    plt.title('CUDA内存使用')
    plt.xlabel('时间')
    plt.ylabel('内存 (MB)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    
    # 进程内存使用
    plt.subplot(1, 2, 2)
    plt.plot(readable_timestamps, process_memory, 'b-o', label='进程内存 (MB)')
    plt.title('进程内存使用')
    plt.xlabel('时间')
    plt.ylabel('内存 (MB)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('memory_usage.png')
    plt.close()
    
    print(f"内存使用图表已保存到 memory_usage.png")

def main():
    parser = argparse.ArgumentParser(description='测试OmniParser API的内存使用')
    parser.add_argument('--image', type=str, required=True, help='测试图像路径')
    parser.add_argument('--num_requests', type=int, default=10, help='请求次数')
    parser.add_argument('--interval', type=float, default=1.0, help='请求间隔(秒)')
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.image):
        print(f"错误: 图像文件 {args.image} 不存在")
        return
    
    # 检查API健康状态
    health_data, health_ok = check_api_health()
    if not health_ok:
        print("API健康检查失败，请确保服务器正在运行")
        return
    
    print(f"API健康状态: {health_data['status']}")
    print(f"模型已加载: {health_data['models_loaded']}")
    print(f"初始内存使用情况: {health_data.get('memory_info', {})}")
    
    # 编码图像
    print(f"正在编码图像 {args.image}...")
    image_base64 = encode_image(args.image)
    
    # 保存内存使用数据
    memory_data = [health_data]
    
    # 发送多次请求
    print(f"\n开始发送 {args.num_requests} 个请求, 间隔 {args.interval} 秒...")
    success_count = 0
    
    for i in tqdm(range(args.num_requests)):
        # 发送请求
        result, success = send_api_request(image_base64)
        if success:
            success_count += 1
        
        # 检查健康状态以获取内存情况
        health_data, health_ok = check_api_health()
        if health_ok:
            memory_data.append(health_data)
        
        # 等待指定间隔
        if i < args.num_requests - 1:
            time.sleep(args.interval)
    
    # 打印结果
    print(f"\n测试完成: {success_count}/{args.num_requests} 个请求成功")
    
    # 检查最终健康状态
    final_health, health_ok = check_api_health()
    if health_ok:
        print(f"最终内存使用情况: {final_health.get('memory_info', {})}")
        memory_data.append(final_health)
    
    # 绘制内存使用图表
    if len(memory_data) > 1:
        plot_memory_usage(memory_data)
    
    # 保存内存数据到文件
    with open('memory_data.json', 'w') as f:
        json.dump(memory_data, f, indent=2)
    print("内存使用数据已保存到 memory_data.json")

if __name__ == "__main__":
    main() 