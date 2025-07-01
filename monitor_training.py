"""
Real-time monitoring script for DeepEarth training
Run in separate terminal with: watch -n 1 python monitor_training.py
"""

import GPUtil
import psutil
import torch
from datetime import datetime

def get_training_stats():
    # GPU stats
    gpus = GPUtil.getGPUs()
    gpu = gpus[0]
    
    # Memory stats
    memory = psutil.virtual_memory()
    
    # Disk stats
    disk = psutil.disk_usage('/')
    
    print(f"{'='*60}")
    print(f"DeepEarth Training Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    print(f"\nGPU Status:")
    print(f"  Device: {gpu.name}")
    print(f"  Memory: {gpu.memoryUsed:.1f} / {gpu.memoryTotal:.1f} GB ({gpu.memoryUtil*100:.1f}%)")
    print(f"  Utilization: {gpu.load*100:.1f}%")
    print(f"  Temperature: {gpu.temperature}°C")
    
    print(f"\nSystem Status:")
    print(f"  CPU: {psutil.cpu_percent()}%")
    print(f"  RAM: {memory.used/(1024**3):.1f} / {memory.total/(1024**3):.1f} GB ({memory.percent}%)")
    print(f"  Disk: {disk.used/(1024**3):.1f} / {disk.total/(1024**3):.1f} GB ({disk.percent}%)")
    
    # Network I/O (useful for data loading)
    net_io = psutil.net_io_counters()
    print(f"\nNetwork I/O:")
    print(f"  Sent: {net_io.bytes_sent/(1024**3):.2f} GB")
    print(f"  Recv: {net_io.bytes_recv/(1024**3):.2f} GB")

if __name__ == "__main__":
    get_training_stats()
