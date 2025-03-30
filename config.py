"""
跆拳道卡腳檢測系統配置文件
"""
import os

# Roboflow API 配置
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "YOUR_API_KEY_HERE")
ROBOFLOW_PROJECT_ID = os.environ.get("ROBOFLOW_PROJECT_ID", "taekwondo-khq7c")
ROBOFLOW_MODEL_VERSION = os.environ.get("ROBOFLOW_MODEL_VERSION", 4)

# 卡腳檢測參數
COLLISION_THRESHOLD = 70  # 腿部碰撞判定閾值(像素)
MIN_BLOCK_DURATION = 0.3  # 最短卡腳持續時間(秒)
MIN_BLOCK_INTERVAL = 1.0  # 兩次卡腳的最小間隔時間(秒)

# 支撐腳檢測參數
STANCE_MOVEMENT_THRESHOLD = 15  # 支撐腳移動閾值(像素)
STANCE_HISTORY_SIZE = 10  # 保存支撐腳位置歷史的幀數

# 重心穩定度檢測參數
COM_HISTORY_SIZE = 15  # 保存重心位置歷史的幀數
COM_STABILITY_THRESHOLD = 25  # 重心波動閾值(像素)
STABILITY_WINDOW = 10  # 計算穩定度的時間窗口(幀數)

# 處理參數
DEFAULT_RESIZE_WIDTH = 480  # 處理時的默認寬度
DEFAULT_RESIZE_HEIGHT = 360  # 處理時的默認高度
DEFAULT_SKIP_FRAMES = 2  # 默認跳過的幀數(每n+1幀處理一次)

# 視覺化參數
RED_COLOR = (0, 0, 255)  # 紅方顏色(BGR格式)
BLUE_COLOR = (255, 0, 0)  # 藍方顏色(BGR格式)