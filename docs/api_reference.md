# 跆拳道卡腳檢測系統 API 參考文檔

本文檔詳細說明了跆拳道卡腳檢測系統的API和主要功能。

## 主要類: BatchTaekwondoDetector

`BatchTaekwondoDetector` 是系統的核心類，負責處理視頻、檢測卡腳和生成分析結果。

### 初始化

```python
def __init__(self, api_key=None, project_id=None, model_version=None)
```

**參數:**
- `api_key` (str, 可選): Roboflow API金鑰
- `project_id` (str, 可選): Roboflow項目ID
- `model_version` (int, 可選): Roboflow模型版本

**說明:**  
初始化卡腳檢測器，設置骨架檢測模型和參數。如果未提供參數，將使用默認值或環境變數。

### 主要方法

#### 視頻處理

```python
def process_video_batch(self, video_path, output_path=None, skip_frames=2, 
                        resize_width=480, resize_height=360, show_progress=True)
```

**參數:**
- `video_path` (str): 輸入視頻路徑
- `output_path` (str, 可選): 輸出視頻路徑
- `skip_frames` (int): 處理時跳過的幀數
- `resize_width` (int): 處理時調整的寬度
- `resize_height` (int): 處理時調整的高度
- `show_progress` (bool): 是否顯示處理進度

**說明:**  
批次處理視頻並檢測卡腳事件。這是最主要的處理函數，整個過程分為兩個階段：
1. 第一階段：檢測每一幀中的人物、骨架和卡腳事件
2. 第二階段：將結果繪製到視頻上並保存

#### 播放結果

```python
def play_result(self, video_path, speed=1.0)
```

**參數:**
- `video_path` (str): 要播放的視頻路徑
- `speed` (float): 播放速度 (1.0=正常速度)

**說明:**  
播放處理後的視頻結果。

#### 顯示統計信息

```python
def show_block_statistics(self)
```

**說明:**  
顯示卡腳的統計信息，包括總次數、平均持續時間、最長/最短持續時間、支撐腳移動統計等。

### 輔助方法

#### 重置檢測狀態

```python
def reset_blocking_state(self)
```

**說明:**  
重置卡腳檢測狀態，包括計數器、時間記錄和支撐腳追蹤資料。

#### 計算重心位置

```python
def calculate_center_of_mass(self, pose_landmarks, width, height)
```

**參數:**
- `pose_landmarks`: MediaPipe姿勢關鍵點
- `width`: 畫面寬度
- `height`: 畫面高度

**返回:**
- 重心坐標 (x, y)

**說明:**  
根據骨架關鍵點計算人體重心位置。

#### 計算穩定度分數

```python
def calculate_stability_score(self, com_history)
```

**參數:**
- `com_history`: 重心位置歷史記錄

**返回:**
- 穩定度分數 (0-100，越高越穩定)

**說明:**  
根據重心歷史位置計算穩定度分數。

#### 分析重心穩定度

```python
def analyze_com_stability(self, com_position, history, scores)
```

**參數:**
- `com_position`: 當前重心位置
- `history`: 重心位置歷史記錄
- `scores`: 穩定度分數歷史記錄

**返回:**
- 當前穩定度分數

**說明:**  
分析重心穩定度並更新歷史記錄。

#### 獲取腿部關鍵點

```python
def get_leg_keypoints(self, pose_landmarks, width, height)
```

**參數:**
- `pose_landmarks`: MediaPipe姿勢關鍵點
- `width`: 畫面寬度
- `height`: 畫面高度

**返回:**
- 包含腿部關鍵點的字典

**說明:**  
從姿勢關鍵點中提取腿部關鍵點。

#### 識別支撐腳

```python
def identify_stance_leg(self, keypoints)
```

**參數:**
- `keypoints`: 人物的腳部關鍵點

**返回:**
- (支撐腳位置, 支撐腳類型 'left' 或 'right')

**說明:**  
根據腳踝的高度來識別支撐腳，較低的腳踝通常是支撐腳。

#### 檢查支撐腳移動

```python
def check_stance_leg_movement(self, current_stance, history)
```

**參數:**
- `current_stance`: 當前支撐腳位置
- `history`: 歷史支撐腳位置列表

**返回:**
- 支撐腳是否移動 (布林值)

**說明:**  
檢測支撐腳是否從初始位置移動超過閾值。

#### 檢查腿部碰撞

```python
def check_leg_collision(self, keypoints1, keypoints2)
```

**參數:**
- `keypoints1`: 第一個人的腿部關鍵點
- `keypoints2`: 第二個人的腿部關鍵點

**返回:**
- (是否碰撞, 最小距離)

**說明:**  
檢測兩個人的腿部是否碰撞。

#### 處理單個人物

```python
def process_person(self, frame, bbox, detector_idx=0)
```

**參數:**
- `frame`: 原始影像
- `bbox`: 人物邊界框 (x1, y1, x2, y2)
- `detector_idx`: 使用哪個姿勢檢測器

**返回:**
- (pose_landmarks, bbox)

**說明:**  
處理單個人物的骨架檢測。

#### 檢測人物和骨架

```python
def detect_people_with_skeleton(self, frame, resize_width=None, resize_height=None)
```

**參數:**
- `frame`: 輸入畫面
- `resize_width`: 調整寬度
- `resize_height`: 調整高度

**返回:**
- 包含檢測結果的列表

**說明:**  
在單一畫面中檢測人物和他們的骨架。

#### 檢測腿部卡腳和重心穩定度

```python
def detect_leg_blocking_and_stability(self, detections, current_time, frame_width, frame_height)
```

**參數:**
- `detections`: 人物檢測結果列表
- `current_time`: 當前視頻時間(秒)
- `frame_width`: 畫面寬度
- `frame_height`: 畫面高度

**返回:**
- (is_blocking, block_duration, min_distance, stance_moved, stability_info)

**說明:**  
檢測腿部卡腳和重心穩定度。

## 參數配置

### 卡腳檢測參數

- `COLLISION_THRESHOLD`: 腿部碰撞判定閾值(像素)
- `MIN_BLOCK_DURATION`: 最短卡腳持續時間(秒)
- `MIN_INTERVAL`: 兩次卡腳的最小間隔時間(秒)

### 支撐腳檢測參數

- `STANCE_MOVEMENT_THRESHOLD`: 支撐腳移動閾值(像素)
- `STANCE_HISTORY_SIZE`: 保存支撐腳位置歷史的幀數

### 重心穩定度檢測參數

- `COM_HISTORY_SIZE`: 保存重心位置歷史的幀數
- `COM_STABILITY_THRESHOLD`: 重心波動閾值(像素)
- `STABILITY_WINDOW`: 計算穩定度的時間窗口(幀數)