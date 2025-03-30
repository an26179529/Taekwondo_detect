# 跆拳道卡腳檢測算法說明

本文檔詳細解釋了跆拳道卡腳檢測系統使用的算法和原理。

## 目錄
1. [總體架構](#總體架構)
2. [人物檢測](#人物檢測)
3. [骨架識別與追蹤](#骨架識別與追蹤)
4. [卡腳檢測算法](#卡腳檢測算法)
5. [支撐腳分析](#支撐腳分析)
6. [重心穩定度分析](#重心穩定度分析)
7. [檢測改進技術](#檢測改進技術)

## 總體架構

卡腳檢測系統的處理流程分為以下幾個階段：

1. **預處理**：調整視頻解析度以提高處理速度
2. **人物檢測**：利用 Roboflow API 檢測畫面中的跆拳道選手
3. **骨架追蹤**：應用 MediaPipe Pose 檢測選手的骨架關鍵點
4. **卡腳分析**：基於骨架關鍵點計算腿部碰撞情況
5. **支撐腳分析**：檢測支撐腳位置和移動情況
6. **重心穩定度分析**：評估選手重心穩定度
7. **結果可視化**：生成帶有標記的視頻輸出

這種多階段處理方法允許系統在不同設備上實現較好的效能，同時保持檢測的準確性。

## 人物檢測

系統使用預訓練的 Roboflow 物件檢測模型來識別畫面中的跆拳道選手。

### 技術細節：

- **模型**：taekwondo-khq7c (v4)
- **檢測信心閾值**：40%
- **預測重疊閾值**：30%
- **紅藍方識別**：基於 x 座標排序（左側為紅方，右側為藍方）

```python
# 檢測選手
results = self.model.predict(resized_frame, confidence=40, overlap=30).json()
predictions = results.get('predictions', [])
            
# 根據x座標排序預測結果，確保穩定的人物編號
# 在跆拳道比賽中，左側通常是紅方，右側是藍方
predictions.sort(key=lambda p: p['x'])
```

## 骨架識別與追蹤

檢測到選手後，系統使用 MediaPipe 姿勢估計模型識別選手骨架。

### 技術細節：

- **模型**：MediaPipe Pose
- **模型複雜度**：1（中等複雜度）
- **檢測信心閾值**：0.5
- **追蹤信心閾值**：0.5
- **優化**：每個選手使用獨立的姿勢檢測器以減少檢測干擾

```python
# 初始化多個檢測器實例
self.pose_detectors = [
    self.mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        smooth_landmarks=True,
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5    
    ) for _ in range(2)  # 創建兩個檢測器實例
]
```

系統從 MediaPipe 檢測的 33 個骨架關鍵點中，特別關注腿部關鍵點：

- 左/右腳踝
- 左/右膝