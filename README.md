# 跆拳道卡腳檢測系統 (Taekwondo Leg Blocking Detection)

這個專案實現了一個基於電腦視覺的跆拳道比賽卡腳檢測系統，能夠自動識別、分析並記錄比賽中的卡腳情況，輔助裁判和教練進行評判和訓練。

![系統示例](examples/example_screenshot.png)

## 功能特點

- **自動卡腳檢測**：實時檢測選手之間的腿部碰撞與卡腳情況
- **雙選手骨架追蹤**：基於 MediaPipe 準確追蹤紅、藍方選手的骨架
- **支撐腳分析**：檢測選手支撐腳的位置與移動情況
- **重心穩定度分析**：計算選手重心穩定性分數
- **批次處理模式**：支持高效的視頻批次處理
- **詳細統計數據**：生成卡腳持續時間、頻率等統計報告
- **可視化輸出**：在視頻中標記卡腳事件、選手骨架和身分識別

## 安裝指南

### 系統要求

- Python 3.7 或更高版本
- 良好的 GPU 加速能提高處理效率

### 安裝步驟

1. 克隆此專案：
   ```bash
   git clone https://github.com/yourusername/Taekwondo_detect.git
   cd Taekwondo_detect
   ```

2. 安裝所需套件：
   ```bash
   pip install -r requirements.txt
   ```

3. 設定 Roboflow API 金鑰：
   - 在 `config.py` 中填入你的 API 金鑰，或設定環境變數

## 使用方法

### 基本用法

```python
from taekwondo import BatchTaekwondoDetector

# 創建檢測器
detector = BatchTaekwondoDetector()

# 處理視頻
detector.process_video_batch(
    input_path="path/to/your/video.mp4", 
    output_path="result_video.mp4", 
    skip_frames=2,      # 每 3 幀處理一次
    resize_width=480,   # 處理時的解析度
    resize_height=360,  
    show_progress=True  # 顯示進度
)

# 顯示卡腳統計
detector.show_block_statistics()

# 播放結果
detector.play_result("result_video.mp4")
```

### 設定不同參數

```python
# 使用自定義 API 金鑰和模型版本
detector = BatchTaekwondoDetector(
    api_key="your_api_key",
    project_id="your_project_id",
    model_version=4
)

# 調整卡腳檢測參數
detector.COLLISION_THRESHOLD = 65  # 腿部碰撞判定閾值(像素)
detector.MIN_BLOCK_DURATION = 0.25  # 最短卡腳持續時間(秒)
```

## 項目架構

```
Taekwondo_detect/
├── taekwondo.py          # 主要程式碼
├── demo.py               # 示例腳本
├── config.py             # 配置文件
├── requirements.txt      # 套件需求
├── examples/             # 示例資料
│   └── demo_video.mp4
├── docs/                 # 文檔
│   ├── algorithm.md      # 算法說明
│   └── api_reference.md  # API 參考
└── tests/                # 測試程式碼
```

## API 文檔概覽

### 主要類: `BatchTaekwondoDetector`

- `__init__(api_key=None, project_id=None, model_version=None)`: 初始化檢測器
- `process_video_batch(video_path, output_path, ...)`: 處理視頻並生成結果
- `show_block_statistics()`: 顯示卡腳統計數據
- `play_result(video_path, speed=1.0)`: 播放處理結果

詳細 API 文檔請參閱 [API 參考文檔](docs/api_reference.md)。

## 算法原理

系統使用了以下技術實現卡腳檢測：

1. 使用 Roboflow 模型識別運動員位置
2. 使用 MediaPipe 追蹤運動員骨架
3. 計算腿部關鍵點之間的最小距離
4. 基於時間和距離閾值判定卡腳事件
5. 分析支撐腳移動情況和重心穩定度

詳細算法說明請參閱 [算法說明文檔](docs/algorithm.md)。

## 注意事項

- 本系統需要良好的視頻質量以獲得最佳結果
- 側面視角的視頻通常能獲得最佳檢測效果
- 預設 Roboflow API 金鑰僅供示範使用，請替換為你自己的金鑰

## 貢獻指南

歡迎透過以下方式貢獻：

1. Fork 此專案
2. 創建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的修改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 開啟一個 Pull Request

## 授權

本專案使用 MIT 授權 - 詳見 [LICENSE](LICENSE) 文件

## 相關資源

- [MediaPipe 姿態估計](https://google.github.io/mediapipe/solutions/pose.html)
- [Roboflow Universe](https://universe.roboflow.com/)
- [跆拳道競賽規則](https://www.worldtaekwondo.org/rules/)