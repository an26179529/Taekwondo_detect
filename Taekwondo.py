import cv2
import numpy as np
from roboflow import Roboflow
import mediapipe as mp
import time
import os


class BatchTaekwondoDetector:
    def __init__(self, api_key=None, project_id=None, model_version=None):
        """Initialize batch taekwondo detector with leg blocking detection
        
        Args:
            api_key: Roboflow API key
            project_id: Roboflow project ID
            model_version: Roboflow model version
        """
        self.api_key = api_key or "gD0lCDeV52L8c4gDvtGh"
        self.project_id = project_id or "taekwondo-khq7c"
        self.model_version = model_version or 4
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 改進點 1: 使用更低的信心閾值來提高骨架點檢測率，但與原設定較接近
        # 改進點 2: 為每個人分配獨立的姿勢檢測器以減少檢測干擾
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
        
        # Connect to Roboflow API and load model
        print("Connecting to Roboflow API...")
        self.rf = Roboflow(api_key=self.api_key)
        self.model = self.rf.project(self.project_id).version(self.model_version).model
        print("Model loaded successfully!")
        
        # 恢復原始卡腳檢測參數
        self.COLLISION_THRESHOLD = 70
        self.MIN_BLOCK_DURATION = 0.3
        self.MIN_INTERVAL = 1.0
        
        # 支撐腳檢測相關參數
        self.STANCE_MOVEMENT_THRESHOLD = 15  # 支撐腳移動閾值 (像素)
        self.STANCE_HISTORY_SIZE = 10  # 保存支撐腳位置歷史的幀數
        
        # 跆拳道選手顏色標識 (紅/藍)
        self.RED_COLOR = (0, 0, 255)    # BGR格式 - 紅色
        self.BLUE_COLOR = (255, 0, 0)   # BGR格式 - 藍色
        
        # Initialize blocking state
        self.reset_blocking_state()

        # 重心穩定度檢測相關參數
        self.COM_HISTORY_SIZE = 15  # 保存重心位置歷史的幀數
        self.COM_STABILITY_THRESHOLD = 25  # 重心波動閾值 (像素)
        self.STABILITY_WINDOW = 10  # 計算穩定度的時間窗口 (幀數)
    
    def reset_blocking_state(self):
        """Reset leg blocking detection state"""
        self.block_count = 0
        self.block_durations = []  # Store all blocking events
        self.is_blocking = False
        self.current_block_start_time = None
        self.last_block_end_time = 0
        
        # 支撐腳追蹤資料
        self.person1_stance_history = []  # 儲存人員1支撐腳位置歷史 (紅方)
        self.person2_stance_history = []  # 儲存人員2支撐腳位置歷史 (藍方)
        self.stance_movement_detected = False  # 標記是否檢測到支撐腳移動
        self.red_stance_moved = False  # 紅方支撐腳是否移動
        self.blue_stance_moved = False  # 藍方支撐腳是否移動

        # 重心追蹤資料
        self.person1_com_history = []  # 儲存人員1重心位置歷史 (紅方)
        self.person2_com_history = []  # 儲存人員2重心位置歷史 (藍方)
        self.person1_stability_scores = []  # 紅方重心穩定度分數歷史
        self.person2_stability_scores = []  # 藍方重心穩定度分數歷史

    def calculate_center_of_mass(self, pose_landmarks, width, height):
        """計算人體重心位置
        
        Args:
            pose_landmarks: MediaPipe pose landmarks
            width: Frame width
            height: Frame height
            
        Returns:
            tuple: 重心坐標 (x, y)
        """
        if not pose_landmarks:
            return None
        
        # 定義主要身體部位及其相對重量 (加總為1.0)
        # 重心計算參考: 頭部(0.08), 軀幹(0.46), 上肢(0.12), 下肢(0.34)
        body_parts_weights = {
            self.mp_pose.PoseLandmark.NOSE: 0.08,  # 頭部
            self.mp_pose.PoseLandmark.LEFT_SHOULDER: 0.06,  # 左上肢
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER: 0.06,  # 右上肢
            self.mp_pose.PoseLandmark.LEFT_HIP: 0.15,  # 左軀幹
            self.mp_pose.PoseLandmark.RIGHT_HIP: 0.15,  # 右軀幹
            self.mp_pose.PoseLandmark.LEFT_KNEE: 0.12,  # 左大腿
            self.mp_pose.PoseLandmark.RIGHT_KNEE: 0.12,  # 右大腿
            self.mp_pose.PoseLandmark.LEFT_ANKLE: 0.13,  # 左小腿
            self.mp_pose.PoseLandmark.RIGHT_ANKLE: 0.13,  # 右小腿
        }
        
        # 計算加權平均位置
        weighted_x = 0
        weighted_y = 0
        total_weight = 0
        
        for landmark_id, weight in body_parts_weights.items():
            landmark = pose_landmarks.landmark[landmark_id]
            
            # 檢查關鍵點可見性，如果可見度過低則忽略
            if landmark.visibility > 0.5:
                weighted_x += landmark.x * width * weight
                weighted_y += landmark.y * height * weight
                total_weight += weight
        
        # 如果沒有足夠可見的關鍵點，返回None
        if total_weight < 0.5:
            return None
            
        # 調整權重總和
        if total_weight > 0:
            weighted_x /= total_weight
            weighted_y /= total_weight
        
        return (int(weighted_x), int(weighted_y))

    def calculate_stability_score(self, com_history):
        """計算重心穩定度分數
        
        Args:
            com_history: 重心位置歷史記錄
            
        Returns:
            float: 穩定度分數 (0-100，越高越穩定)
        """
        if len(com_history) < 3:
            return 100.0  # 資料不足時默認為穩定
        
        # 取最近的N個資料點計算穩定度
        recent_points = com_history[-self.STABILITY_WINDOW:] if len(com_history) > self.STABILITY_WINDOW else com_history
        
        # 計算連續點之間的移動距離
        distances = []
        for i in range(1, len(recent_points)):
            if recent_points[i] and recent_points[i-1]:  # 確保兩點都有效
                dist = np.linalg.norm(np.array(recent_points[i]) - np.array(recent_points[i-1]))
                distances.append(dist)
        
        if not distances:
            return 100.0  # 無有效距離數據時默認為穩定
        
        # 計算移動距離的標準差和平均值
        mean_movement = np.mean(distances)
        std_movement = np.std(distances) if len(distances) > 1 else 0
        
        # 綜合評估穩定度 (考慮移動平均和波動)
        stability_factor = mean_movement + std_movement
        
        # 將穩定度因子轉換為0-100的分數 (指數衰減)
        stability_score = 100 * np.exp(-stability_factor / self.COM_STABILITY_THRESHOLD)
        
        # 限制分數範圍
        stability_score = max(0, min(100, stability_score))
        
        return stability_score

    def analyze_com_stability(self, com_position, history, scores):
        """分析重心穩定度並更新歷史記錄
        
        Args:
            com_position: 當前重心位置
            history: 重心位置歷史記錄
            scores: 穩定度分數歷史記錄
            
        Returns:
            float: 當前穩定度分數
        """
        # 更新重心位置歷史
        if com_position:
            if len(history) >= self.COM_HISTORY_SIZE:
                history.pop(0)  # 移除最舊的記錄
            history.append(com_position)
        
        # 計算穩定度分數
        stability_score = self.calculate_stability_score(history)
        
        # 更新穩定度分數歷史
        if len(scores) >= self.COM_HISTORY_SIZE:
            scores.pop(0)
        scores.append(stability_score)
        
        return stability_score
    
    def get_leg_keypoints(self, pose_landmarks, width, height):
        """Extract leg keypoints from pose landmarks
        
        Args:
            pose_landmarks: MediaPipe pose landmarks
            width: Frame width
            height: Frame height
            
        Returns:
            dict: Dictionary of leg keypoints
        """
        if not pose_landmarks:
            return None
        
        keypoints = {
            'left_ankle': (
                int(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].x * width),
                int(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].y * height)
            ),
            'right_ankle': (
                int(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE].x * width),
                int(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE].y * height)
            ),
            'left_knee': (
                int(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].x * width),
                int(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].y * height)
            ),
            'right_knee': (
                int(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE].x * width),
                int(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE].y * height)
            ),
            # 添加臀部關鍵點以協助判斷姿勢
            'left_hip': (
                int(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x * width),
                int(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y * height)
            ),
            'right_hip': (
                int(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].x * width),
                int(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].y * height)
            )
        }
        return keypoints
    
    def identify_stance_leg(self, keypoints):
        """識別支撐腳（站立的腳）
        
        根據腳踝的高度來識別支撐腳，較低的腳踝通常是支撐腳
        
        Args:
            keypoints: 人物的腳部關鍵點
            
        Returns:
            tuple: (支撐腳位置, 支撐腳類型 'left' 或 'right')
        """
        if not keypoints:
            return None, None
            
        # 比較左右腳踝的y座標（較大的y座標表示較低的位置）
        left_y = keypoints['left_ankle'][1]
        right_y = keypoints['right_ankle'][1]
        
        if left_y > right_y:  # 左腳較低
            return keypoints['left_ankle'], 'left'
        else:  # 右腳較低
            return keypoints['right_ankle'], 'right'
    
    def check_stance_leg_movement(self, current_stance, history):
        """檢查支撐腳是否移動
        
        Args:
            current_stance: 當前支撐腳位置
            history: 歷史支撐腳位置列表
            
        Returns:
            bool: 支撐腳是否移動
        """
        if not current_stance or not history:
            return False
            
        # 如果歷史記錄少於2幀，不計算移動
        if len(history) < 2:
            return False
            
        # 計算當前位置與初始位置的距離
        initial_stance = history[0]
        movement = np.linalg.norm(np.array(current_stance) - np.array(initial_stance))
        
        return movement > self.STANCE_MOVEMENT_THRESHOLD
    
    def check_leg_collision(self, keypoints1, keypoints2):
        """Check if legs are colliding
        
        Args:
            keypoints1: First person leg keypoints
            keypoints2: Second person leg keypoints
            
        Returns:
            tuple: (is_colliding, min_distance)
        """
        if not keypoints1 or not keypoints2:
            return False, float('inf')
        
        points1 = [keypoints1['left_ankle'], keypoints1['right_ankle'],
                  keypoints1['left_knee'], keypoints1['right_knee']]
        points2 = [keypoints2['left_ankle'], keypoints2['right_ankle'],
                  keypoints2['left_knee'], keypoints2['right_knee']]
        
        # 改進點 4: 使用更精確的向量範數計算距離
        min_distance = float('inf')
        for p1 in points1:
            for p2 in points2:
                # 直接使用numpy的範數計算，更精確
                dist = np.linalg.norm(np.array(p1) - np.array(p2))
                min_distance = min(min_distance, dist)
        
        return min_distance < self.COLLISION_THRESHOLD, min_distance
    
    def process_person(self, frame, bbox, detector_idx=0):
        """處理單個人物的骨架檢測
        
        Args:
            frame: 原始影像
            bbox: 人物邊界框 (x1, y1, x2, y2)
            detector_idx: 使用哪個姿勢檢測器
            
        Returns:
            tuple: (pose_landmarks, bbox)
        """
        x1, y1, x2, y2 = bbox
        
        # 確保座標在影像範圍內
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        # 改進點 5: 增加ROI邊界填充到50像素（從30提高）
        padding = 50
        roi_x1 = max(0, x1 - padding)
        roi_y1 = max(0, y1 - padding)
        roi_x2 = min(frame.shape[1], x2 + padding)
        roi_y2 = min(frame.shape[0], y2 + padding)
        
        # 提取人物ROI
        person_image = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if person_image.size == 0 or person_image.shape[0] < 10 or person_image.shape[1] < 10:
            # 若ROI太小則跳過
            return None, bbox
        
        # 改進點 6: 使用對應人物的專用檢測器
        person_rgb = cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB)
        mp_results = self.pose_detectors[detector_idx % len(self.pose_detectors)].process(person_rgb)
        
        # 處理骨架座標
        if mp_results.pose_landmarks:
            for landmark in mp_results.pose_landmarks.landmark:
                # 轉換回原始影像座標
                px = landmark.x * (roi_x2 - roi_x1)
                py = landmark.y * (roi_y2 - roi_y1)
                
                landmark.x = (px + roi_x1) / frame.shape[1]
                landmark.y = (py + roi_y1) / frame.shape[0]
                
        return mp_results.pose_landmarks, bbox
    
    def detect_people_with_skeleton(self, frame, resize_width=None, resize_height=None):
        """Detect people and their skeletons in a single frame
        
        Args:
            frame: Input frame
            resize_width: Resize width for processing (None=keep original)
            resize_height: Resize height for processing (None=keep original)
            
        Returns:
            list: Detection results with skeleton landmarks
        """
        orig_height, orig_width = frame.shape[:2]
        
        # Resize if needed
        if resize_width and resize_height:
            resized_frame = cv2.resize(frame, (resize_width, resize_height))
            do_resize = True
        else:
            resized_frame = frame
            do_resize = False
            resize_width, resize_height = orig_width, orig_height
        
        try:
            # 改進點 7: 調整信心閾值和重疊閾值
            results = self.model.predict(resized_frame, confidence=40, overlap=30).json()
            predictions = results.get('predictions', [])
            
            # 根據x座標排序預測結果，確保穩定的人物編號
            # 在跆拳道比賽中，左側通常是紅方，右側是藍方
            predictions.sort(key=lambda p: p['x'])
            
            # Process each detection
            detections = []
            for idx, pred in enumerate(predictions[:2]):  # 只處理最多2個人
                # Get bounding box coordinates
                x1 = int(pred['x'] - pred['width'] / 2)
                y1 = int(pred['y'] - pred['height'] / 2)
                x2 = int(pred['x'] + pred['width'] / 2)
                y2 = int(pred['y'] + pred['height'] / 2)
                
                # Adjust coordinates if resized
                if do_resize:
                    x1 = int(x1 * orig_width / resize_width)
                    y1 = int(y1 * orig_height / resize_height)
                    x2 = int(x2 * orig_width / resize_width)
                    y2 = int(y2 * orig_height / resize_height)
                
                # 處理單個人物
                landmarks, bbox = self.process_person(frame, (x1, y1, x2, y2), idx)
                
                # 標識選手顏色 (idx=0是紅方，idx=1是藍方)
                # 根據排序，左側為紅方，右側為藍方
                player_color = "red" if idx == 0 else "blue"
                
                # Save detection results
                detection = {
                    'bbox': bbox,
                    'landmarks': landmarks,
                    'label': pred.get('class', 'pose'),
                    'confidence': pred.get('confidence', 0),
                    'color': player_color  # 添加顏色標識
                }
                
                detections.append(detection)
            
            return detections
        
        except Exception as e:
            print(f"Error detecting people: {e}")
            return []
    
    def detect_leg_blocking_and_stability(self, detections, current_time, frame_width, frame_height):
        """檢測腿部卡腳和重心穩定度
        
        Args:
            detections: List of person detections with landmarks
            current_time: Current video time in seconds
            frame_width: Frame width
            frame_height: Frame height
            
        Returns:
            tuple: (is_blocking, block_duration, min_distance, stance_moved, stability_info)
        """
        if len(detections) < 2:
            return False, 0, float('inf'), False, None
        
        # 創建穩定度信息結構
        stability_info = {
            'red_com': None,
            'blue_com': None,
            'red_stability': 0,
            'blue_stability': 0
        }
        
        # 嘗試識別紅藍方選手
        red_player = None
        blue_player = None
        
        for det in detections:
            if det.get('color') == 'red':
                red_player = det
            elif det.get('color') == 'blue':
                blue_player = det
        
        # 如果成功識別了紅藍方選手
        if red_player and blue_player:
            # 獲取兩位選手的腿部關鍵點
            keypoints_red = self.get_leg_keypoints(red_player['landmarks'], frame_width, frame_height)
            keypoints_blue = self.get_leg_keypoints(blue_player['landmarks'], frame_width, frame_height)
            
            # 計算紅方選手重心
            red_com = self.calculate_center_of_mass(red_player['landmarks'], frame_width, frame_height)
            # 計算藍方選手重心
            blue_com = self.calculate_center_of_mass(blue_player['landmarks'], frame_width, frame_height)
            
            # 分析紅方重心穩定度
            red_stability = self.analyze_com_stability(red_com, self.person1_com_history, self.person1_stability_scores)
            # 分析藍方重心穩定度
            blue_stability = self.analyze_com_stability(blue_com, self.person2_com_history, self.person2_stability_scores)
            
            # 更新穩定度信息
            stability_info = {
                'red_com': red_com,
                'blue_com': blue_com,
                'red_stability': red_stability,
                'blue_stability': blue_stability
            }
            
            # 檢測支撐腳
            stance_leg_red, stance_type_red = self.identify_stance_leg(keypoints_red)
            stance_leg_blue, stance_type_blue = self.identify_stance_leg(keypoints_blue)
            
            # 更新紅方支撐腳歷史
            if stance_leg_red:
                if len(self.person1_stance_history) >= self.STANCE_HISTORY_SIZE:
                    self.person1_stance_history.pop(0)  # 移除最舊的記錄
                self.person1_stance_history.append(stance_leg_red)
                
            # 更新藍方支撐腳歷史
            if stance_leg_blue:
                if len(self.person2_stance_history) >= self.STANCE_HISTORY_SIZE:
                    self.person2_stance_history.pop(0)  # 移除最舊的記錄
                self.person2_stance_history.append(stance_leg_blue)
            
            # 檢查紅方支撐腳是否移動
            red_moved = self.check_stance_leg_movement(stance_leg_red, self.person1_stance_history)
            # 檢查藍方支撐腳是否移動
            blue_moved = self.check_stance_leg_movement(stance_leg_blue, self.person2_stance_history)
            
            # 任一人支撐腳移動則標記為移動
            stance_moved = red_moved or blue_moved
            
            # Check if legs are colliding
            collision, min_distance = self.check_leg_collision(keypoints_red, keypoints_blue)
            current_duration = 0
            
            if collision:
                if not self.is_blocking:
                    if current_time - self.last_block_end_time >= self.MIN_INTERVAL:
                        self.is_blocking = True
                        self.current_block_start_time = current_time
                        print(f"\n偵測到新的卡腳 #{len(self.block_durations) + 1}，時間點: {current_time:.2f}秒")
                        
                        # 重置支撐腳歷史，重新開始追蹤
                        self.person1_stance_history = [stance_leg_red] if stance_leg_red else []
                        self.person2_stance_history = [stance_leg_blue] if stance_leg_blue else []
                        self.stance_movement_detected = False
                        self.red_stance_moved = False
                        self.blue_stance_moved = False
                
                if self.is_blocking:
                    current_duration = current_time - self.current_block_start_time
                    
                    # 檢測並記錄紅方/藍方支撐腳移動
                    if red_moved and not self.red_stance_moved:
                        self.red_stance_moved = True
                        self.stance_movement_detected = True
                        print(f"  紅方選手支撐腳移動，時間點: {current_time:.2f}秒")
                        
                    if blue_moved and not self.blue_stance_moved:
                        self.blue_stance_moved = True
                        self.stance_movement_detected = True
                        print(f"  藍方選手支撐腳移動，時間點: {current_time:.2f}秒")
            else:
                if self.is_blocking:
                    duration = current_time - self.current_block_start_time
                    if duration >= self.MIN_BLOCK_DURATION:
                        self.block_count += 1
                        self.block_durations.append({
                            'start_time': self.current_block_start_time,
                            'end_time': current_time,
                            'duration': duration,
                            'stance_moved': self.stance_movement_detected,
                            'red_moved': self.red_stance_moved,
                            'blue_moved': self.blue_stance_moved,
                            'red_stability': red_stability,  # 添加紅方重心穩定度
                            'blue_stability': blue_stability  # 添加藍方重心穩定度
                        })
                        self.last_block_end_time = current_time
                        
                        # 在終端顯示卡腳結束及持續時間
                        print(f"卡腳 #{self.block_count} 結束")
                        print(f"  開始時間: {self.current_block_start_time:.2f}秒")
                        print(f"  結束時間: {current_time:.2f}秒")
                        print(f"  持續時間: {duration:.2f}秒")
                        print(f"  紅方支撐腳移動: {'是' if self.red_stance_moved else '否'}")
                        print(f"  藍方支撐腳移動: {'是' if self.blue_stance_moved else '否'}")
                        print(f"  紅方重心穩定度: {red_stability:.1f}%")
                        print(f"  藍方重心穩定度: {blue_stability:.1f}%")
                    
                    self.is_blocking = False
                    self.current_block_start_time = None
                    self.stance_movement_detected = False
                    self.red_stance_moved = False
                    self.blue_stance_moved = False
            
            return self.is_blocking, current_duration, min_distance, stance_moved, stability_info
        
        # 如果無法識別紅藍方選手，退回到基於位置的方法
        else:
            # 使用原始方法（基於位置）
            keypoints1 = self.get_leg_keypoints(detections[0]['landmarks'], frame_width, frame_height)
            keypoints2 = self.get_leg_keypoints(detections[1]['landmarks'], frame_width, frame_height)
            
            # 計算選手1和選手2的重心
            com1 = self.calculate_center_of_mass(detections[0]['landmarks'], frame_width, frame_height)
            com2 = self.calculate_center_of_mass(detections[1]['landmarks'], frame_width, frame_height)
            
            # 分析重心穩定度
            stability1 = self.analyze_com_stability(com1, self.person1_com_history, self.person1_stability_scores)
            stability2 = self.analyze_com_stability(com2, self.person2_com_history, self.person2_stability_scores)
            
            # 更新穩定度信息
            stability_info = {
                'person1_com': com1,
                'person2_com': com2,
                'person1_stability': stability1,
                'person2_stability': stability2
            }
            
            # 檢測支撐腳
            stance_leg1, stance_type1 = self.identify_stance_leg(keypoints1)
            stance_leg2, stance_type2 = self.identify_stance_leg(keypoints2)
            
            # 更新支撐腳歷史
            if stance_leg1:
                if len(self.person1_stance_history) >= self.STANCE_HISTORY_SIZE:
                    self.person1_stance_history.pop(0)
                self.person1_stance_history.append(stance_leg1)
                
            if stance_leg2:
                if len(self.person2_stance_history) >= self.STANCE_HISTORY_SIZE:
                    self.person2_stance_history.pop(0)
                self.person2_stance_history.append(stance_leg2)
            
            # 檢查支撐腳是否移動
            person1_moved = self.check_stance_leg_movement(stance_leg1, self.person1_stance_history)
            person2_moved = self.check_stance_leg_movement(stance_leg2, self.person2_stance_history)
            
            # 任一人支撐腳移動則標記為移動
            stance_moved = person1_moved or person2_moved
            
            # Check if legs are colliding
            collision, min_distance = self.check_leg_collision(keypoints1, keypoints2)
            current_duration = 0
            
            if collision:
                if not self.is_blocking:
                    if current_time - self.last_block_end_time >= self.MIN_INTERVAL:
                        self.is_blocking = True
                        self.current_block_start_time = current_time
                        print(f"\n偵測到新的卡腳 #{len(self.block_durations) + 1}，時間點: {current_time:.2f}秒")
                        
                        # 重置支撐腳歷史，重新開始追蹤
                        self.person1_stance_history = [stance_leg1] if stance_leg1 else []
                        self.person2_stance_history = [stance_leg2] if stance_leg2 else []
                        self.stance_movement_detected = False
                        self.person1_moved = False
                        self.person2_moved = False
                
                if self.is_blocking:
                    current_duration = current_time - self.current_block_start_time
                    
                    # 如果發現支撐腳移動且尚未記錄移動
                    if person1_moved and not getattr(self, 'person1_moved', False):
                        self.person1_moved = True
                        self.stance_movement_detected = True
                        print(f"  選手1支撐腳移動，時間點: {current_time:.2f}秒")
                        
                    if person2_moved and not getattr(self, 'person2_moved', False):
                        self.person2_moved = True
                        self.stance_movement_detected = True
                        print(f"  選手2支撐腳移動，時間點: {current_time:.2f}秒")
            else:
                if self.is_blocking:
                    duration = current_time - self.current_block_start_time
                    if duration >= self.MIN_BLOCK_DURATION:
                        self.block_count += 1
                        self.block_durations.append({
                            'start_time': self.current_block_start_time,
                            'end_time': current_time,
                            'duration': duration,
                            'stance_moved': self.stance_movement_detected,
                            'person1_moved': getattr(self, 'person1_moved', False),
                            'person2_moved': getattr(self, 'person2_moved', False),
                            'person1_stability': stability1,  # 添加選手1重心穩定度
                            'person2_stability': stability2   # 添加選手2重心穩定度
                        })
                        self.last_block_end_time = current_time
                        
                        # 在終端顯示卡腳結束及持續時間
                        print(f"卡腳 #{self.block_count} 結束")
                        print(f"  開始時間: {self.current_block_start_time:.2f}秒")
                        print(f"  結束時間: {current_time:.2f}秒")
                        print(f"  持續時間: {duration:.2f}秒")
                        print(f"  選手1支撐腳移動: {'是' if getattr(self, 'person1_moved', False) else '否'}")
                        print(f"  選手2支撐腳移動: {'是' if getattr(self, 'person2_moved', False) else '否'}")
                        print(f"  選手1重心穩定度: {stability1:.1f}%")
                        print(f"  選手2重心穩定度: {stability2:.1f}%")
                    
                    self.is_blocking = False
                    self.current_block_start_time = None
                    self.stance_movement_detected = False
                    if hasattr(self, 'person1_moved'):
                        self.person1_moved = False
                    if hasattr(self, 'person2_moved'):
                        self.person2_moved = False
            
            return self.is_blocking, current_duration, min_distance, stance_moved, stability_info
    
    def process_video_batch(self, video_path, output_path=None, skip_frames=2,
                           resize_width=480, resize_height=360, show_progress=True):
        """Process video in batch mode with leg blocking detection
        
        Args:
            video_path: Input video path
            output_path: Output video path
            skip_frames: Number of frames to skip (process every n+1 frames)
            resize_width: Processing width (None=keep original)
            resize_height: Processing height (None=keep original)
            show_progress: Whether to show progress
        """
        print(f"Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return
        
        # Get video info
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info:")
        print(f"  - Resolution: {width}x{height}")
        print(f"  - FPS: {fps}")
        print(f"  - Total frames: {total_frames}")
        print(f"  - Duration: {total_frames/fps:.2f} seconds")
        print(f"  - Processing frequency: every {skip_frames+1} frames")
        
        # Setup output
        video_writer = None
        if output_path:
            # Create output directory
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Use mp4v codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Initialize variables
        frame_count = 0
        start_time = time.time()
        last_log_time = time.time()
        processed_frames = 0
        all_detections = []  # Store all frame detection results
        all_blocking_info = []  # Store blocking info for each frame
        
        # Reset blocking state
        self.reset_blocking_state()
        
        # Phase 1: Process all frames
        print("Phase 1: Detecting people, skeletons, and leg blocks...")
        
        try:
            while cap.isOpened():
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Increment frame count
                frame_count += 1
                
                # Get current video time
                current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                
                # Process every nth frame
                if frame_count % (skip_frames + 1) == 0:
                    # 改進點 9: 對每一幀都調整大小以加快處理
                    resized_frame = cv2.resize(frame, (resize_width, resize_height))
                    
                    # Detect people and skeletons
                    detections = self.detect_people_with_skeleton(resized_frame)
                    
                    # 將檢測結果座標調整回原始大小
                    for detection in detections:
                        if 'bbox' in detection:
                            x1, y1, x2, y2 = detection['bbox']
                            detection['bbox'] = (
                                int(x1 * width / resize_width),
                                int(y1 * height / resize_height),
                                int(x2 * width / resize_width),
                                int(y2 * height / resize_height)
                            )
                    
                    # Detect leg blocking
                    is_blocking, block_duration, min_distance, stance_moved, stability_info = self.detect_leg_blocking_and_stability(
                        detections, current_time, width, height)
                    
                    # Save blocking info
                    blocking_info = {
                        'is_blocking': is_blocking,
                        'block_duration': block_duration,
                        'min_distance': min_distance,
                        'current_time': current_time,
                        'stance_moved': stance_moved,
                        'stability_info': stability_info
                    }
                    
                    processed_frames += 1
                else:
                    # Use empty list for non-processed frames
                    detections = []
                    blocking_info = None
                
                # Save detection results
                all_detections.append(detections)
                all_blocking_info.append(blocking_info)
                
                # Show progress
                if show_progress and (frame_count % 10 == 0 or time.time() - last_log_time > 1.0):
                    elapsed = time.time() - start_time
                    print(f"Progress: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%) "
                         f"- Time: {current_time:.2f}/{total_frames/fps:.2f}s "
                         f"- Speed: {frame_count/elapsed:.1f} FPS")
                    last_log_time = time.time()
        
        except Exception as e:
            print(f"Error processing video: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Reset video for phase 2
            cap.release()
        
        # Phase 1 complete
        phase1_time = time.time() - start_time
        print(f"Phase 1 complete! Processed {processed_frames} frames in {phase1_time:.2f} seconds")
        print(f"Average processing speed: {processed_frames/phase1_time:.2f} FPS")
        print(f"Detected {self.block_count} leg blocks")
        
        # Skip phase 2 if no output path
        if not output_path:
            print("No output path set, skipping phase 2")
            self.show_block_statistics()
            return
        
        # Phase 2: Draw results to video and save
        print("Phase 2: Saving result video...")
        
        second_start_time = time.time()
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        last_valid_detections = []
        last_valid_blocking_info = None
        
        try:
            while cap.isOpened():
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Increment frame count
                frame_count += 1
                
                # Get current frame detections
                if frame_count <= len(all_detections):
                    current_detections = all_detections[frame_count - 1]
                    if current_detections:
                        last_valid_detections = current_detections
                    
                    current_blocking_info = all_blocking_info[frame_count - 1]
                    if current_blocking_info:
                        last_valid_blocking_info = current_blocking_info
                
                # Create output frame
                output_frame = frame.copy()
                
                # Draw detection results
                for idx, detection in enumerate(last_valid_detections):
                    # 獲取選手顏色
                    player_color = detection.get('color', 'unknown')
                    box_color = self.RED_COLOR if player_color == 'red' else self.BLUE_COLOR if player_color == 'blue' else (255, 0, 0)
                    
                    # Draw bounding box with color corresponding to player color
                    x1, y1, x2, y2 = detection['bbox']
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), box_color, 2)
                    
                    # Draw label
                    label = detection['label']
                    confidence = detection['confidence']
                    player_label = f"{'red' if player_color == 'red' else 'blue' if player_color == 'blue' else 'player'+str(idx+1)}"
                    cv2.putText(output_frame, f"{player_label}: {confidence:.2f}", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
                    
                    # Draw skeleton if available
                    if detection['landmarks']:
                        self.mp_drawing.draw_landmarks(
                            output_frame, 
                            detection['landmarks'],
                            self.mp_pose.POSE_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=box_color, thickness=2, circle_radius=2),
                            self.mp_drawing.DrawingSpec(color=box_color, thickness=2)
                        )
                
                # Draw blocking info
                if last_valid_blocking_info:
                    current_time = last_valid_blocking_info['current_time']
                    is_blocking = last_valid_blocking_info['is_blocking']
                    min_distance = last_valid_blocking_info['min_distance']
                    
                    # Display leg distance
                    cv2.putText(output_frame, f"Leg distance: {min_distance:.1f}px", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                               
                    # 不在影片上顯示卡腳持續時間，移至終端輸出
                    # 但仍然顯示是否正在卡腳中
                    if is_blocking:
                        cv2.putText(output_frame, "Leg Blocking!", (width - 150, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Add frame info
                current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                cv2.putText(output_frame, f"Frame: {frame_count}/{total_frames}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(output_frame, f"Time: {current_time:.2f}s", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(output_frame, f"Total blocks: {self.block_count}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Save to output video
                video_writer.write(output_frame)
                
                # Show progress
                if show_progress and frame_count % 30 == 0:
                    print(f"Saving progress: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
        
        except Exception as e:
            print(f"Error saving video: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Release resources
            cap.release()
            if video_writer is not None:
                video_writer.release()
            
            # Show processing statistics
            second_phase_time = time.time() - second_start_time
            total_time = time.time() - start_time
            print(f"\nProcessing statistics:")
            print(f"  - Phase 1 time (detection): {phase1_time:.2f}s")
            print(f"  - Phase 2 time (rendering): {second_phase_time:.2f}s")
            print(f"  - Total processing time: {total_time:.2f}s")
            print(f"  - Processed frames: {processed_frames}")
            print(f"  - Detection speed: {processed_frames/phase1_time:.2f} FPS")
            print(f"  - Results saved to: {output_path}")
            
            # Show blocking statistics
            self.show_block_statistics()
    
    def show_block_statistics(self):
        """顯示卡腳統計資訊"""
        print("\n卡腳統計資訊摘要:")
        print("="* 50)
        print(f"總卡腳次數: {self.block_count}")
        
        if self.block_durations:
            durations = [block['duration'] for block in self.block_durations]
            print(f"平均持續時間: {np.mean(durations):.2f}秒")
            print(f"最長持續時間: {max(durations):.2f}秒")
            print(f"最短持續時間: {min(durations):.2f}秒")
            
            # 計算紅藍方支撐腳移動的統計
            red_moved_blocks = [block for block in self.block_durations if block.get('red_moved', False)]
            blue_moved_blocks = [block for block in self.block_durations if block.get('blue_moved', False)]
            print(f"紅方支撐腳移動的卡腳次數: {len(red_moved_blocks)}")
            print(f"藍方支撐腳移動的卡腳次數: {len(blue_moved_blocks)}")

            # 使用正確的鍵名 'red_stability' 而不是 'red_stability_avg'
            red_stability_avg = np.mean([block.get('red_stability', 0) for block in self.block_durations])
            blue_stability_avg = np.mean([block.get('blue_stability', 0) for block in self.block_durations])
            print(f"紅方平均重心穩定度: {red_stability_avg:.2f}%")
            print(f"藍方平均重心穩定度: {blue_stability_avg:.2f}%")
            
        print("\n各卡腳詳細資訊:")
        print("="* 50)
        for i, block in enumerate(self.block_durations, 1):
            print(f"卡腳 #{i}:")
            print(f"  開始時間: {block['start_time']:.2f}秒")
            print(f"  結束時間: {block['end_time']:.2f}秒")
            print(f"  持續時間: {block['duration']:.2f}秒")
            print(f"  紅方支撐腳移動: {'是' if block.get('red_moved', False) else '否'}")
            print(f"  藍方支撐腳移動: {'是' if block.get('blue_moved', False) else '否'}")
            # 使用正確的鍵名並添加 .get 方法以防止鍵不存在
            print(f"  紅方重心穩定度: {block.get('red_stability', 0):.2f}%")
            print(f"  藍方重心穩定度: {block.get('blue_stability', 0):.2f}%")
            
        print("="* 50)
    
    def play_result(self, video_path, speed=1.0):
        """Play processed video result
        
        Args:
            video_path: Video path
            speed: Playback speed (1.0=normal speed)
        """
        print(f"Playing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return
        
        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Adjust playback delay
        delay = int(1000 / (fps * speed))
        
        # Play video
        print(f"Press 'q' to quit, 'space' to pause/resume")
        playing = True
        frame_count = 0
        
        try:
            while cap.isOpened():
                if playing:
                    # Read frame
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Increment frame count
                    frame_count += 1
                    
                    # Show frame
                    cv2.imshow(f"Result - {os.path.basename(video_path)}", frame)
                    
                    # Show progress
                    if frame_count % 30 == 0:
                        current_time = frame_count / fps
                        print(f"Playback: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%) "
                             f"- Time: {current_time:.2f}s")
                
                # Handle key press
                key = cv2.waitKey(delay) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):  # space key
                    playing = not playing
                    print("Paused" if not playing else "Resumed")
        
        except Exception as e:
            print(f"Error playing video: {e}")
        
        finally:
            # Release resources
            cap.release()
            cv2.destroyAllWindows()


    # Main program
if __name__ == "__main__":
    # Create detector
    detector = BatchTaekwondoDetector()
    
    try:
        # Set input and output paths
        input_path = r"E:\Taekwondo_detect\video\卡腳\側面_卡腳踢擊.mov"
        output_path = "result_with_leg_blocking.mp4"
        
        # Process video (detect people, skeletons, and leg blocks)
        detector.process_video_batch(
            input_path, 
            output_path, 
            skip_frames=2,      
            resize_width=480,   
            resize_height=360,  
            show_progress=True
        )
        
        # Play result
        print("\nProcessing complete! Press Enter to play the result...")
        input()
        detector.play_result(output_path, speed=1.0)  # 1.0=normal speed
    
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

        ##備份 API