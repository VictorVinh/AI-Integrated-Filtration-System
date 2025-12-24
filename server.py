import os
import time
import json
import threading
import logging
import csv
import struct
from pathlib import Path
from datetime import datetime, timedelta
import queue
import numpy as np
import cv2
import requests
import pandas as pd
from ultralytics import YOLO
import itertools
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
from collections import deque
import sys
import signal
import socket
import urllib
try:
    from scipy.special import softmax as _softmax
except Exception:
    def _softmax(x, axis=1):
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / e.sum(axis=axis, keepdims=True)

try:
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ================== CONFIGURATION ==================
# WiFi Communication
ESP32_CAM_IP = "192.168.4.1"
ESP32_BASE_URL = f"http://{ESP32_CAM_IP}"

# URLs for ESP32-CAM communication
STREAM_URL = f"{ESP32_BASE_URL}/stream"
SENSOR_DATA_URL = f"{ESP32_BASE_URL}/get_sensor"
COMMAND_URL = f"{ESP32_BASE_URL}/set_command"
STATUS_URL = f"{ESP32_BASE_URL}/status"

# Model paths
YOLO_MODEL_PATH = r'D:\Water Filter\my_model\my_model.pt'
MLP_MODEL_1_TFLITE_PATH = r'D:\Water Filter\water_project_ei\pretrained-model\model.tflite'
MODEL_1_PARAMS_PATH = r'D:\Water Filter\water_project_ei\model\parameters.json'

MLP_MODEL_2_TFLITE_PATH = r'D:\Water Filter\filter_project_ei\pretrained-model\model.tflite'
MODEL_2_PARAMS_PATH = r'D:\Water Filter\filter_project_ei\model\parameters.json'

AUTO_RETRAIN_MODEL2 = True

# File paths
WATER_DATA_CSV = r'D:\Water Filter\water_data.csv'
SENSOR_DATA_CSV = "sensor_data.csv"
TRIAL_RESULTS_CSV = "trial_results.csv"
FILTER_TRAINING_CSV = "filter_training.csv"
WATER_SIGNATURES_JSON = "water_signatures.json"
DISTILLED_REP_CSV = "distilled_representation.csv"
DATA_TRAINING_READY_FLAG = "data_training_ready.txt"

# Relay configuration
RELAY_ORDER = [
    ("R1", "a"), ("R2", "b"), ("R4", "c"), 
    ("R5", "d"), ("R6", "e"), ("R7", "f"), ("R8", "g")
]

RELAY_COMMAND_MAP = {
    "activated_carbon": {"R1", "R2", "R5"},
    "coarse_filter": {"R1", "R2", "R4"},
    "fine_filter": {"R1", "R2", "R4", "R5"},
    "ro_filter": {"R1", "R2", "R6", "R7", "R8"},
    "ultrasonic_filter": {"R1", "R2", "R7"},
    "ultrasonic_coarse": {"R1", "R2", "R4", "R7"},
    "ultrasonic_carbon": {"R1", "R2", "R5", "R7"},
    "OFF": set()
}

FILTER_COMBINATIONS = [
    ["ultrasonic_filter"],
    ["activated_carbon"], 
    ["coarse_filter"],
    ["ultrasonic_filter", "activated_carbon"],
    ["ultrasonic_filter", "coarse_filter"],
    ["activated_carbon", "coarse_filter"],
    ["ultrasonic_filter", "activated_carbon", "coarse_filter"]
]

_prediction_history = deque(maxlen=10)
_interpreter_lock1 = threading.Lock()

trial_cancel_requested = False
trial_cancel_lock = threading.Lock()
SKIP_TRIAL_LABELS = {'nothing', 'bestwater'}

# Timing configuration
TRIAL_STABILIZE_SECONDS = 8
APPLY_ACCEPTED_DURATION = 15
SENSOR_READ_INTERVAL = 5
WIFI_RETRY_INTERVAL = 3

OOD_ZSCORE_THRESHOLD = 2.5
WATER_CONFIDENCE_THRESHOLD = 0.7
IMPROVEMENT_THRESHOLD = 0.15
SENSOR_SIMILARITY_THRESHOLD = 0.92

current_water_type = "Unknown"
current_water_confidence = 0.0
current_water_characteristics = []
current_ood_status = False
current_ood_reasons = []
current_recommended_method = "OFF"
current_method_source = "None"
current_trial_info = {
    "status": "Idle",  # Idle, Running, Completed, Failed
    "current_trial": 0,
    "total_trials": 0,
    "best_method": "None",
    "best_score": 0.0,
    "progress": "0%"
}
last_processing_time = 0

WINDOW_NAME = "Advanced Water Filter System - WIFI MODE"
DISPLAY_WIDTH = 1200
DISPLAY_HEIGHT = 800

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('water_filter_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AdvancedWaterFilter")

frame_lock = threading.Lock()
state_lock = threading.Lock()
csv_lock = threading.Lock()
trial_lock = threading.Lock()

latest_frame = None
latest_frame_with_boxes = None
current_sensor_data = {}
current_relay_state = "OFF"
current_command_chars = "abcdefg"
system_status = "Initializing"
yolo_detections = []
trial_in_progress = False
last_processing_result = {}
system_running = True
sensor_thresholds = {}
last_sensor_read_time = 0
last_threshold_update = 0
AUTO_THRESHOLD_UPDATE_INTERVAL = 3600
stream_active = False
last_sensor_request_time = 0
SENSOR_REQUEST_INTERVAL = 30

distilled_representation = None
water_signatures_cache = {}

# Model variables
yolo_model = None
interp1 = None; input1_details = None; output1_details = None; mean1 = None; scale1 = None; names1 = None
interp2 = None; input2_details = None; output2_details = None; mean2 = None; scale2 = None; names2 = None

class SmartTrialLearningSystem:
    """
    Hệ thống học từ trial thực tế:
    1. Thử nghiệm các phương pháp lọc
    2. Đánh giá hiệu quả
    3. Lưu vào database
    4. Dùng để train Model 2
    """
    
    def __init__(self):
        self.learning_data = []
        self.trial_history = []
        
        # Các phương pháp lọc cơ bản (đơn lẻ)
        self.base_filters = [
            "activated_carbon",  # Than hoạt tính
            "coarse_filter",     # Lọc thô
            "fine_filter",       # Lọc tinh
            "ro_filter",         # RO
            "ultrasonic_filter"  # Siêu âm
        ]
        
        # Ưu tiên theo loại chất gây ô nhiễm
        self.filter_specialization = {
            "high_tds": ["ro_filter", "fine_filter"],
            "high_turbidity": ["coarse_filter", "fine_filter", "ultrasonic_filter"],
            "high_voc": ["activated_carbon", "ultrasonic_filter"],
            "low_ph": ["activated_carbon"],  # Than hoạt tính có thể điều chỉnh pH
            "high_ph": ["activated_carbon"]
        }
        
    def analyze_water_characteristics(self, sensor_data):
        """Phân tích đặc điểm nước để chọn phương pháp phù hợp"""
        characteristics = []
        
        if sensor_data.get('TDS', 0) > 300:
            characteristics.append("high_tds")
        elif sensor_data.get('TDS', 0) < 50:
            characteristics.append("low_tds")
            
        if sensor_data.get('turbidity', 0) > 2.0:
            characteristics.append("high_turbidity")
            
        if sensor_data.get('VOC', 0) > 0.2:
            characteristics.append("high_voc")
            
        if sensor_data.get('ph', 7) < 6.5:
            characteristics.append("low_ph")
        elif sensor_data.get('ph', 7) > 8.5:
            characteristics.append("high_ph")
            
        return characteristics
    
    def suggest_filter_combinations(self, sensor_data, water_type):
        """
        Đề xuất các tổ hợp lọc dựa trên đặc điểm nước
        Dựa trên kiến thức chuyên môn về xử lý nước
        """
        characteristics = self.analyze_water_characteristics(sensor_data)
        
        # Các tổ hợp cơ bản theo đặc điểm
        suggested_combos = []
        
        # Luôn có tổ hợp OFF (không lọc)
        suggested_combos.append(["OFF"])
        
        # Dựa trên đặc điểm
        for char in characteristics:
            if char in self.filter_specialization:
                for filter_type in self.filter_specialization[char]:
                    suggested_combos.append([filter_type])
        
        # Thêm tổ hợp phức hợp cho vấn đề phức tạp
        if len(characteristics) >= 2:
            # Nhiều vấn đề -> cần tổ hợp
            if "high_tds" in characteristics and "high_voc" in characteristics:
                suggested_combos.append(["ro_filter", "activated_carbon"])
            if "high_turbidity" in characteristics and "high_voc" in characteristics:
                suggested_combos.append(["coarse_filter", "activated_carbon"])
            if "high_turbidity" in characteristics and "high_tds" in characteristics:
                suggested_combos.append(["coarse_filter", "ro_filter"])
        
        # Thêm tổ hợp toàn diện
        suggested_combos.append(["coarse_filter", "activated_carbon", "ro_filter"])
        suggested_combos.append(["ultrasonic_filter", "activated_carbon", "fine_filter"])
        
        # Loại bỏ trùng lặp
        unique_combos = []
        seen = set()
        for combo in suggested_combos:
            combo_key = ','.join(sorted(combo))
            if combo_key not in seen:
                seen.add(combo_key)
                unique_combos.append(combo)
        
        return unique_combos
    
    def evaluate_filter_performance(self, before_data, after_data):
        """
        Đánh giá hiệu suất của bộ lọc
        Trả về điểm số từ 0-1
        """
        try:
            # Chuẩn hóa giá trị lý tưởng (nước sạch)
            ideal_values = {
                'ph': 7.0,
                'TDS': 50,
                'turbidity': 0.5,
                'VOC': 0.05
            }
            
            scores = []
            
            for param, ideal in ideal_values.items():
                before_val = before_data.get(param, ideal)
                after_val = after_data.get(param, ideal)
                
                if param == 'ph':
                    # pH: càng gần 7 càng tốt
                    before_score = 1 - abs(before_val - ideal) / 7  # Max deviation 7
                    after_score = 1 - abs(after_val - ideal) / 7
                else:
                    # Các thông số khác: càng thấp càng tốt
                    max_val = max(before_val, ideal * 5, 1)  # Tránh chia 0
                    before_score = 1 - (before_val / max_val)
                    after_score = 1 - (after_val / max_val)
                
                # Cải thiện tương đối
                if before_score > 0:  # Tránh chia 0
                    improvement = (after_score - before_score) / before_score
                    scores.append(max(0, improvement))
            
            # Điểm trung bình
            if scores:
                return float(np.mean(scores))
            return 0.0
            
        except Exception as e:
            logger.error(f"Error evaluating filter performance: {e}")
            return 0.0
    
    def record_trial_result(self, sensor_before, sensor_after, filter_combo, water_type, performance_score):
        """Ghi lại kết quả trial với điểm số hiệu suất"""
        trial_result = {
            'timestamp': datetime.now().isoformat(),
            'sensor_before': sensor_before,
            'sensor_after': sensor_after,
            'filter_combo': filter_combo,
            'water_type': water_type,
            'performance_score': performance_score,
            'characteristics': self.analyze_water_characteristics(sensor_before)
        }
        
        self.trial_history.append(trial_result)
        
        # Lưu vào training data nếu performance tốt
        if performance_score > 0.3:
            self.add_to_training_data(trial_result)
        
        return trial_result
    
    def add_to_training_data(self, trial_result):
        """Thêm vào dataset training cho Model 2"""
        try:
            sensor_before = trial_result['sensor_before']
            filter_combo = trial_result['filter_combo']
            performance = trial_result['performance_score']
            
            # Chỉ lưu nếu performance tốt (score > 0.3)
            if performance > 0.3:
                with csv_lock, open(FILTER_TRAINING_CSV, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        float(sensor_before.get('ph', 0)),
                        float(sensor_before.get('TDS', 0)),
                        float(sensor_before.get('turbidity', 0)),
                        float(sensor_before.get('VOC', 0)),
                        trial_result.get('water_type', 'unknown'),
                        ','.join(filter_combo) if isinstance(filter_combo, list) else filter_combo,
                        performance  # Thêm cột performance
                    ])
                
                logger.info(f"Added to training data: {filter_combo} (score: {performance:.3f})")
                return True
                
        except Exception as e:
            logger.error(f"Error adding to training data: {e}")
        
        return False
    
    def get_best_method_for_water(self, sensor_data, water_type):
        """Tìm phương pháp tốt nhất cho loại nước này (dựa trên lịch sử)"""
        if not self.trial_history:
            return None
        
        best_score = -1
        best_method = None
        
        for trial in self.trial_history:
            # Tính độ tương đồng giữa sensor data hiện tại và lịch sử
            similarity = self.calculate_sensor_similarity(sensor_data, trial['sensor_before'])
            
            # Nếu tương đồng cao và performance tốt
            if similarity > 0.8 and trial['performance_score'] > best_score:
                best_score = trial['performance_score']
                best_method = trial['filter_combo']
        
        return best_method if best_score > 0.5 else None
    
    def calculate_sensor_similarity(self, data1, data2):
        """Tính độ tương đồng giữa hai sensor data"""
        try:
            params = ['ph', 'TDS', 'turbidity', 'VOC']
            vec1 = [data1.get(p, 0) for p in params]
            vec2 = [data2.get(p, 0) for p in params]
            
            # Chuẩn hóa
            vec1 = np.array(vec1, dtype=np.float32)
            vec2 = np.array(vec2, dtype=np.float32)
            
            # Cosine similarity
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

learning_system = SmartTrialLearningSystem()

class Model2BasedOODDetector:
    """
    Sử dụng Model 2 (MLP) để detect OOD dựa trên:
    1. Confidence Score: Nếu model không tự tin → data lạ
    2. Output Entropy: Nếu predictions phân tán đều → data애매
    3. Historical Comparison: So với predictions trước đó
    """
    
    def __init__(self, 
                 confidence_threshold=0.6,      # Dưới 0.6 = OOD
                 entropy_threshold=1.5,          # Trên 1.5 = OOD
                 history_size=100):
        
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        self.history_size = history_size
        
        # Lưu lịch sử predictions để detect anomaly
        self.prediction_history = deque(maxlen=history_size)
        self.confidence_history = deque(maxlen=history_size)
        
        # Statistics
        self.stats = {
            'total_predictions': 0,
            'ood_count': 0,
            'avg_confidence': 0.0,
            'last_update': None
        }
        
        self.stats_file = 'model2_ood_stats.json'
        self.load_stats()
    
    def calculate_entropy(self, probabilities):
        """
        Tính entropy của prediction distribution
        Entropy cao = model không chắc chắn = có thể OOD
        
        Args:
            probabilities: array of probabilities [p1, p2, ..., pn]
            
        Returns:
            float: entropy value
        """
        # Tránh log(0)
        probs = np.clip(probabilities, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log(probs))
        return float(entropy)
    
    def detect_ood(self, model_output, sensor_data, yolo_count=0):
        """
        Phát hiện OOD dựa trên output của Model 2
        
        Args:
            model_output: numpy array - output từ Model 2 (probabilities)
            sensor_data: dict - sensor values
            yolo_count: int - số detections từ YOLO
            
        Returns:
            tuple: (is_ood: bool, explanation: dict)
        """
        try:
            # 1. Tính confidence (max probability)
            max_confidence = float(np.max(model_output))
            predicted_class = int(np.argmax(model_output))
            
            # 2. Tính entropy
            entropy = self.calculate_entropy(model_output[0])
            
            # 3. Phân tích prediction distribution
            top3_probs = np.sort(model_output[0])[-3:][::-1]  # 3 xác suất cao nhất
            prob_gap = float(top3_probs[0] - top3_probs[1])  # Khoảng cách giữa top 1 và top 2
            
            # 4. So sánh với lịch sử
            historical_confidence = self._get_historical_confidence()
            confidence_deviation = abs(max_confidence - historical_confidence) if historical_confidence else 0
            
            # ================== LOGIC PHÁT HIỆN OOD ==================
            
            ood_reasons = []
            is_ood = False
            
            # Reason 1: Confidence quá thấp
            if max_confidence < self.confidence_threshold:
                ood_reasons.append(f"low_confidence({max_confidence:.3f}<{self.confidence_threshold})")
                is_ood = True
            
            # Reason 2: Entropy quá cao (model confused)
            if entropy > self.entropy_threshold:
                ood_reasons.append(f"high_entropy({entropy:.3f}>{self.entropy_threshold})")
                is_ood = True
            
            # Reason 3: Top 2 predictions gần nhau (model không chắc chắn)
            if prob_gap < 0.1:  # Nếu top 1 chỉ hơn top 2 < 10%
                ood_reasons.append(f"ambiguous_prediction(gap={prob_gap:.3f})")
                is_ood = True
            
            # Reason 4: Deviation lớn so với lịch sử
            if historical_confidence and confidence_deviation > 0.3:
                ood_reasons.append(f"unusual_confidence(deviation={confidence_deviation:.3f})")
                is_ood = True
            
            # Reason 5: Kết hợp với sensor extremes
            sensor_z_scores = self._calculate_sensor_z_scores(sensor_data)
            extreme_sensors = [f"{k}(z={v:.1f})" for k, v in sensor_z_scores.items() if abs(v) > 3]
            if extreme_sensors:
                ood_reasons.append(f"extreme_sensors:{','.join(extreme_sensors)}")
                is_ood = True
            
            
            explanation = {
                'is_ood': is_ood,
                'ood_score': float(1.0 - max_confidence),  # Score càng cao = càng OOD
                'reasons': ood_reasons,
                
                # Model 2 metrics
                'confidence': max_confidence,
                'predicted_class': predicted_class,
                'entropy': entropy,
                'top3_probs': top3_probs.tolist(),
                'prob_gap': prob_gap,
                
                # Historical comparison
                'historical_confidence': historical_confidence,
                'confidence_deviation': confidence_deviation,
                
                # Sensor analysis
                'sensor_z_scores': sensor_z_scores,
                'extreme_sensors': extreme_sensors,
                
                # Additional context
                'yolo_count': yolo_count,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update history
            self.prediction_history.append(predicted_class)
            self.confidence_history.append(max_confidence)
            
            # Update stats
            self.stats['total_predictions'] += 1
            if is_ood:
                self.stats['ood_count'] += 1
            
            self.stats['avg_confidence'] = float(np.mean(self.confidence_history)) if self.confidence_history else 0.0
            self.stats['last_update'] = datetime.now().isoformat()
            
            # Log
            if is_ood:
                logger.info(f"MODEL2-OOD DETECTED!")
                logger.info(f"Confidence: {max_confidence:.3f} | Entropy: {entropy:.3f} | Gap: {prob_gap:.3f}")
                logger.info(f"Reasons: {', '.join(ood_reasons)}")
                logger.info(f"Top-3 Probs: {top3_probs}")
                if extreme_sensors:
                    logger.info(f"   Extreme sensors: {', '.join(extreme_sensors)}")
            else:
                logger.debug(f"Normal prediction: class {predicted_class}, conf {max_confidence:.3f}")
            
            return is_ood, explanation
            
        except Exception as e:
            logger.error(f"Error in Model2-based OOD detection: {e}")
            return False, {"error": str(e)}
    
    def _get_historical_confidence(self):
        """Lấy confidence trung bình từ lịch sử"""
        if len(self.confidence_history) < 10:
            return None
        return float(np.mean(self.confidence_history))
    
    def _calculate_sensor_z_scores(self, sensor_data):
        """
        Tính z-score của sensor data dựa trên lịch sử
        (Đơn giản hơn Isolation Forest nhưng vẫn effective)
        """
        z_scores = {}
        
        # Nếu không có lịch sử, không tính z-score
        if len(self.prediction_history) < 20:
            return z_scores
        
        # Lấy sensor data từ prediction history (cần modify để lưu sensor history)
        # Tạm thời dùng hardcoded ranges (có thể improve sau)
        
        typical_ranges = {
            'ph': (6.0, 8.0, 7.0, 1.0),      # (min, max, mean, std)
            'TDS': (0, 300, 100, 80),
            'turbidity': (0, 5, 2, 1.5),
            'VOC': (0, 0.3, 0.05, 0.1)
        }
        
        for key, (min_val, max_val, mean, std) in typical_ranges.items():
            value = sensor_data.get(key, mean)
            z_score = (value - mean) / (std + 1e-8)
            z_scores[key] = float(z_score)
        
        return z_scores
    
    def save_stats(self):
        """Lưu statistics"""
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving Model2-OOD stats: {e}")
    
    def load_stats(self):
        """Tải statistics"""
        try:
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r') as f:
                    self.stats = json.load(f)
                logger.info(f"Loaded Model2-OOD stats: {self.stats['total_predictions']} predictions")
        except Exception as e:
            logger.warning(f"Could not load Model2-OOD stats: {e}")
    
    def get_ood_rate(self):
        """Tính tỷ lệ OOD"""
        if self.stats['total_predictions'] == 0:
            return 0.0
        return self.stats['ood_count'] / self.stats['total_predictions']

class SensorCalibrationSystem:
    """
    Hệ thống calibration và validation cho sensors:
    1. Warm-up period: Bỏ qua readings đầu tiên
    2. Moving average: Làm mịn noise
    3. Outlier detection: Loại bỏ readings bất thường
    4. Stability check: Chỉ chấp nhận data ổn định
    """
    
    def __init__(self, warm_up_seconds=60, window_size=5, stability_threshold=0.05):
        """
        Args:
            warm_up_seconds: Thời gian warm-up (bỏ qua readings đầu)
            window_size: Số readings để tính moving average
            stability_threshold: Ngưỡng ổn định (CV coefficient)
        """
        self.warm_up_seconds = warm_up_seconds
        self.window_size = window_size
        self.stability_threshold = stability_threshold
        
        # Tracking
        self.system_start_time = datetime.now()
        self.is_warmed_up = False
        
        # Moving average buffers cho mỗi sensor
        self.buffers = {
            'ph': deque(maxlen=window_size),
            'TDS': deque(maxlen=window_size),
            'turbidity': deque(maxlen=window_size),
            'VOC': deque(maxlen=window_size)
        }
        
        # Statistics
        self.stats = {
            'total_readings': 0,
            'rejected_warmup': 0,
            'rejected_outlier': 0,
            'rejected_unstable': 0,
            'accepted_readings': 0
        }
        
        # Sensor ranges (để detect outliers)
        self.sensor_ranges = {
            'ph': (0.0, 14.0),
            'TDS': (0.0, 2000.0),
            'turbidity': (0.0, 100.0),
            'VOC': (0.0, 10.0)
        }
        
    def is_in_warmup(self):
        """Kiểm tra xem có đang trong warm-up period không"""
        elapsed = (datetime.now() - self.system_start_time).total_seconds()
        return elapsed < self.warm_up_seconds
    
    def validate_reading(self, sensor_data):
        """
        Kiểm tra xem sensor reading có hợp lệ không
        
        Args:
            sensor_data: dict với sensor values
            
        Returns:
            tuple: (is_valid: bool, reason: str, corrected_data: dict)
        """
        self.stats['total_readings'] += 1
        
        # 1. CHECK WARM-UP
        if self.is_in_warmup():
            elapsed = (datetime.now() - self.system_start_time).total_seconds()
            remaining = self.warm_up_seconds - elapsed
            self.stats['rejected_warmup'] += 1
            logger.debug(f"Warm-up period: {remaining:.0f}s remaining")
            return False, f"warmup({remaining:.0f}s)", None
        else:
            if not self.is_warmed_up:
                self.is_warmed_up = True
                logger.info("Sensor warm-up completed!")
        
        # 2. CHECK OUTLIERS (giá trị ngoài range vật lý)
        outliers = []
        for key, value in sensor_data.items():
            if key in self.sensor_ranges:
                min_val, max_val = self.sensor_ranges[key]
                if value < min_val or value > max_val:
                    outliers.append(f"{key}={value:.2f} not in [{min_val},{max_val}]")
        
        if outliers:
            self.stats['rejected_outlier'] += 1
            logger.warning(f"Outlier detected: {', '.join(outliers)}")
            return False, f"outlier:{','.join(outliers)}", None
        
        # 3. UPDATE BUFFERS
        for key in ['ph', 'TDS', 'turbidity', 'VOC']:
            if key in sensor_data:
                self.buffers[key].append(sensor_data[key])
        
        # 4. CHECK STABILITY (chỉ khi buffer đầy)
        if all(len(self.buffers[key]) >= self.window_size for key in ['ph', 'TDS', 'turbidity', 'VOC']):
            unstable_sensors = []
            
            for key in ['ph', 'TDS', 'turbidity', 'VOC']:
                values = list(self.buffers[key])
                mean = np.mean(values)
                std = np.std(values)
                
                # Coefficient of Variation (CV)
                cv = std / (mean + 1e-8) if mean != 0 else 0
                
                # Nếu CV > threshold → không ổn định
                if cv > self.stability_threshold:
                    unstable_sensors.append(f"{key}(CV={cv:.3f})")
            
            if unstable_sensors:
                self.stats['rejected_unstable'] += 1
                logger.debug(f"Unstable sensors: {', '.join(unstable_sensors)}")
                return False, f"unstable:{','.join(unstable_sensors)}", None
        
        # 5. CALCULATE SMOOTHED VALUES (moving average)
        smoothed_data = {}
        for key in ['ph', 'TDS', 'turbidity', 'VOC']:
            if key in sensor_data and len(self.buffers[key]) > 0:
                smoothed_data[key] = float(np.mean(self.buffers[key]))
            else:
                smoothed_data[key] = sensor_data.get(key, 0.0)
        
        self.stats['accepted_readings'] += 1
        
        logger.debug(f"Valid reading (smoothed)")
        return True, "valid", smoothed_data
    
    def get_acceptance_rate(self):
        """Tính tỷ lệ readings được chấp nhận"""
        if self.stats['total_readings'] == 0:
            return 0.0
        return self.stats['accepted_readings'] / self.stats['total_readings']
    
    def get_stats_summary(self):
        """Lấy summary statistics"""
        return {
            'total': self.stats['total_readings'],
            'accepted': self.stats['accepted_readings'],
            'acceptance_rate': self.get_acceptance_rate(),
            'rejected_warmup': self.stats['rejected_warmup'],
            'rejected_outlier': self.stats['rejected_outlier'],
            'rejected_unstable': self.stats['rejected_unstable'],
            'is_warmed_up': self.is_warmed_up
        }


class DataValidationSystem:
    """
    Hệ thống validation cho training data:
    1. Kiểm tra data quality
    2. Loại bỏ samples không hợp lệ
    3. Balance dataset
    """
    
    def __init__(self):
        self.validation_stats = {
            'total_samples': 0,
            'invalid_samples': 0,
            'duplicate_samples': 0,
            'nothing_samples': 0,
            'valid_samples': 0
        }
    
    def validate_training_data(self, csv_path, output_path=None):
        """
        Kiểm tra và clean training data
        
        Args:
            csv_path: Đường dẫn CSV cần validate
            output_path: Đường dẫn lưu cleaned data (None = overwrite)
            
        Returns:
            DataFrame: Cleaned data
        """
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Validating {len(df)} training samples...")
            
            self.validation_stats['total_samples'] = len(df)
            original_count = len(df)
            
            # 1. LOẠI BỎ "nothing" với data không hợp lệ
            # "nothing" chỉ chấp nhận nếu data thực sự bất thường
            nothing_mask = df['water_label'] == 'nothing'
            nothing_samples = df[nothing_mask]
            
            # Kiểm tra xem "nothing" có data bất thường không
            valid_nothing = []
            for idx, row in nothing_samples.iterrows():
                # "nothing" hợp lệ nếu có ít nhất 1 sensor bất thường
                ph_bad = row['pH'] < 4 or row['pH'] > 10
                tds_bad = row['TDS_ppm'] > 500
                turbidity_bad = row['turbidity_NTU'] > 10
                voc_bad = row['VOC_mg_L'] > 1.0
                
                if ph_bad or tds_bad or turbidity_bad or voc_bad:
                    valid_nothing.append(idx)
            
            # Loại bỏ "nothing" không hợp lệ
            invalid_nothing = nothing_samples[~nothing_samples.index.isin(valid_nothing)]
            if len(invalid_nothing) > 0:
                logger.warning(f"Removing {len(invalid_nothing)} invalid 'nothing' samples")
                df = df[~df.index.isin(invalid_nothing.index)]
                self.validation_stats['invalid_samples'] += len(invalid_nothing)
            
            self.validation_stats['nothing_samples'] = len(valid_nothing)
            
            # 2. LOẠI BỎ DUPLICATES (sensor data giống hệt nhau)
            sensor_cols = ['pH', 'TDS_ppm', 'turbidity_NTU', 'VOC_mg_L']
            duplicates = df.duplicated(subset=sensor_cols, keep='first')
            if duplicates.sum() > 0:
                logger.warning(f"Removing {duplicates.sum()} duplicate samples")
                df = df[~duplicates]
                self.validation_stats['duplicate_samples'] = duplicates.sum()
            
            # 3. KIỂM TRA OUTLIERS
            outlier_mask = (
                (df['pH'] < 0) | (df['pH'] > 14) |
                (df['TDS_ppm'] < 0) | (df['TDS_ppm'] > 2000) |
                (df['turbidity_NTU'] < 0) | (df['turbidity_NTU'] > 100) |
                (df['VOC_mg_L'] < 0) | (df['VOC_mg_L'] > 10)
            )
            
            if outlier_mask.sum() > 0:
                logger.warning(f"Removing {outlier_mask.sum()} outlier samples")
                df = df[~outlier_mask]
                self.validation_stats['invalid_samples'] += outlier_mask.sum()
            
            # 4. KIỂM TRA FILTER METHODS HỢP LỆ
            valid_methods = set(RELAY_COMMAND_MAP.keys()) - {'OFF'}
            valid_combos = [','.join(combo) for combo in FILTER_COMBINATIONS]
            all_valid = list(valid_methods) + valid_combos
            
            invalid_methods = ~df['filter_methods'].isin(all_valid)
            if invalid_methods.sum() > 0:
                logger.warning(f"Removing {invalid_methods.sum()} samples with invalid filter methods")
                df = df[~invalid_methods]
                self.validation_stats['invalid_samples'] += invalid_methods.sum()
            
            self.validation_stats['valid_samples'] = len(df)
            
            # 5. REPORT
            logger.info("=" * 60)
            logger.info("TRAINING DATA VALIDATION REPORT")
            logger.info("=" * 60)
            logger.info(f"Original samples:        {original_count}")
            logger.info(f"Valid samples:           {len(df)} ({len(df)/original_count*100:.1f}%)")
            logger.info(f"Removed - Invalid:       {self.validation_stats['invalid_samples']}")
            logger.info(f"Removed - Duplicates:    {self.validation_stats['duplicate_samples']}")
            logger.info(f"Valid 'nothing' samples: {self.validation_stats['nothing_samples']}")
            logger.info("=" * 60)
            
            # 6. CLASS DISTRIBUTION
            logger.info("\nClass Distribution:")
            water_dist = df['water_label'].value_counts()
            for label, count in water_dist.items():
                logger.info(f"  {label}: {count} samples ({count/len(df)*100:.1f}%)")
            
            logger.info("\nFilter Method Distribution:")
            method_dist = df['filter_methods'].value_counts().head(10)
            for method, count in method_dist.items():
                logger.info(f"  {method}: {count} samples ({count/len(df)*100:.1f}%)")
            
            # 7. SAVE CLEANED DATA
            output_path = output_path or csv_path
            df.to_csv(output_path, index=False)
            logger.info(f"\nCleaned data saved to: {output_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error validating training data: {e}")
            return None
    
    def check_if_ready_for_training(self, csv_path, min_samples=100, min_classes=3):
        """
        Kiểm tra xem data đã sẵn sàng cho training chưa
        
        Args:
            csv_path: Đường dẫn CSV
            min_samples: Số samples tối thiểu
            min_classes: Số classes tối thiểu
            
        Returns:
            tuple: (is_ready: bool, reason: str, stats: dict)
        """
        try:
            df = pd.read_csv(csv_path)
            
            # Check 1: Số lượng samples
            if len(df) < min_samples:
                return False, f"Not enough samples: {len(df)}/{min_samples}", None
            
            # Check 2: Số lượng classes
            n_classes = df['water_label'].nunique()
            if n_classes < min_classes:
                return False, f"Not enough classes: {n_classes}/{min_classes}", None
            
            # Check 3: Class balance
            class_counts = df['water_label'].value_counts()
            min_class_count = class_counts.min()
            max_class_count = class_counts.max()
            imbalance_ratio = max_class_count / (min_class_count + 1e-8)
            
            if imbalance_ratio > 10:
                return False, f"Severe class imbalance: {imbalance_ratio:.1f}x", None
            
            # Check 4: Filter method diversity
            n_methods = df['filter_methods'].nunique()
            if n_methods < 3:
                return False, f"Not enough filter methods: {n_methods}/3", None
            
            stats = {
                'total_samples': len(df),
                'n_classes': n_classes,
                'n_methods': n_methods,
                'imbalance_ratio': imbalance_ratio,
                'class_distribution': class_counts.to_dict()
            }
            
            return True, "Ready for training", stats
            
        except Exception as e:
            logger.error(f"Error checking training readiness: {e}")
            return False, f"Error: {str(e)}", None
    

class TimeoutSession(requests.Session):
    """Session với timeout mặc định"""
    def __init__(self, default_timeout=(2, 5)):
        super().__init__()
        self.default_timeout = default_timeout
    
    def request(self, method, url, **kwargs):
        # Chỉ thêm timeout mặc định nếu không có timeout được chỉ định
        if 'timeout' not in kwargs:
            kwargs['timeout'] = self.default_timeout
        return super().request(method, url, **kwargs)

def create_optimized_session():
    """Tạo session với timeout ngắn hơn cho ESP32"""
    session = TimeoutSession(default_timeout=(2, 3))  # Timeout ngắn hơn
    
    retry_strategy = Retry(
        total=1,  # Chỉ retry 1 lần
        backoff_factor=0.1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    
    adapter = HTTPAdapter(
        pool_connections=1,
        pool_maxsize=1,
        max_retries=retry_strategy
    )
    
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

esp32_session = create_optimized_session()

def get_sensor_data_from_arduino():
    """Get sensor data - KHÔNG tắt stream"""
    try:
        # Sử dụng timeout ngắn
        response = esp32_session.get(SENSOR_DATA_URL, timeout=2)
        
        if response.status_code == 200:
            try:
                sensor_data = response.json()
                
                # Kiểm tra xem có phải error response không
                if isinstance(sensor_data, dict) and 'error' in sensor_data:
                    logger.debug(f"Sensor error: {sensor_data['error']}")
                    return None
                
                processed_data = {}
                if 'ph' in sensor_data:
                    processed_data['ph'] = float(sensor_data['ph'])
                if 'tds' in sensor_data:
                    processed_data['TDS'] = float(sensor_data['tds'])
                elif 'TDS' in sensor_data:
                    processed_data['TDS'] = float(sensor_data['TDS'])
                if 'turbidity' in sensor_data:
                    processed_data['turbidity'] = float(sensor_data['turbidity'])
                if 'voc' in sensor_data:
                    processed_data['VOC'] = float(sensor_data['voc'])
                elif 'VOC' in sensor_data:
                    processed_data['VOC'] = float(sensor_data['VOC'])
                
                logger.info(f"Raw sensor data from ESP32: {sensor_data}")
                logger.info(f"Processed sensor data: {processed_data}")
                
                return processed_data
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Invalid JSON from sensor: {e}, Response: {response.text}")
                return None
        else:
            logger.warning(f"No sensor data (HTTP {response.status_code})")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.warning(f"Sensor data request failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting sensor data: {e}")
        return None

def send_command_to_arduino(command_chars):
    try:
        if not command_chars.startswith("CHARS:"):
            command_chars = f"CHARS:{command_chars}"
            
        logger.info(f"Sending command: {command_chars}")
        
        response = esp32_session.post(COMMAND_URL, 
                                     data=command_chars, 
                                     headers={'Content-Type': 'text/plain'},
                                     timeout=2)
        
        if response.status_code == 200:
            logger.info(f"Command sent: {command_chars}")
            return True
        else:
            logger.warning(f"Command send failed with status: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.warning(f"Command send failed: {e}")
        return False
    except Exception as e:
        logger.warning(f"Unexpected error sending command: {e}")
        return False

def get_esp32_status():
    """Lấy trạng thái ESP32 - TỐI ƯU"""
    try:
        response = esp32_session.get(STATUS_URL)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def test_wifi_connection():
    """Kiểm tra kết nối WiFi đến ESP32-CAM"""
    try:
        response = requests.get(STATUS_URL, timeout=5)
        return response.status_code == 200
    except:
        return False

def apply_filter_method(method_name):
    if method_name in RELAY_COMMAND_MAP:
        relay_set = RELAY_COMMAND_MAP[method_name]
        command_chars = "".join([ch.upper() if r in relay_set else ch.lower() for r, ch in RELAY_ORDER])
        success = send_command_to_arduino(command_chars)  
        
        if success:
            global current_command_chars, current_relay_state
            with state_lock:
                current_command_chars = command_chars
                current_relay_state = method_name
            logger.info(f"Applied filter method via WiFi: {method_name}")
            return True
        else:
            logger.error(f"Failed to apply filter method: {method_name}")
            return False
    else:
        logger.warning(f"Unknown filter method: {method_name}")
        return False

def optimized_video_stream():
    global stream_active, latest_frame
    
    while system_running:
        try:
            if not stream_active:
                time.sleep(0.5)
                continue
                
            logger.info("Connecting to camera stream...")
            
            cap = cv2.VideoCapture(STREAM_URL)
            
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not cap.isOpened():
                logger.warning("Cannot connect to camera stream")
                time.sleep(2)
                continue
            
            logger.info("Camera stream connected!")
            
            frame_count = 0
            last_success_time = time.time()
            
            while system_running and stream_active:
                try:
                    # Đọc frame
                    ret, frame = cap.read()
                    if not ret:
                        # Kiểm tra timeout
                        if time.time() - last_success_time > 5:
                            logger.warning("Stream timeout, reconnecting...")
                            break
                        continue
                    
                    frame_count += 1
                    last_success_time = time.time()
                    
                    with frame_lock:
                        latest_frame = frame.copy()
                    
                    del frame
                    
                    time.sleep(0.2)  
                    
                except Exception as e:
                    logger.error(f"Error reading frame: {e}")
                    break
                
            cap.release()
            logger.info(f"Stream ended ({frame_count} frames)")
            
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Stream error: {e}")
            time.sleep(2)

def optimized_yolo_processing():
    """Xử lý YOLO trên frame từ video stream"""
    global latest_frame, latest_frame_with_boxes, yolo_detections
    
    while system_running:
        try:
            with frame_lock:
                frame = latest_frame.copy() if latest_frame is not None else None
            
            if frame is None:
                time.sleep(0.1)
                continue

            detections = []
            if yolo_model:
                try:
                    results = yolo_model(frame, imgsz=320, conf=0.25, iou=0.45, verbose=False, device='cpu')
                    
                    for result in results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                conf = box.conf[0].cpu().numpy()
                                cls = int(box.cls[0].cpu().numpy())
                                class_name = yolo_model.names[cls]
                                
                                if conf > 0.3:
                                    detections.append({
                                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                        'confidence': float(conf),
                                        'class': class_name,
                                        'class_id': cls
                                    })
                except Exception as e:
                    logger.error(f"YOLO inference error: {e}")

            with state_lock:
                yolo_detections = detections
            
            # Tạo frame với bounding boxes để hiển thị
            if frame is not None:
                display_frame = frame.copy()
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    conf = det['confidence']
                    class_name = det['class']
                    
                    color = (0, 255, 0) if 'bacteria' in class_name.lower() else (255, 0, 0)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name} {conf:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
                    cv2.putText(display_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                with frame_lock:
                    latest_frame_with_boxes = display_frame
                
        except Exception as e:
            logger.error(f"YOLO processing error: {e}")
            time.sleep(0.1)
        
        time.sleep(0.1)

# ================== DATA MANAGEMENT FUNCTIONS ==================

def ensure_data_files():
    """Đảm bảo các file dữ liệu tồn tại"""
    files_config = [
        (SENSOR_DATA_CSV, ["timestamp", "pH", "TDS_ppm", "turbidity_NTU", "VOC_mg_L"]),
        (TRIAL_RESULTS_CSV, ["timestamp", "combo", "pH_before", "TDS_before", "turbidity_before", "VOC_before", 
                           "pH_after", "TDS_after", "turbidity_after", "VOC_after", "improvement", "accepted"]),
        (FILTER_TRAINING_CSV, ["timestamp", "pH", "TDS_ppm", "turbidity_NTU", "VOC_mg_L", "water_label", "filter_methods"]),
        (DISTILLED_REP_CSV, ["parameter", "mean_value", "std_value", "count", "timestamp"])
    ]
    
    for filepath, headers in files_config:
        try:
            if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                logger.info(f"Created {filepath}")
        except Exception as e:
            logger.error(f"Error creating {filepath}: {e}")

def append_sensor_data(sensor_data):
    """Thêm dữ liệu cảm biến vào CSV"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with csv_lock, open(SENSOR_DATA_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                float(sensor_data.get('ph', 0.0)),
                float(sensor_data.get('TDS', sensor_data.get('tds', 0.0))),
                float(sensor_data.get('turbidity', 0.0)),
                float(sensor_data.get('VOC', sensor_data.get('voc', 0.0)))
            ])
        return True
    except Exception as e:
        logger.error(f"Error appending sensor data: {e}")
        return False

def append_trial_result(trial_data):
    """Thêm kết quả trial vào CSV"""
    try:
        with csv_lock, open(TRIAL_RESULTS_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(trial_data)
        return True
    except Exception as e:
        logger.error(f"Error appending trial result: {e}")
        return False

def append_filter_training(sensor_data, water_label, filter_method):
    """Thêm dữ liệu training vào CSV"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-d %H:%M:%S")
        with csv_lock, open(FILTER_TRAINING_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                float(sensor_data.get('ph', 0.0)),
                float(sensor_data.get('TDS', sensor_data.get('tds', 0.0))),
                float(sensor_data.get('turbidity', 0.0)),
                float(sensor_data.get('VOC', sensor_data.get('voc', 0.0))),
                str(water_label),
                str(filter_method)
            ])
        logger.info(f"Added training data: {water_label} -> {filter_method}")
        return True
    except Exception as e:
        logger.error(f"Error appending filter training: {e}")
        return False

def load_yolo_model():
    """Tải mô hình YOLO"""
    global yolo_model
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        logger.info(f"YOLO model loaded: {YOLO_MODEL_PATH}")
        return True
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        return False

def load_tflite_model(model_path):
    try:
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            return None, None, None
            
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        logger.info(f"MODEL1 input_details: {input_details}")
        logger.info(f"MODEL1 output_details: {output_details}")
        return interpreter, input_details, output_details
    except Exception as e:
        logger.error(f"Error loading TFLite model {model_path}: {e}")
        return None, None, None

def load_model_params(params_path):
    """Tải tham số mô hình"""
    try:
        with open(params_path, 'r', encoding='utf-8') as f:
            params = json.load(f)
        
        scaler_params = params.get('scaler_parameters', {}).get('model', {})
        mean = np.array(scaler_params.get('mean', []), dtype=np.float32)
        scale = np.array(scaler_params.get('scale', []), dtype=np.float32)
        
        class_names = params.get('class_names', {}).get('classes', [])
        
        return mean, scale, class_names
    except Exception as e:
        logger.error(f"Error loading model params {params_path}: {e}")
        return None, None, None

def predict_with_model(interpreter, input_details, output_details, normalized_input, apply_softmax=True):
    data = np.asarray(normalized_input, dtype=np.float32)
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)

    # prepare quantization info
    in_quant = input_details[0].get('quantization', (0.0,0))
    out_quant = output_details[0].get('quantization', (0.0,0))
    in_scale, in_zero = (in_quant if isinstance(in_quant, (list,tuple)) else (0.0,0))
    out_scale, out_zero = (out_quant if isinstance(out_quant, (list,tuple)) else (0.0,0))

    try:
        with _interpreter_lock1:
            # quantize input if needed
            if in_scale and in_scale != 0:
                q = np.round(data / in_scale + in_zero).astype(input_details[0]['dtype'])
                interpreter.set_tensor(input_details[0]['index'], q)
            else:
                interpreter.set_tensor(input_details[0]['index'], data.astype(input_details[0]['dtype']))

            interpreter.invoke()
            raw_out = interpreter.get_tensor(output_details[0]['index']).copy()  # copy to avoid view issues

        # dequantize if needed
        if out_scale and out_scale != 0:
            pred = (raw_out.astype(np.float32) - out_zero) * out_scale
        else:
            pred = raw_out.astype(np.float32)

        if pred.ndim == 1:
            pred = np.expand_dims(pred, axis=0)

        # **LOG RAW VALUES FOR DEBUGGING**
        logger.debug(f"_tflite_predict raw_out(before dequant/softmax) = {raw_out}")
        logger.debug(f"_tflite_predict dequantized = {pred}")

        # apply softmax if needed
        if apply_softmax:
            if not np.allclose(pred.sum(axis=1), 1.0, atol=1e-3):
                pred = _softmax(pred, axis=1)

        return pred

    except Exception as e:
        logger.exception(f"_tflite_predict error: {e}")
        return None

def initialize_models():
    """Khởi tạo tất cả mô hình AI"""
    global interp1, input1_details, output1_details, mean1, scale1, names1
    global interp2, input2_details, output2_details, mean2, scale2, names2
    
    load_yolo_model()
    
    if TF_AVAILABLE:
        interp1, input1_details, output1_details = load_tflite_model(MLP_MODEL_1_TFLITE_PATH)
        if interp1:
            mean1, scale1, names1 = load_model_params(MODEL_1_PARAMS_PATH)
            logger.info(f"Model 1 loaded successfully - Mean: {mean1}, Scale: {scale1}, Names: {names1}")
        
        interp2, input2_details, output2_details = load_tflite_model(MLP_MODEL_2_TFLITE_PATH)
        if interp2:
            mean2, scale2, names2 = load_model_params(MODEL_2_PARAMS_PATH)
            logger.info(f"Model 2 loaded successfully - Mean: {mean2}, Scale: {scale2}, Names: {names2}")

def load_water_signatures():
    """Tải water signatures từ file"""
    global water_signatures_cache
    try:
        if os.path.exists(WATER_SIGNATURES_JSON):
            with open(WATER_SIGNATURES_JSON, 'r', encoding='utf-8') as f:
                water_signatures_cache = json.load(f)
            logger.info(f"Loaded {len(water_signatures_cache)} water signatures")
            return water_signatures_cache
        return {}
    except Exception as e:
        logger.error(f"Error loading water signatures: {e}")
        return {}

def save_water_signatures():
    """Lưu water signatures vào file"""
    try:
        with open(WATER_SIGNATURES_JSON, 'w', encoding='utf-8') as f:
            json.dump(water_signatures_cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving water signatures: {e}")

def create_water_signature(sensor_data):
    """Tạo signature từ dữ liệu cảm biến"""
    try:
        ph = round(sensor_data.get('ph', 0), 1)
        tds = round(sensor_data.get('TDS', 0), -1)
        turbidity = round(sensor_data.get('turbidity', 0), 1)
        voc = round(sensor_data.get('VOC', 0), 2)
        return f"{ph}_{tds}_{turbidity}_{voc}"
    except Exception as e:
        logger.error(f"Error creating water signature: {e}")
        return "unknown"

def find_similar_water_signature(current_sensor_data, threshold=0.95):
    """Tìm water signature tương tự"""
    global water_signatures_cache
    
    if not water_signatures_cache:
        return None
    
    current_signature = create_water_signature(current_sensor_data)
    best_similarity = 0
    best_match = None
    
    for signature, data in water_signatures_cache.items():
        stored_sensor_data = data.get('sensor_data', {})
        similarity = calculate_sensor_similarity(current_sensor_data, stored_sensor_data)
        
        if similarity > best_similarity and similarity >= threshold:
            best_similarity = similarity
            best_match = data
    
    if best_match:
        logger.info(f"Found similar water signature: similarity {best_similarity:.3f}")
        return best_match
    
    return None

def calculate_sensor_similarity(sensor_data1, sensor_data2):
    """Tính độ tương đồng giữa hai sensor data"""
    try:
        features1 = [sensor_data1.get('ph', 0), sensor_data1.get('TDS', 0), sensor_data1.get('turbidity', 0), sensor_data1.get('VOC', 0)]
        features2 = [sensor_data2.get('ph', 0), sensor_data2.get('TDS', 0), sensor_data2.get('turbidity', 0), sensor_data2.get('VOC', 0)]
        
        features1 = np.array(features1)
        features2 = np.array(features2)
        
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        cosine_sim = np.dot(features1, features2) / (norm1 * norm2)
        return max(0.0, min(1.0, cosine_sim))
        
    except Exception as e:
        logger.error(f"Error calculating sensor similarity: {e}")
        return 0.0

def update_water_signature(sensor_data, best_method, improvement_score, achieved_threshold=False):
    """Cập nhật water signature"""
    global water_signatures_cache
    
    signature = create_water_signature(sensor_data)
    
    current_data = {
        'sensor_data': sensor_data,
        'best_method': best_method,
        'improvement_score': improvement_score,
        'achieved_threshold': achieved_threshold,
        'last_updated': datetime.now().isoformat(),
        'usage_count': 0
    }
    
    if signature in water_signatures_cache:
        existing_data = water_signatures_cache[signature]
        if improvement_score > existing_data.get('improvement_score', 0):
            water_signatures_cache[signature] = current_data
            logger.info(f"Updated water signature: {best_method} (improvement: {improvement_score:.3f})")
    else:
        water_signatures_cache[signature] = current_data
        logger.info(f"Created new water signature: {best_method} (improvement: {improvement_score:.3f})")
    
    save_water_signatures()
    return True

def calculate_distilled_representation():
    """Tính toán distilled water representation"""
    global distilled_representation
    
    try:
        if not os.path.exists(WATER_DATA_CSV):
            logger.warning("water_data.csv not found for distilled representation")
            return None
        
        df = pd.read_csv(WATER_DATA_CSV)
        distilled_data = df[df['label'].str.contains('distilled|clean|pure|bestwater', case=False, na=False)]
        
        if len(distilled_data) < 10:
            logger.warning(f"Not enough distilled water samples: {len(distilled_data)}")
            return None
        
        representation = {
            'pH_mean': float(distilled_data['pH'].mean()),
            'pH_std': float(distilled_data['pH'].std()),
            'TDS_ppm_mean': float(distilled_data['TDS_ppm'].mean()),
            'TDS_ppm_std': float(distilled_data['TDS_ppm'].std()),
            'turbidity_NTU_mean': float(distilled_data['turbidity_NTU'].mean()),
            'turbidity_NTU_std': float(distilled_data['turbidity_NTU'].std()),
            'VOC_mg_L_mean': float(distilled_data['VOC_mg_L'].mean()),
            'VOC_mg_L_std': float(distilled_data['VOC_mg_L'].std()),
            'count': len(distilled_data),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(DISTILLED_REP_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['parameter', 'mean_value', 'std_value', 'count', 'timestamp'])
            for key, value in representation.items():
                if key not in ['count', 'timestamp']:
                    param_name = key.replace('_mean', '').replace('_std', '')
                    if '_mean' in key:
                        writer.writerow([param_name, value, representation.get(f'{param_name}_std', 0), 
                                       representation['count'], representation['timestamp']])
        
        logger.info(f"Calculated distilled representation from {len(distilled_data)} samples")
        return representation
        
    except Exception as e:
        logger.error(f"Error calculating distilled representation: {e}")
        return None

def load_distilled_representation():
    """Tải distilled representation"""
    global distilled_representation
    
    try:
        if os.path.exists(DISTILLED_REP_CSV):
            df = pd.read_csv(DISTILLED_REP_CSV)
            representation = {}
            for _, row in df.iterrows():
                param = row['parameter']
                representation[f'{param}_mean'] = row['mean_value']
                representation[f'{param}_std'] = row['std_value']
            
            representation['count'] = int(df['count'].iloc[0])
            representation['timestamp'] = df['timestamp'].iloc[0]
            
            distilled_representation = representation
            logger.info("Loaded distilled water representation")
            return representation
        else:
            return calculate_distilled_representation()
    except Exception as e:
        logger.error(f"Error loading distilled representation: {e}")
        return calculate_distilled_representation()

def advanced_start_automated_filter_trial(initial_sensor, water_type):
    global trial_in_progress
    
    water_signature = create_water_signature(initial_sensor)
    
    similar_water = find_similar_water_signature(initial_sensor, SENSOR_SIMILARITY_THRESHOLD)
    if similar_water and similar_water.get('achieved_threshold', False):
        best_method = similar_water['best_method']
        logger.info(f"Found proven method in cache: {best_method}")
        
        if isinstance(best_method, str) and ',' in best_method:
            combo_methods = [m.strip() for m in best_method.split(',')]
            relay_set = set()
            for method in combo_methods:
                relay_set |= RELAY_COMMAND_MAP.get(method, set())
            
            command_chars = "".join([ch.upper() if r in relay_set else ch.lower() for r, ch in RELAY_ORDER])
            send_command_to_arduino(command_chars)
            logger.info(f"Applied combo method: {combo_methods}")
        else:
            # Single method
            apply_filter_method(best_method)
        
        return True
    
    def trial_worker():
        global trial_in_progress, current_sensor_data, distilled_representation
        
        logger.info("=== STARTING AUTOMATIC FILTER TRIAL ===")
        
        if distilled_representation is None:
            distilled_representation = load_distilled_representation()
            if distilled_representation is None:
                distilled_representation = {'pH_mean': 7.0, 'TDS_ppm_mean': 0, 'turbidity_NTU_mean': 0, 'VOC_mg_L_mean': 0}
        
        best_method = None
        best_improvement = -1
        trial_results = []

        for i, combo in enumerate(FILTER_COMBINATIONS):
            if not system_running:
                break
                
            logger.info(f"Trial {i+1}/{len(FILTER_COMBINATIONS)}: Testing {combo}")

            relay_set = set()
            for method in combo:
                relay_set |= RELAY_COMMAND_MAP.get(method, set())
            
            command_chars = "".join([ch.upper() if r in relay_set else ch.lower() for r, ch in RELAY_ORDER])
            send_command_to_arduino(command_chars)
            
            time.sleep(TRIAL_STABILIZE_SECONDS)

            # Lấy dữ liệu cảm biến sau khi áp dụng bộ lọc
            current_sensor = get_sensor_data_from_arduino()
            if not current_sensor:
                logger.warning("Failed to get sensor data during trial, skipping...")
                continue
            
            improvement_score = calculate_comprehensive_improvement(initial_sensor, current_sensor, distilled_representation)
            similarity_to_distilled = calculate_similarity(current_sensor, distilled_representation)
            
            meets_threshold = improvement_score >= IMPROVEMENT_THRESHOLD

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            trial_data = [
                timestamp, ",".join(combo),
                initial_sensor.get('ph', 0), initial_sensor.get('TDS', 0), 
                initial_sensor.get('turbidity', 0), initial_sensor.get('VOC', 0),
                current_sensor.get('ph', 0), current_sensor.get('TDS', 0),
                current_sensor.get('turbidity', 0), current_sensor.get('VOC', 0),
                improvement_score, int(meets_threshold)
            ]
            append_trial_result(trial_data)
            trial_results.append((combo, improvement_score, similarity_to_distilled, meets_threshold))
            
            logger.info(f"Trial {i+1} result: improvement={improvement_score:.3f}, accepted={meets_threshold}")

            if improvement_score > best_improvement:
                best_improvement = improvement_score
                best_method = combo

        if best_method:
            method_name = ",".join(best_method)
            achieved_threshold = best_improvement >= IMPROVEMENT_THRESHOLD
            
            logger.info(f"Best method found: {best_method} with improvement {best_improvement:.3f}")
            
            update_water_signature(initial_sensor, method_name, best_improvement, achieved_threshold)
            
            if achieved_threshold:
                relay_set = set()
                for method in best_method:
                    relay_set |= RELAY_COMMAND_MAP.get(method, set())
                
                command_chars = "".join([ch.upper() if r in relay_set else ch.lower() for r, ch in RELAY_ORDER])
                send_command_to_arduino(command_chars)
                
                append_filter_training(initial_sensor, water_type, method_name)
                logger.info(f"Applied best method: {best_method}")
                time.sleep(APPLY_ACCEPTED_DURATION)
            else:
                logger.warning(f"Best method did not meet threshold: {best_method}")
                send_command_to_arduino("abcdefg")
        else:
            logger.warning("No suitable filter method found in trial")
            send_command_to_arduino("abcdefg")
            update_water_signature(initial_sensor, "OFF", 0, False)
        
        if AUTO_RETRAIN_MODEL2:
            logger.info("Auto-retrain is ENABLED - preparing data for training...")
            def data_prep_worker():
                try:
                    success = prepare_data_for_colab_training()
                    if success:
                        logger.info("Data prepared for training!")
                    else:
                        logger.warning("Data preparation failed")
                except Exception as e:
                    logger.error(f"Error in data preparation worker: {e}")
            
            data_thread = threading.Thread(target=data_prep_worker, daemon=True)
            data_thread.start()
        else:
            logger.info("Auto-retrain disabled - data saved for manual processing")
        
        trial_in_progress = False
        logger.info("=== AUTOMATIC FILTER TRIAL COMPLETED ===")
    
    trial_in_progress = True
    thread = threading.Thread(target=trial_worker, daemon=True)
    thread.start()
    return True

def calculate_comprehensive_improvement(before_sensor, after_sensor, distilled_rep):
    """Tính toán điểm cải thiện toàn diện"""
    try:
        similarity_improvement = calculate_similarity(after_sensor, distilled_rep)
        
        turbidity_improvement = max(0, before_sensor.get('turbidity', 0) - after_sensor.get('turbidity', 0))
        tds_improvement = max(0, before_sensor.get('TDS', 0) - after_sensor.get('TDS', 0))
        voc_improvement = max(0, before_sensor.get('VOC', 0) - after_sensor.get('VOC', 0))
        
        max_turbidity = max(before_sensor.get('turbidity', 1), 1)
        max_tds = max(before_sensor.get('TDS', 1), 1)
        max_voc = max(before_sensor.get('VOC', 0.1), 0.1)
        
        normalized_turbidity_improvement = turbidity_improvement / max_turbidity
        normalized_tds_improvement = tds_improvement / max_tds
        normalized_voc_improvement = voc_improvement / max_voc
        
        comprehensive_score = (
            similarity_improvement * 0.5 +
            normalized_turbidity_improvement * 0.2 +
            normalized_tds_improvement * 0.2 +
            normalized_voc_improvement * 0.1
        )
        
        return min(comprehensive_score, 1.0)
        
    except Exception as e:
        logger.error(f"Error calculating comprehensive improvement: {e}")
        return 0.0

def calculate_similarity(sensor_data1, sensor_data2):
    """Tính độ tương đồng giữa hai sensor data"""
    try:
        features1 = [sensor_data1.get('ph', 0), sensor_data1.get('TDS', 0), sensor_data1.get('turbidity', 0), sensor_data1.get('VOC', 0)]
        
        if 'pH_mean' in sensor_data2:
            features2 = [sensor_data2.get('pH_mean', 0), sensor_data2.get('TDS_ppm_mean', 0), 
                        sensor_data2.get('turbidity_NTU_mean', 0), sensor_data2.get('VOC_mg_L_mean', 0)]
        else:
            features2 = [sensor_data2.get('pH', 0), sensor_data2.get('TDS_ppm', 0), 
                        sensor_data2.get('turbidity_NTU', 0), sensor_data2.get('VOC_mg_L', 0)]
        
        v1 = np.array(features1)
        v2 = np.array(features2)
        
        cosine_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        return float(cosine_sim)
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        return 0.0

model2_ood_detector = Model2BasedOODDetector(
    confidence_threshold=0.6,   
    entropy_threshold=1.5,       
    history_size=100
)

def start_smart_trial(initial_sensor, water_type):
    """Bắt đầu trial thông minh với learning system"""
    global trial_in_progress
    
    def trial_worker():
        global trial_in_progress
        
        trial_in_progress = True
        logger.info("=== STARTING SMART TRIAL ===")
        
        # Đề xuất các tổ hợp lọc dựa trên đặc điểm nước
        suggested_combos = learning_system.suggest_filter_combinations(initial_sensor, water_type)
        
        logger.info(f"Suggested combos: {suggested_combos}")
        
        best_combo = None
        best_score = -1
        best_after_data = None
        
        # Thử từng tổ hợp
        for i, combo in enumerate(suggested_combos):
            if not system_running:
                break
            
            logger.info(f"Trial {i+1}/{len(suggested_combos)}: Testing {combo}")
            
            # Áp dụng bộ lọc
            if combo == ["OFF"]:
                # Tắt tất cả relay
                send_command_to_arduino("abcdefg")
                time.sleep(3)
            else:
                relay_set = set()
                for method in combo:
                    relay_set |= RELAY_COMMAND_MAP.get(method, set())
                
                command_chars = "".join([ch.upper() if r in relay_set else ch.lower() 
                                        for r, ch in RELAY_ORDER])
                send_command_to_arduino(command_chars)
                time.sleep(8)  # Chờ ổn định
            
            # Lấy sensor data sau khi lọc
            after_sensor = get_sensor_data_from_arduino()
            if not after_sensor:
                logger.warning("Failed to get sensor data after filter, skipping...")
                continue
            
            # Đánh giá hiệu suất
            score = learning_system.evaluate_filter_performance(initial_sensor, after_sensor)
            
            # Ghi lại kết quả
            trial_result = learning_system.record_trial_result(
                initial_sensor, after_sensor, combo, water_type
            )
            
            logger.info(f"Trial {i+1} result: {combo} -> Score: {score:.3f}")
            
            # Cập nhật best
            if score > best_score:
                best_score = score
                best_combo = combo
                best_after_data = after_sensor
        
        # Kết thúc trial
        if best_combo and best_score > 0.3:
            logger.info(f"Best combo found: {best_combo} (score: {best_score:.3f})")
            
            # Áp dụng phương pháp tốt nhất
            if best_combo != ["OFF"]:
                relay_set = set()
                for method in best_combo:
                    relay_set |= RELAY_COMMAND_MAP.get(method, set())
                
                command_chars = "".join([ch.upper() if r in relay_set else ch.lower() 
                                        for r, ch in RELAY_ORDER])
                send_command_to_arduino(command_chars)
            
            # Cập nhật water signature
            update_water_signature(initial_sensor, ','.join(best_combo) if isinstance(best_combo, list) else best_combo, 
                                 best_score, True)
            
        else:
            logger.warning("No good filter combo found. Turning off all filters.")
            send_command_to_arduino("abcdefg")
        
        trial_in_progress = False
        logger.info("=== SMART TRIAL COMPLETED ===")
    
    # Chạy trial trong thread riêng
    thread = threading.Thread(target=trial_worker, daemon=True)
    thread.start()
    return True

def analyze_water_characteristics(sensor_data):
    """Phân tích đặc điểm nước từ sensor data"""
    characteristics = []
    
    ph = sensor_data.get('ph', 7.0)
    tds = sensor_data.get('TDS', 0.0)
    turbidity = sensor_data.get('turbidity', 0.0)
    voc = sensor_data.get('VOC', 0.0)
    
    # Phân loại pH
    if ph < 6.0:
        characteristics.append("Acidic")
    elif ph > 8.5:
        characteristics.append("Alkaline")
    else:
        characteristics.append("Neutral pH")
    
    # Phân loại TDS (độ cứng)
    if tds < 50:
        characteristics.append("Soft Water")
    elif tds < 150:
        characteristics.append("Medium Hardness")
    elif tds < 300:
        characteristics.append("Hard Water")
    else:
        characteristics.append("Very Hard Water")
    
    # Phân loại độ đục
    if turbidity < 1.0:
        characteristics.append("Clear")
    elif turbidity < 5.0:
        characteristics.append("Slightly Turbid")
    elif turbidity < 10.0:
        characteristics.append("Turbid")
    else:
        characteristics.append("Very Turbid")
    
    # Phân loại VOC
    if voc < 0.1:
        characteristics.append("Low VOC")
    elif voc < 0.5:
        characteristics.append("Medium VOC")
    else:
        characteristics.append("High VOC")
    
    return characteristics

def classify_water_simple(sensor_data):
    """Phân loại nước đơn giản khi không có model"""
    ph = sensor_data.get('ph', 7.0)
    tds = sensor_data.get('TDS', 0.0)
    turbidity = sensor_data.get('turbidity', 0.0)
    voc = sensor_data.get('VOC', 0.0)
    
    if tds > 500 and turbidity > 5.0:
        return "Heavily Contaminated"
    elif turbidity > 2.0:
        return "Turbid Water"
    elif tds > 300:
        return "Hard Water"
    elif voc > 0.2:
        return "Chemical Contaminated"
    elif ph < 6.5 or ph > 8.5:
        return "pH Abnormal"
    else:
        return "Normal Water"
    
def get_suggested_combinations(characteristics):
    """Đề xuất các tổ hợp lọc dựa trên đặc điểm"""
    combos = [["OFF"]]  # Luôn có tùy chọn tắt
    
    # Dựa trên đặc điểm
    if "Very Turbid" in characteristics or "Turbid" in characteristics:
        combos.append(["coarse_filter"])
        combos.append(["ultrasonic_filter"])
        combos.append(["coarse_filter", "fine_filter"])
    
    if "Very Hard Water" in characteristics or "Hard Water" in characteristics:
        combos.append(["ro_filter"])
        combos.append(["fine_filter"])
    
    if "High VOC" in characteristics or "Chemical Contaminated" in characteristics:
        combos.append(["activated_carbon"])
        combos.append(["ultrasonic_filter"])
        combos.append(["activated_carbon", "ultrasonic_filter"])
    
    # Tổ hợp toàn diện
    combos.append(["coarse_filter", "activated_carbon", "ro_filter"])
    combos.append(["ultrasonic_filter", "activated_carbon", "fine_filter"])
    
    return combos

def start_smart_trial(initial_sensor, water_type, characteristics):
    """Bắt đầu trial thông minh dựa trên đặc điểm nước"""
    global trial_in_progress, current_trial_info
    
    def trial_worker():
        global trial_in_progress, current_trial_info, current_sensor_data
        
        trial_in_progress = True
        logger.info("=== STARTING SMART TRIAL ===")
        
        # Đề xuất tổ hợp dựa trên đặc điểm
        suggested_combos = get_suggested_combinations(characteristics)
        
        # Cập nhật thông tin trial
        current_trial_info.update({
            "status": "Running",
            "current_trial": 0,
            "total_trials": len(suggested_combos),
            "best_method": "None",
            "best_score": 0.0,
            "progress": "0%"
        })
        
        logger.info(f"Testing {len(suggested_combos)} filter combinations")
        logger.info(f"Water type: {water_type}")
        logger.info(f"Characteristics: {characteristics}")
        
        best_combo = None
        best_score = -1
        best_after_data = None
        
        # Thử từng tổ hợp
        for i, combo in enumerate(suggested_combos):
            if not system_running:
                break
            
            # CẬP NHẬT TIẾN ĐỘ HIỂN THỊ
            current_trial_info["current_trial"] = i + 1
            current_trial_info["progress"] = f"{int((i+1)/len(suggested_combos)*100)}%"
            
            logger.info(f"Trial {i+1}/{len(suggested_combos)}: Testing {combo}")
            
            # Áp dụng bộ lọc
            if combo == ["OFF"]:
                send_command_to_arduino("abcdefg")
                time.sleep(3)
            else:
                relay_set = set()
                for method in combo:
                    relay_set |= RELAY_COMMAND_MAP.get(method, set())
                
                command_chars = "".join([ch.upper() if r in relay_set else ch.lower() 
                                        for r, ch in RELAY_ORDER])
                send_command_to_arduino(command_chars)
                time.sleep(TRIAL_STABILIZE_SECONDS)
            
            # Lấy sensor data sau khi lọc
            after_sensor = get_sensor_data_from_arduino()
            if not after_sensor:
                logger.warning("Failed to get sensor data after filter, skipping...")
                continue
            
            # Đánh giá hiệu suất
            score = calculate_improvement_score(initial_sensor, after_sensor)
            
            # Lưu kết quả trial
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            trial_data = [
                timestamp, ",".join(combo),
                initial_sensor.get('ph', 0), initial_sensor.get('TDS', 0), 
                initial_sensor.get('turbidity', 0), initial_sensor.get('VOC', 0),
                after_sensor.get('ph', 0), after_sensor.get('TDS', 0),
                after_sensor.get('turbidity', 0), after_sensor.get('VOC', 0),
                score, int(score >= IMPROVEMENT_THRESHOLD)
            ]
            append_trial_result(trial_data)
            
            logger.info(f"Trial {i+1} result: {combo} -> Score: {score:.3f}")
            
            # Cập nhật best
            if score > best_score:
                best_score = score
                best_combo = combo
                best_after_data = after_sensor
                
                # CẬP NHẬT HIỂN THỊ
                current_trial_info["best_method"] = ",".join(combo)
                current_trial_info["best_score"] = score
        
        # Kết thúc trial
        trial_in_progress = False
        
        if best_combo and best_score > IMPROVEMENT_THRESHOLD:
            logger.info(f"Best combo found: {best_combo} (score: {best_score:.3f})")
            
            # Áp dụng phương pháp tốt nhất
            if best_combo != ["OFF"]:
                relay_set = set()
                for method in best_combo:
                    relay_set |= RELAY_COMMAND_MAP.get(method, set())
                
                command_chars = "".join([ch.upper() if r in relay_set else ch.lower() 
                                        for r, ch in RELAY_ORDER])
                send_command_to_arduino(command_chars)
            
            # Cập nhật water signature
            update_water_signature(initial_sensor, ','.join(best_combo), best_score, True)
            
            # Lưu training data
            append_filter_training(initial_sensor, water_type, ','.join(best_combo))
            
            # CẬP NHẬT HIỂN THỊ KẾT QUẢ
            current_recommended_method = ','.join(best_combo)
            current_method_source = "Trial Result"
            current_trial_info["status"] = "Completed"
            
        else:
            logger.warning("No good filter combo found")
            send_command_to_arduino("abcdefg")
            
            current_recommended_method = "OFF"
            current_method_source = "Trial Failed"
            current_trial_info["status"] = "Failed"
        
        logger.info("=== SMART TRIAL COMPLETED ===")
    
    # Chạy trial trong thread riêng
    thread = threading.Thread(target=trial_worker, daemon=True)
    thread.start()
    return True

def calculate_improvement_score(before, after):
    """Tính điểm cải thiện"""
    try:
        # Chỉ số lý tưởng
        ideal = {'ph': 7.0, 'TDS': 50, 'turbidity': 0.5, 'VOC': 0.05}
        
        scores = []
        for param in ['ph', 'TDS', 'turbidity', 'VOC']:
            before_val = before.get(param, ideal[param])
            after_val = after.get(param, ideal[param])
            
            if param == 'ph':
                # pH: càng gần 7 càng tốt
                before_diff = abs(before_val - 7.0)
                after_diff = abs(after_val - 7.0)
                if before_diff > 0:
                    improvement = (before_diff - after_diff) / before_diff
                    scores.append(max(0, improvement))
            else:
                # Các chỉ số khác: càng thấp càng tốt
                if before_val > 0:
                    improvement = (before_val - after_val) / before_val
                    scores.append(max(0, improvement))
        
        return float(np.mean(scores)) if scores else 0.0
    except:
        return 0.0

def should_skip_trial(water_label: str) -> bool:
    """Kiểm tra xem có nên bỏ qua trial không (nước sạch, nước cất, không cần lọc)"""
    if not water_label:
        return False
    label = str(water_label).strip().lower()
    return label in SKIP_TRIAL_LABELS 

def has_good_known_solution(sensor_data, water_type) -> tuple[bool, str]:
    """
    Kiểm tra xem có phương pháp tốt đã biết cho loại nước này chưa
    Trả về: (có_không, phương_pháp_tốt_nhất hoặc None)
    """
    if not learning_system.trial_history:
        return False, None
    
    # 1. Dùng learning system tìm phương pháp tốt nhất từ lịch sử
    best_from_history = learning_system.get_best_method_for_water(sensor_data, water_type)
    if best_from_history and learning_system.calculate_sensor_similarity(
        sensor_data, learning_system.trial_history[-1]['sensor_before'] if learning_system.trial_history else sensor_data
    ) > 0.85:
        return True, best_from_history

    # 2. Dùng water signature cache (nếu có)
    similar_sig = find_similar_water_signature(sensor_data, threshold=SENSOR_SIMILARITY_THRESHOLD)
    if similar_sig and similar_sig.get('achieved_threshold', False):
        return True, similar_sig['best_method']
    
    return False, None

def intelligent_process_sensor_data(sensor_data):
    """
    XỬ LÝ THÔNG MINH - PHIÊN BẢN HOÀN CHỈNH
    Chỉ chạy trial khi:
    - Không phải nothing/bestwater
    - Không có phương pháp tốt đã biết từ lịch sử
    - Không có water signature tương tự
    """
    global current_water_type, current_water_confidence, current_water_characteristics
    global current_ood_status, current_ood_reasons, current_recommended_method, current_method_source
    global trial_in_progress, last_processing_result

    if not sensor_data:
        return

    last_processing_time = time.time()
    logger.info(f"Processing sensor: pH={sensor_data.get('ph','?'):.2f}, "
                f"TDS={sensor_data.get('TDS','?'):.1f}, Turb={sensor_data.get('turbidity','?'):.2f}, "
                f"VOC={sensor_data.get('VOC','?'):.3f}")

    # 1. Phân tích đặc điểm nước
    current_water_characteristics = analyze_water_characteristics(sensor_data)

    # 2. Phân loại bằng Model 1 (có fallback)
    water_type, water_confidence, _ = classify_water_with_model1(sensor_data)
    current_water_type = water_type
    current_water_confidence = water_confidence

    logger.info(f"Detected water type: '{water_type}' (conf: {water_confidence:.3f})")

    # ================== LOGIC QUYẾT ĐỊNH TRIAL ==================
    trial_needed = False
    trial_reason = ""

    # Trường hợp 1: Nước sạch → bỏ qua hoàn toàn
    if should_skip_trial(water_type):
            logger.info(f"Detected CLEAN water ('{water_type}') → Turning OFF all filters")

            current_ood_status = False
            current_recommended_method = "OFF"
            current_method_source = "CleanWater"
            current_ood_reasons = [f"Label '{water_type}' → no filtering needed"]

            # TẮT RELAY NGAY
            apply_filter_method("OFF")

            # HỦY TRIAL NẾU ĐANG CHẠY
            if trial_in_progress:
                with trial_cancel_lock:
                    trial_cancel_requested = True
                logger.info("Trial cancelled due to clean water detection!")

            return
        
    else:
        # Trường hợp 2: Có phương pháp tốt đã biết từ lịch sử?
        has_solution, known_method = has_good_known_solution(sensor_data, water_type)
        if has_solution:
            current_ood_status = False
            current_ood_reasons = ["Known good solution"]
            current_recommended_method = known_method
            current_method_source = "History/Signature"
            apply_filter_method_from_name(known_method)
            logger.info(f"Applying known solution: {known_method}")

            # Vẫn ghi training data để củng cố
            append_filter_training(sensor_data, water_type, known_method)

        else:
            # Trường hợp 3: Thật sự OOD → cần trial
            current_ood_status = True
            current_ood_reasons = ["No known good solution", f"Type: {water_type}"]
            current_method_source = "OOD_TrialRequired"

            if not trial_in_progress:
                logger.info(f"NEW/UNKNOWN water detected → Starting smart trial for '{water_type}'")
                start_smart_trial_v2(sensor_data, water_type, current_water_characteristics)
            else:
                current_recommended_method = "Trial in progress..."
                logger.info("Trial already running...")

    # Cập nhật kết quả xử lý
    with state_lock:
        last_processing_result = {
            'timestamp': datetime.now().isoformat(),
            'sensor': sensor_data.copy(),
            'water_type': water_type,
            'confidence': water_confidence,
            'characteristics': current_water_characteristics,
            'is_ood': current_ood_status,
            'ood_reasons': current_ood_reasons,
            'recommended': current_recommended_method,
            'source': current_method_source,
            'trial_in_progress': trial_in_progress
        }

    logger.info(f"Decision: OOD={current_ood_status} | Method: {current_recommended_method} | Source: {current_method_source}")


def prepare_data_for_colab_training():
    """Chuẩn bị dữ liệu cho training trên Colab"""
    try:
        logger.info("Preparing data for Colab training...")
        
        if not os.path.exists(FILTER_TRAINING_CSV):
            logger.warning("No training data found for Model 2")
            return False
    
        df = pd.read_csv(FILTER_TRAINING_CSV)
        
        if len(df) < 30: 
            logger.warning(f"Not enough training data: {len(df)} samples (need at least 30)")
            return False
        
        dataset_info = {
            'total_samples': len(df),
            'preparation_time': datetime.now().isoformat(),
            'data_files': {
                'filter_training': FILTER_TRAINING_CSV,
                'trial_results': TRIAL_RESULTS_CSV if os.path.exists(TRIAL_RESULTS_CSV) else None,
                'sensor_data': SENSOR_DATA_CSV if os.path.exists(SENSOR_DATA_CSV) else None
            },
            'columns': list(df.columns),
            'unique_methods': df['filter_methods'].unique().tolist() if 'filter_methods' in df.columns else [],
            'unique_water_labels': df['water_label'].unique().tolist() if 'water_label' in df.columns else []
        }
        
        with open('colab_dataset_info.json', 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        with open(DATA_TRAINING_READY_FLAG, 'w') as f:
            f.write(f"Data ready for Colab training at: {datetime.now().isoformat()}\n")
            f.write(f"Total samples: {len(df)}\n")
            f.write(f"Files to upload to Colab:\n")
            f.write(f"- {FILTER_TRAINING_CSV}\n")
            f.write(f"- {TRIAL_RESULTS_CSV} (if exists)\n")
            f.write(f"- colab_dataset_info.json\n")
        
        logger.info(f"Data prepared for training: {len(df)} samples")
        
        return True
        
    except Exception as e:
        logger.error(f"Error preparing data for Colab: {e}")
        return False

sensor_calibration = SensorCalibrationSystem(
    warm_up_seconds=60,      # 1 phút warm-up
    window_size=5,            # Trung bình 5 readings
    stability_threshold=0.05  # CV < 5% = ổn định
)

data_validator = DataValidationSystem()

def automated_control_loop():
    """Control loop chính - xử lý sensor data và hiển thị"""
    global last_sensor_request_time, current_sensor_data, stream_active
    global current_water_type, current_water_confidence
    
    sensor_fail_count = 0
    
    while system_running:
        try:
            current_time = time.time()
            
            # Lấy sensor data mỗi 3 giây
            if current_time - last_sensor_request_time >= 3:
                # Tạm dừng stream nếu đang bật để lấy sensor data
                was_streaming = stream_active
                if was_streaming:
                    stream_active = False
                    time.sleep(0.3)
                
                # Lấy sensor data từ Arduino
                raw_sensor_data = get_sensor_data_from_arduino()
                
                if raw_sensor_data:
                    sensor_fail_count = 0
                    
                    # Cập nhật sensor data
                    with state_lock:
                        current_sensor_data = raw_sensor_data
                    
                    # XỬ LÝ NGAY để hiển thị water_type
                    # KHÔNG chờ trial, luôn xử lý để hiển thị
                    intelligent_process_sensor_data(raw_sensor_data)
                    
                else:
                    sensor_fail_count += 1
                    if sensor_fail_count >= 3:
                        logger.warning(f"Failed to get sensor data {sensor_fail_count} times")
                        
                        # Cập nhật hiển thị lỗi
                        with state_lock:
                            current_water_type = "SENSOR ERROR"
                            current_water_confidence = 0.0
                        
                        sensor_fail_count = 0
                
                last_sensor_request_time = current_time
                
                # Bật lại stream nếu cần
                if was_streaming:
                    time.sleep(0.3)
                    stream_active = True
            
            time.sleep(0.5)  # Giữ cho loop chạy mượt
            
        except Exception as e:
            logger.error(f"Control loop error: {e}")
            time.sleep(2)

def classify_water_with_model1(sensor_data):
    """
    Wrapper dùng model 1: dùng các biến toàn cục interp1, input1_details, output1_details, mean1, scale1, names1
    Trả về: (label:str, confidence:float, probs:np.array)
    """
    global interp1, input1_details, output1_details, mean1, scale1, names1, _prediction_history

    # Kiểm tra điều kiện ban đầu
    if interp1 is None or input1_details is None or output1_details is None or mean1 is None or scale1 is None or names1 is None:
        logger.error("Model1 not initialized (interp1/input1_details/output1_details/mean1/scale1/names1).")
        return "Unknown", 0.0, None

    try:
        # Build feature vector: chắc chắn thứ tự feature giống bên training
        x = np.array([
            sensor_data.get('ph', 0.0),
            sensor_data.get('TDS', 0.0),
            sensor_data.get('turbidity', 0.0),
            sensor_data.get('VOC', 0.0)
        ], dtype=np.float32)

        logger.debug(f"[MODEL1] raw sensor: {x.tolist()}")

        # ensure mean1/scale1 là numpy
        m = np.asarray(mean1, dtype=np.float32)
        s = np.asarray(scale1, dtype=np.float32)

        # align lengths (pad/truncate) nếu cần
        if m.size != x.size or s.size != x.size:
            logger.warning(f"[MODEL1] mean/scale length mismatch (mean={m.size}, scale={s.size}, x={x.size}). Adjusting.")
            target = x.size
            if m.size < target:
                m = np.pad(m, (0, target - m.size), 'constant', constant_values=0.0)
            else:
                m = m[:target]
            if s.size < target:
                s = np.pad(s, (0, target - s.size), 'constant', constant_values=1.0)
            else:
                s = s[:target]

        denom = np.where(s == 0, 1.0, s)
        x_norm = (x - m) / denom
        logger.debug(f"[MODEL1] normalized input: {x_norm.tolist()}")

        # predict
        probs = predict_with_model(interp1, input1_details, output1_details, x_norm.reshape(1, -1))
        if probs is None:
            logger.error("[MODEL1] Prediction returned None")
            return "Unknown", 0.0, None

        logger.debug(f"[MODEL1] probs: {probs}")

        idx = int(np.argmax(probs, axis=1)[0])
        conf = float(np.max(probs, axis=1)[0])
        label = names1[idx] if idx < len(names1) else str(idx)

        # stuck detection: nếu model luôn trả cùng 1 class với conf rất cao trong khi sensor thay đổi
        _prediction_history.append((idx, conf, x.tolist()))
        if len(_prediction_history) == _prediction_history.maxlen:
            same_high = sum(1 for p in _prediction_history if p[0] == idx and p[1] > 0.99)
            if same_high >= int(_prediction_history.maxlen * 0.8):
                sens = np.array([p[2] for p in _prediction_history], dtype=np.float32)
                if np.any(np.var(sens, axis=0) > 1e-4):
                    logger.warning("[MODEL1] Detected stuck predictions despite sensor changes. Degrading confidence.")
                    conf = min(conf, 0.6)

        logger.info(f"[MODEL1] label={label}, conf={conf:.3f}")
        return label, conf, probs

    except Exception as e:
        logger.error(f"[MODEL1] classify error: {e}")
        return "Unknown", 0.0, None

def apply_filter_method_from_name(method_name):
    """Áp dụng phương pháp lọc từ tên"""
    if isinstance(method_name, str) and ',' in method_name:
        # Combo method
        combo_methods = [m.strip() for m in method_name.split(',')]
        apply_filter_combination(combo_methods)
    else:
        # Single method
        apply_filter_method(method_name)

def apply_filter_combination(combo):
    """Áp dụng tổ hợp lọc"""
    relay_set = set()
    for method in combo:
        relay_set |= RELAY_COMMAND_MAP.get(method, set())
    
    command_chars = "".join([ch.upper() if r in relay_set else ch.lower() 
                            for r, ch in RELAY_ORDER])
    send_command_to_arduino(command_chars)

def start_smart_trial_v2(initial_sensor, water_type, characteristics):
    """
    Smart Trial phiên bản 2 - Thông minh, có thể hủy giữa chừng khi phát hiện nước sạch
    """
    global trial_in_progress, current_trial_info, trial_cancel_requested

    def trial_worker():
        global trial_in_progress, current_trial_info, current_recommended_method, current_method_source
        global trial_cancel_requested

        # Đặt cờ bắt đầu trial
        with trial_cancel_lock:
            if trial_cancel_requested:
                trial_cancel_requested = False
                return
            trial_in_progress = True

        logger.info("=== SMART TRIAL V2 STARTED ===")
        logger.info(f"Initial water type: {water_type}")
        logger.info(f"Sensor before trial: pH={initial_sensor.get('ph'):.2f}, TDS={initial_sensor.get('TDS'):.1f}, "
                    f"Turbidity={initial_sensor.get('turbidity'):.2f}, VOC={initial_sensor.get('VOC'):.3f}")

        # Dùng learning_system để đề xuất combo thông minh
        suggested_combos = learning_system.suggest_filter_combinations(initial_sensor, water_type)
        logger.info(f"Smart trial will test {len(suggested_combos)} combinations")

        # Cập nhật UI
        current_trial_info.update({
            "status": "Running",
            "current_trial": 0,
            "total_trials": len(suggested_combos),
            "best_method": "None",
            "best_score": 0.0,
            "progress": "0%"
        })

        best_combo = None
        best_score = -1.0
        best_after_data = None

        try:
            for i, combo in enumerate(suggested_combos):
                # === KIỂM TRA HỦY TRIAL GIỮA CHỪNG ===
                with trial_cancel_lock:
                    if trial_cancel_requested:
                        logger.info("Trial CANCELLED: Clean water detected during trial!")
                        send_command_to_arduino("abcdefg")  # Tắt hết ngay lập tức
                        current_trial_info.update({
                            "status": "Cancelled",
                            "progress": "Cancelled (clean water)",
                            "best_method": "OFF"
                        })
                        current_recommended_method = "OFF"
                        current_method_source = "TrialCancelled_CleanWater"
                        trial_cancel_requested = False
                        trial_in_progress = False
                        return

                current_trial_info["current_trial"] = i + 1
                current_trial_info["progress"] = f"{int((i + 1) / len(suggested_combos) * 100)}%"
                logger.info(f"Trial {i+1}/{len(suggested_combos)}: Testing {combo}")

                # Áp dụng tổ hợp lọc
                if combo == ["OFF"]:
                    send_command_to_arduino("abcdefg")
                    time.sleep(3)
                else:
                    relay_set = set()
                    for method in combo:
                        relay_set |= RELAY_COMMAND_MAP.get(method, set())
                    command_chars = "".join([
                        ch.upper() if r in relay_set else ch.lower()
                        for r, ch in RELAY_ORDER
                    ])
                    send_command_to_arduino(f"CHARS:{command_chars}")
                    time.sleep(TRIAL_STABILIZE_SECONDS)

                # Đọc sensor sau khi lọc
                after_sensor = get_sensor_data_from_arduino()
                if not after_sensor:
                    logger.warning("No sensor data after filter, skipping this combo")
                    continue

                # Đánh giá hiệu suất
                score = learning_system.evaluate_filter_performance(initial_sensor, after_sensor)
                logger.info(f"Combo {combo}, Score: {score:.4f}")

                # Ghi lại kết quả bằng learning_system
                trial_result = learning_system.record_trial_result(
                    sensor_before=initial_sensor,
                    sensor_after=after_sensor,
                    filter_combo=combo,
                    water_type=water_type,
                    performance_score=score
                )

                # Cập nhật best
                if score > best_score:
                    best_score = score
                    best_combo = combo
                    best_after_data = after_sensor
                    current_trial_info["best_method"] = ",".join(combo)
                    current_trial_info["best_score"] = round(score, 4)

        except Exception as e:
            logger.error(f"Error during smart trial: {e}")
        finally:
            # === KẾT THÚC TRIAL ===
            trial_in_progress = False

            if best_combo and best_score > IMPROVEMENT_THRESHOLD:
                method_name = ",".join(best_combo)
                logger.info(f"TRIAL SUCCESS! Best method: {method_name} (Score: {best_score:.4f})")
                apply_filter_combination(best_combo)

                # GHI NHỚ VĨNH VIỄN vào water signature
                update_water_signature(
                    sensor_data=initial_sensor,
                    best_method=method_name,
                    score=best_score,
                    achieved_threshold=True
                )

                current_recommended_method = method_name
                current_method_source = "SmartTrial_Success"
                current_trial_info["status"] = "Completed"

            else:
                logger.warning("Trial failed - No good method found")
                send_command_to_arduino("abcdefg")
                current_recommended_method = "OFF"
                current_method_source = "TrialFailed"
                current_trial_info["status"] = "Failed"

            logger.info("=== SMART TRIAL V2 FINISHED ===")

    # Chạy trong thread riêng
    thread = threading.Thread(target=trial_worker, daemon=True)
    thread.start()
    return True
    
    # Chạy trial
    thread = threading.Thread(target=trial_worker, daemon=True)
    thread.start()
    return True

def intelligent_control_loop():
    """Control loop thông minh tích hợp learning system"""
    global last_sensor_request_time, current_sensor_data, stream_active
    global current_water_type, current_water_confidence, trial_in_progress
    
    sensor_fail_count = 0
    
    while system_running:
        try:
            current_time = time.time()
            
            # Lấy sensor data mỗi 3 giây
            if current_time - last_sensor_request_time >= 3:
                # Tạm dừng stream để lấy sensor data
                was_streaming = stream_active
                if was_streaming:
                    stream_active = False
                    time.sleep(0.3)
                
                raw_sensor_data = get_sensor_data_from_arduino()
                
                if raw_sensor_data:
                    sensor_fail_count = 0
                    
                    # Cập nhật sensor data
                    with state_lock:
                        current_sensor_data = raw_sensor_data
                    
                    # ======== XỬ LÝ THÔNG MINH ========
                    # 1. Phân tích đặc điểm
                    characteristics = analyze_water_characteristics(raw_sensor_data)
                    
                    # 2. Phân loại nước bằng Model 1
                    water_type, water_confidence, _ = classify_water_with_model1(raw_sensor_data)
                    
                    # Cập nhật hiển thị NGAY
                    global current_water_type, current_water_confidence, current_water_characteristics
                    current_water_type = water_type
                    current_water_confidence = water_confidence
                    current_water_characteristics = characteristics
                    
                    logger.info(f"Detected: {water_type} (Confidence: {water_confidence:.2f})")
                    
                    # 3. KIỂM TRA VÀ XỬ LÝ
                    if not trial_in_progress:
                        # Kiểm tra với learning system
                        best_method = learning_system.get_best_method_for_water(raw_sensor_data, water_type)
                        
                        if best_method:
                            # Có phương pháp tốt nhất đã biết
                            logger.info(f"Applying known best method: {best_method}")
                            apply_filter_method_from_name(best_method)
                            
                            current_recommended_method = best_method
                            current_method_source = "Learning System"
                            current_ood_status = False
                        else:
                            # OOD - Cần trial
                            logger.info(f"OOD detected: {water_type} -> Starting smart trial")
                            start_smart_trial_v2(raw_sensor_data, water_type, characteristics)
                            
                    # Lưu vào CSV
                    append_sensor_data(raw_sensor_data)
                    
                else:
                    sensor_fail_count += 1
                    if sensor_fail_count >= 3:
                        logger.warning("Sensor data unavailable")
                        sensor_fail_count = 0
                
                last_sensor_request_time = current_time
                
                # Bật lại stream
                if was_streaming:
                    time.sleep(0.3)
                    stream_active = True
            
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Intelligent control loop error: {e}")
            time.sleep(2)

def signal_handler(sig, frame):
    global system_running
    logger.info("Shutdown signal received")
    system_running = False

def validate_and_clean_training_data():
    """
    Command để validate và clean training data
    Gọi hàm này trước khi train Model 2
    """
    logger.info("Starting training data validation...")
    
    # Validate và clean
    cleaned_df = data_validator.validate_training_data(
        FILTER_TRAINING_CSV,
        output_path=FILTER_TRAINING_CSV + ".cleaned"  # Backup
    )
    
    if cleaned_df is not None:
        # Check xem đã ready chưa
        is_ready, reason, stats = data_validator.check_if_ready_for_training(
            FILTER_TRAINING_CSV + ".cleaned",
            min_samples=100,
            min_classes=3
        )
        
        if is_ready:
            logger.info("Data is READY for Model 2 training!")
            logger.info(f"{stats['total_samples']} samples, {stats['n_classes']} classes, {stats['n_methods']} methods")
            
            # Copy cleaned data để dùng
            import shutil
            shutil.copy(FILTER_TRAINING_CSV + ".cleaned", FILTER_TRAINING_CSV)
            logger.info(f"Cleaned data applied to {FILTER_TRAINING_CSV}")
            
            return True
        else:
            logger.warning(f"Data NOT ready: {reason}")
            if stats:
                logger.info(f"Current stats: {stats}")
            return False
    
    return False

# ================== DISPLAY FUNCTIONS ==================

def create_display_frame():
    """Tạo frame hiển thị với đầy đủ thông tin"""
    display_frame = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
    
    # Lấy thời gian hiện tại
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # ============ PHẦN HIỂN THỊ BÊN TRÁI (VIDEO) ============
    left_x = 20
    left_y = 20
    
    # Tiêu đề
    cv2.putText(display_frame, "ESP32-CAM STREAM", (left_x, left_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Hiển thị video
    with frame_lock:
        video_frame = latest_frame_with_boxes if latest_frame_with_boxes is not None else latest_frame
    
    if video_frame is not None:
        video_height, video_width = video_frame.shape[:2]
        scale_factor = min(500 / video_width, 350 / video_height)
        new_width = int(video_width * scale_factor)
        new_height = int(video_height * scale_factor)
        resized_video = cv2.resize(video_frame, (new_width, new_height))
        display_frame[left_y+30:left_y+30+new_height, left_x:left_x+new_width] = resized_video
    else:
        # Placeholder khi không có video
        placeholder_text = "CAMERA OFFLINE" if not stream_active else "CONNECTING..."
        color = (0, 0, 255) if not stream_active else (0, 255, 255)
        cv2.putText(display_frame, placeholder_text, (left_x + 50, left_y + 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    
    middle_x = 550
    middle_y = 20
    
    # Tiêu đề phần sensor
    cv2.putText(display_frame, "REAL-TIME SENSOR DATA", (middle_x, middle_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Hiển thị sensor data
    with state_lock:
        sensor_info = current_sensor_data
        yolo_count = len(yolo_detections)
    
    sensor_y = middle_y + 40
    if sensor_info and len(sensor_info) > 0:
        # Hiển thị từng giá trị sensor
        sensor_display = [
            (f"pH Level: {sensor_info.get('ph', 0):.2f}", 
             (0, 255, 0) if 6.5 <= sensor_info.get('ph', 0) <= 8.5 else (0, 0, 255)),
            (f"TDS: {sensor_info.get('TDS', 0):.1f} ppm", 
             (0, 255, 0) if sensor_info.get('TDS', 0) < 300 else (0, 165, 255) if sensor_info.get('TDS', 0) < 500 else (0, 0, 255)),
            (f"Turbidity: {sensor_info.get('turbidity', 0):.2f} NTU", 
             (0, 255, 0) if sensor_info.get('turbidity', 0) < 2.0 else (0, 165, 255) if sensor_info.get('turbidity', 0) < 5.0 else (0, 0, 255)),
            (f"VOC: {sensor_info.get('VOC', 0):.3f} mg/L", 
             (0, 255, 0) if sensor_info.get('VOC', 0) < 0.1 else (0, 165, 255) if sensor_info.get('VOC', 0) < 0.5 else (0, 0, 255))
        ]
        
        for i, (text, color) in enumerate(sensor_display):
            cv2.putText(display_frame, text, (middle_x, sensor_y + i * 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        cv2.putText(display_frame, "NO SENSOR DATA", (middle_x, sensor_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Hiển thị YOLO detections
    yolo_y = sensor_y + 140
    cv2.putText(display_frame, f"OBJECTS DETECTED: {yolo_count}", 
                (middle_x, yolo_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    right_x = 850
    right_y = 20
    
    # Tiêu đề phần AI
    cv2.putText(display_frame, "AI WATER ANALYSIS", (right_x, right_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Hiển thị water type và confidence - QUAN TRỌNG
    ai_y = right_y + 40
    cv2.putText(display_frame, f"Water Type: {current_water_type}", 
                (right_x, ai_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.putText(display_frame, f"Confidence: {current_water_confidence:.3f}", 
                (right_x, ai_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                (0, 255, 0) if current_water_confidence > 0.7 else (0, 165, 255) if current_water_confidence > 0.5 else (0, 0, 255), 2)
    
    # Hiển thị characteristics
    if current_water_characteristics:
        chars_y = ai_y + 60
        cv2.putText(display_frame, "Characteristics:", (right_x, chars_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Hiển thị tối đa 4 characteristics
        for i, char in enumerate(current_water_characteristics[:4]):
            cv2.putText(display_frame, f"  • {char}", (right_x, chars_y + 20 + i * 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Hiển thị OOD status
    ood_y = ai_y + 150
    ood_color = (0, 0, 255) if current_ood_status else (0, 255, 0)
    ood_text = "OUT-OF-DISTRIBUTION (NEEDS TRIAL)" if current_ood_status else "NORMAL (KNOWN WATER)"
    cv2.putText(display_frame, ood_text, (right_x, ood_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ood_color, 1)
    
    # Hiển thị recommended method
    method_y = ood_y + 30
    cv2.putText(display_frame, f"Recommended Filter: {current_recommended_method}", 
                (right_x, method_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                (0, 255, 0) if current_recommended_method != "OFF" and "Trial" not in current_recommended_method else (255, 255, 0), 2)
    
    cv2.putText(display_frame, f"Source: {current_method_source}", 
                (right_x, method_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # ============ PHẦN TRIAL INFO ============
    trial_x = 20
    trial_y = 400
    
    if trial_in_progress or current_trial_info["status"] != "Idle":
        status_color = {
            "Running": (0, 255, 255),
            "Completed": (0, 255, 0),
            "Failed": (0, 0, 255),
            "Idle": (128, 128, 128)
        }.get(current_trial_info["status"], (255, 255, 255))
        
        cv2.putText(display_frame, f"Status: {current_trial_info['status']}", 
                    (trial_x, trial_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Hiển thị tiến độ
        if current_trial_info["status"] == "Running":
            progress_y = trial_y + 70
            cv2.putText(display_frame, f"Progress: {current_trial_info['progress']}", 
                        (trial_x, progress_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display_frame, f"Trial: {current_trial_info['current_trial']}/{current_trial_info['total_trials']}", 
                        (trial_x, progress_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Hiển thị kết quả tốt nhất
        if current_trial_info["best_method"] != "None":
            best_y = trial_y + 120
            cv2.putText(display_frame, "Best Result So Far:", 
                        (trial_x, best_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(display_frame, f"Method: {current_trial_info['best_method']}", 
                        (trial_x, best_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(display_frame, f"Score: {current_trial_info['best_score']:.3f}", 
                        (trial_x, best_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                        (0, 255, 0) if current_trial_info['best_score'] > 0.5 else (255, 165, 0), 1)
    
    relay_x = 550
    relay_y = 400
    
    cv2.putText(display_frame, "RELAY STATUS", (relay_x, relay_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Hiển thị từng relay với màu sắc
    relay_y_pos = relay_y + 40
    for i, (relay_name, relay_char) in enumerate(RELAY_ORDER):
        is_on = current_command_chars[i].isupper() if i < len(current_command_chars) else False
        color = (0, 255, 0) if is_on else (100, 100, 100)
        state = "ON" if is_on else "OFF"
        
        # Hiển thị theo 2 cột
        col_offset = 0 if i < 4 else 150
        row_offset = (i % 4) * 25
        
        cv2.putText(display_frame, f"{relay_name} ({relay_char}): {state}", 
                    (relay_x + col_offset, relay_y_pos + row_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Hiển thị filter method name
    cv2.putText(display_frame, f"Active Filter: {current_relay_state}", 
                (relay_x, relay_y_pos + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                (0, 255, 0) if current_relay_state != "OFF" else (255, 255, 255), 2)
    
    # ============ PHẦN SYSTEM INFO ============
    info_x = 20
    info_y = 550
    
    # Hiển thị thông tin hệ thống
    info_lines = [
        f"Time: {current_time_str}",
        f"Water Signatures: {len(water_signatures_cache)}",
        f"Stream: {'ACTIVE' if stream_active else 'INACTIVE'}",
        f"ESP32-CAM IP: {ESP32_CAM_IP}",
        f"Last Process: {datetime.fromtimestamp(last_processing_time).strftime('%H:%M:%S') if last_processing_time > 0 else 'Never'}",
        "Controls: S - Toggle Stream | Q - Quit",
        "Mode: AUTOMATIC TRIAL & LEARNING SYSTEM"
    ]
    
    for i, line in enumerate(info_lines):
        cv2.putText(display_frame, line, (info_x, info_y + 30 + i * 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # ============ PHẦN STATUS BAR ============
    status_bar_y = 780
    status_color = (0, 255, 0) if system_running and test_wifi_connection() else (0, 0, 255)
    status_text = "SYSTEM READY" if system_running and test_wifi_connection() else "SYSTEM ERROR"
    
    # Vẽ background cho status bar
    cv2.rectangle(display_frame, (0, status_bar_y), (DISPLAY_WIDTH, DISPLAY_HEIGHT), (40, 40, 40), -1)
    
    # Hiển thị status
    cv2.putText(display_frame, f"Status: {status_text}", (20, status_bar_y + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    # Hiển thị trial status nếu đang chạy
    if trial_in_progress:
        cv2.putText(display_frame, "TRIAL IN PROGRESS - PLEASE WAIT...", 
                    (300, status_bar_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return display_frame

def display_loop():
    global system_running, stream_active
    
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)
    
    # Bật stream ngay khi khởi động
    stream_active = True
    
    logger.info("Display system started - Press 'S' to toggle stream, 'Q' to quit")
    
    while system_running:
        try:
            # Tạo và hiển thị frame
            display_frame = create_display_frame()
            cv2.imshow(WINDOW_NAME, display_frame)
            
            # Xử lý phím bấm
            key = cv2.waitKey(100) & 0xFF  # 100ms = 10 FPS
            
            if key == ord('q') or key == ord('Q'):
                logger.info("Quit requested by user")
                break
            elif key == ord('s') or key == ord('S'):
                stream_active = not stream_active
                logger.info(f"Stream {'ENABLED' if stream_active else 'DISABLED'}")
            elif key == ord('t') or key == ord('T'):
                # Manual trigger trial (for testing)
                if current_sensor_data and not trial_in_progress:
                    logger.info("Manual trial triggered")
                    start_smart_trial_v2(current_sensor_data, current_water_type, current_water_characteristics)
            
        except Exception as e:
            logger.error(f"Display error: {e}")
            time.sleep(0.1)
    
    stream_active = False
    cv2.destroyAllWindows()
    system_running = False

def start_background_threads():
    threads = []
    
    # 1. Control loop thông minh
    t1 = threading.Thread(target=intelligent_control_loop, daemon=True)
    t1.start()
    threads.append(t1)
    
    time.sleep(1)
    
    # 2. YOLO processing
    t2 = threading.Thread(target=optimized_yolo_processing, daemon=True)
    t2.start()
    threads.append(t2)
    
    time.sleep(1)
    
    # 3. Video stream
    t3 = threading.Thread(target=optimized_video_stream, daemon=True)
    t3.start()
    threads.append(t3)
    
    logger.info(f"Started {len(threads)} background threads")
    return threads

def initialize_system():
    """Khởi tạo hệ thống - THÊM VALIDATION"""
    logger.info("Initializing WIFI Water Filter System...")
    
    ensure_data_files()
    
    # Validate training data nếu tồn tại
    if os.path.exists(FILTER_TRAINING_CSV):
        logger.info("Checking training data quality...")
        validate_and_clean_training_data()
    
    # Load water signatures và distilled representation
    global water_signatures_cache, distilled_representation
    water_signatures_cache = load_water_signatures()
    distilled_representation = load_distilled_representation()
    
    # Khởi tạo các mô hình AI
    initialize_models()
    
    # Khởi tạo Model2 OOD detector
    model2_ood_detector.load_stats()
    logger.info(f"Model2-OOD: {model2_ood_detector.stats['total_predictions']} predictions, "
                f"{model2_ood_detector.get_ood_rate()*100:.1f}% OOD rate")
    
    # Kiểm tra kết nối WiFi
    logger.info("Testing WiFi connection to ESP32-CAM...")
    try:
        response = requests.get(STATUS_URL, timeout=5)
        if response.status_code == 200:
            esp32_status = response.json()
            logger.info(f"Connected to ESP32-CAM: {esp32_status}")
            logger.info("WiFi connection established successfully!")
        else:
            logger.warning(f"Connected to ESP32-CAM but failed to get status (HTTP {response.status_code})")
    except Exception as e:
        logger.error(f"Cannot connect to ESP32-CAM: {e}")
        logger.error("Please check WiFi connection and try again.")
    
    start_background_threads()
    
    logger.info("WiFi Water Filter System initialized successfully!")
    logger.info(f"Colab Data Preparation: {'ENABLED' if AUTO_RETRAIN_MODEL2 else 'DISABLED'}")

def cleanup_system():
    """Dọn dẹp hệ thống khi kết thúc"""
    global system_running
    system_running = False
    
    # Tắt tất cả relay khi kết thúc
    logger.info("Turning off all relays...")
    send_command_to_arduino("abcdefg")
    
    # Lưu Model2 OOD stats
    model2_ood_detector.save_stats()
    
    logger.info("System cleanup completed")

if __name__ == '__main__':
    try:
        initialize_system()
        time.sleep(2)
        display_loop()
        
    except KeyboardInterrupt:
        logger.info("System interrupted by user")
    except Exception as e:
        logger.error(f"System error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_system()
        logger.info("WiFi Water Filter System stopped")