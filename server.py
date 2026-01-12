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
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from collections import deque
import sys
import signal
import socket
import urllib

try:
    import tensorflow as tf
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

# Random Forest Model 2
RANDOM_FOREST_MODEL_PATH = "model2_random_forest.pkl"
RF_MIN_TRAINING_SAMPLES = 30  # Số mẫu tối thiểu để train
RF_RETRAIN_INTERVAL_HOURS = 24  # Retrain mỗi 24 giờ
RF_MIN_NEW_SAMPLES_FOR_RETRAIN = 10  # Số mẫu mới tối thiểu để retrain

AUTO_RETRAIN_MODEL2 = True

# File paths
WATER_DATA_CSV = r'D:\Water Filter\water_data.csv'
SENSOR_DATA_CSV = "sensor_data.csv"
TRIAL_RESULTS_CSV = "trial_results.csv"
FILTER_TRAINING_CSV = "filter_training.csv"
WATER_SIGNATURES_JSON = "water_signatures.json"
DISTILLED_REP_CSV = "distilled_representation.csv"
DATA_TRAINING_READY_FLAG = "data_training_ready.txt"
RF_MODEL_INFO_FILE = "rf_model_info.json"

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
SKIP_TRIAL_LABELS = {'nothing', 'bestwater', 'distilled', 'clean', 'pure'}

# Timing configuration
TRIAL_STABILIZE_SECONDS = 8
APPLY_ACCEPTED_DURATION = 15
SENSOR_READ_INTERVAL = 5
WIFI_RETRY_INTERVAL = 3

OOD_ZSCORE_THRESHOLD = 2.5
WATER_CONFIDENCE_THRESHOLD = 0.7
IMPROVEMENT_THRESHOLD = 0.15
SENSOR_SIMILARITY_THRESHOLD = 0.92

# RF Model 2 thresholds
RF_CONFIDENCE_THRESHOLD = 0.6
RF_UNCERTAINTY_THRESHOLD = 2.0

current_water_type = "Unknown"
current_water_confidence = 0.0
current_water_characteristics = []
current_ood_status = False
current_ood_reasons = []
current_recommended_method = "OFF"
current_method_source = "None"
current_trial_info = {
    "status": "Idle",  # Idle, Running, Completed, Failed, Cancelled
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

# Random Forest Model 2
rf_model2 = None
rf_scaler2 = None
rf_label_encoder2 = None
rf_class_names2 = []
rf_is_trained = False
rf_last_training_time = None
rf_training_samples = 0

# ================== RANDOM FOREST MODEL 2 CLASS ==================
class RandomForestModel2:
    """Model 2 sử dụng Random Forest - HOÀN CHỈNH"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.class_names = []
        self.is_trained = False
        self.last_training_time = None
        self.training_samples = 0
        self.training_score = 0.0
        self.testing_score = 0.0
        
    def load(self):
        """Load model từ file"""
        try:
            if os.path.exists(RANDOM_FOREST_MODEL_PATH):
                model_data = joblib.load(RANDOM_FOREST_MODEL_PATH)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.label_encoder = model_data['label_encoder']
                self.class_names = model_data['class_names']
                self.last_training_time = model_data.get('training_time', datetime.now().isoformat())
                self.training_samples = model_data.get('training_samples', 0)
                self.training_score = model_data.get('train_score', 0.0)
                self.testing_score = model_data.get('test_score', 0.0)
                self.is_trained = True
                
                logger.info(f"Random Forest Model 2 loaded: {len(self.class_names)} classes, {self.training_samples} samples")
                logger.info(f"   Training R²: {self.training_score:.4f}, Testing R²: {self.testing_score:.4f}")
                return True
            else:
                logger.warning("Random Forest Model 2 file not found")
                return False
        except Exception as e:
            logger.error(f"Failed to load Random Forest Model 2: {e}")
            return False
    
    def train(self):
        """Train model từ training data"""
        try:
            if not os.path.exists(FILTER_TRAINING_CSV):
                logger.warning("No training data available")
                return False
            
            df = pd.read_csv(FILTER_TRAINING_CSV)
            
            # Clean data
            df = df.dropna(subset=['pH', 'TDS_ppm', 'turbidity_NTU', 'VOC_mg_L', 'water_label', 'filter_methods'])
            
            if len(df) < RF_MIN_TRAINING_SAMPLES:
                logger.info(f"Insufficient data: {len(df)}/{RF_MIN_TRAINING_SAMPLES} samples")
                return False
            
            logger.info(f"Training Random Forest Model 2 with {len(df)} samples...")
            
            # Prepare features
            X = df[['pH', 'TDS_ppm', 'turbidity_NTU', 'VOC_mg_L']].values
            
            # Encode labels
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(df['filter_methods'])
            self.class_names = self.label_encoder.classes_.tolist()
            
            logger.info(f"   Classes: {self.class_names}")
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train Random Forest
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            self.training_score = self.model.score(X_train_scaled, y_train)
            self.testing_score = self.model.score(X_test_scaled, y_test)
            
            # Save model
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'class_names': self.class_names,
                'training_samples': len(df),
                'training_time': datetime.now().isoformat(),
                'train_score': self.training_score,
                'test_score': self.testing_score,
                'n_estimators': 100,
                'feature_names': ['pH', 'TDS_ppm', 'turbidity_NTU', 'VOC_mg_L']
            }
            
            joblib.dump(model_data, RANDOM_FOREST_MODEL_PATH)
            
            self.is_trained = True
            self.last_training_time = datetime.now().isoformat()
            self.training_samples = len(df)
            
            logger.info(f"Random Forest Model 2 trained successfully!")
            logger.info(f"   Training R²: {self.training_score:.4f}")
            logger.info(f"   Testing R²: {self.testing_score:.4f}")
            logger.info(f"   Saved to: {RANDOM_FOREST_MODEL_PATH}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to train Random Forest Model 2: {e}")
            return False
    
    def predict(self, sensor_data):
        """Dự đoán filter method từ sensor data"""
        if not self.is_trained or self.model is None:
            return None, 0.0, 1.0, True  # Not trained = OOD
        
        try:
            # Prepare input
            features = np.array([
                sensor_data.get('ph', 0),
                sensor_data.get('TDS', 0),
                sensor_data.get('turbidity', 0),
                sensor_data.get('VOC', 0)
            ]).reshape(1, -1)
            
            # Scale
            features_scaled = self.scaler.transform(features)
            
            # Get predictions from all trees
            tree_predictions = []
            for tree in self.model.estimators_:
                pred = tree.predict(features_scaled)[0]
                tree_predictions.append(pred)
            
            tree_predictions = np.array(tree_predictions)
            
            # Calculate statistics
            pred_mean = np.mean(tree_predictions)
            pred_std = np.std(tree_predictions)
            
            # Get predicted class (sử dụng mode của các cây)
            from scipy import stats
            mode_result = stats.mode(np.round(tree_predictions).astype(int))
            predicted_idx = int(mode_result.mode[0] if len(mode_result.mode) > 0 else np.round(pred_mean))
            predicted_idx = np.clip(predicted_idx, 0, len(self.class_names) - 1)
            
            # Calculate confidence (tỷ lệ cây vote cho class này)
            votes_for_class = np.sum(np.round(tree_predictions).astype(int) == predicted_idx)
            confidence = votes_for_class / len(tree_predictions) if len(tree_predictions) > 0 else 0.0
            
            # Get class name
            filter_method = self.class_names[predicted_idx]
            
            # Check OOD
            is_ood = (pred_std > RF_UNCERTAINTY_THRESHOLD) or (confidence < RF_CONFIDENCE_THRESHOLD)
            
            return filter_method, confidence, pred_std, is_ood
            
        except Exception as e:
            logger.error(f"Random Forest prediction error: {e}")
            return None, 0.0, 1.0, True
    
    def should_retrain(self):
        """Kiểm tra xem có nên retrain không"""
        if not self.is_trained:
            return True
        
        # Check time since last training
        if self.last_training_time:
            last_train = datetime.fromisoformat(self.last_training_time)
            hours_since = (datetime.now() - last_train).total_seconds() / 3600
            if hours_since >= RF_RETRAIN_INTERVAL_HOURS:
                logger.info(f"Time to retrain: {hours_since:.1f} hours since last training")
                return True
        
        # Check new data
        if os.path.exists(FILTER_TRAINING_CSV):
            df = pd.read_csv(FILTER_TRAINING_CSV)
            new_samples = len(df) - self.training_samples
            if new_samples >= RF_MIN_NEW_SAMPLES_FOR_RETRAIN:
                logger.info(f"Enough new data to retrain: {new_samples} new samples")
                return True
        
        return False
    
    def get_info(self):
        """Lấy thông tin model"""
        return {
            'is_trained': self.is_trained,
            'n_classes': len(self.class_names),
            'classes': self.class_names,
            'training_samples': self.training_samples,
            'last_training': self.last_training_time,
            'training_score': self.training_score,
            'testing_score': self.testing_score,
            'n_trees': len(self.model.estimators_) if self.model else 0
        }

# Initialize Random Forest Model 2
rf_model2_handler = RandomForestModel2()

# ================== SMART TRIAL LEARNING SYSTEM ==================
class SmartTrialLearningSystem:
    """Hệ thống học từ trial thực tế"""
    
    def __init__(self):
        self.learning_data = []
        self.trial_history = []
        
        self.base_filters = [
            "activated_carbon",  # Than hoạt tính
            "coarse_filter",     # Lọc thô
            "fine_filter",       # Lọc tinh
            "ro_filter",         # RO
            "ultrasonic_filter"  # Siêu âm
        ]
        
        self.filter_specialization = {
            "high_tds": ["ro_filter", "fine_filter"],
            "high_turbidity": ["coarse_filter", "fine_filter", "ultrasonic_filter"],
            "high_voc": ["activated_carbon", "ultrasonic_filter"],
            "low_ph": ["activated_carbon"],
            "high_ph": ["activated_carbon"]
        }
        
    def analyze_water_characteristics(self, sensor_data):
        """Phân tích đặc điểm nước"""
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
        """Đề xuất các tổ hợp lọc"""
        characteristics = self.analyze_water_characteristics(sensor_data)
        
        suggested_combos = []
        suggested_combos.append(["OFF"])
        
        for char in characteristics:
            if char in self.filter_specialization:
                for filter_type in self.filter_specialization[char]:
                    suggested_combos.append([filter_type])
        
        if len(characteristics) >= 2:
            if "high_tds" in characteristics and "high_voc" in characteristics:
                suggested_combos.append(["ro_filter", "activated_carbon"])
            if "high_turbidity" in characteristics and "high_voc" in characteristics:
                suggested_combos.append(["coarse_filter", "activated_carbon"])
            if "high_turbidity" in characteristics and "high_tds" in characteristics:
                suggested_combos.append(["coarse_filter", "ro_filter"])
        
        suggested_combos.append(["coarse_filter", "activated_carbon", "ro_filter"])
        suggested_combos.append(["ultrasonic_filter", "activated_carbon", "fine_filter"])
        
        unique_combos = []
        seen = set()
        for combo in suggested_combos:
            combo_key = ','.join(sorted(combo))
            if combo_key not in seen:
                seen.add(combo_key)
                unique_combos.append(combo)
        
        return unique_combos
    
    def evaluate_filter_performance(self, before_data, after_data):
        """Đánh giá hiệu suất của bộ lọc"""
        try:
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
                    before_score = 1 - abs(before_val - ideal) / 7
                    after_score = 1 - abs(after_val - ideal) / 7
                else:
                    max_val = max(before_val, ideal * 5, 1)
                    before_score = 1 - (before_val / max_val)
                    after_score = 1 - (after_val / max_val)
                
                if before_score > 0:
                    improvement = (after_score - before_score) / before_score
                    scores.append(max(0, improvement))
            
            if scores:
                return float(np.mean(scores))
            return 0.0
            
        except Exception as e:
            logger.error(f"Error evaluating filter performance: {e}")
            return 0.0
    
    def record_trial_result(self, sensor_before, sensor_after, filter_combo, water_type, performance_score):
        """Ghi lại kết quả trial"""
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
        
        if performance_score > 0.3:
            self.add_to_training_data(trial_result)
        
        return trial_result
    
    def add_to_training_data(self, trial_result):
        """Thêm vào training data cho Random Forest"""
        try:
            sensor_before = trial_result['sensor_before']
            filter_combo = trial_result['filter_combo']
            performance = trial_result['performance_score']
            
            if performance > 0.3:
                filter_method_str = ','.join(filter_combo) if isinstance(filter_combo, list) else filter_combo
                
                with csv_lock, open(FILTER_TRAINING_CSV, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        float(sensor_before.get('ph', 0)),
                        float(sensor_before.get('TDS', 0)),
                        float(sensor_before.get('turbidity', 0)),
                        float(sensor_before.get('VOC', 0)),
                        trial_result.get('water_type', 'unknown'),
                        filter_method_str,
                        performance
                    ])
                
                logger.info(f"Added training data: {filter_method_str} (score: {performance:.3f})")
                return True
                
        except Exception as e:
            logger.error(f"Error adding to training data: {e}")
        
        return False
    
    def get_best_method_for_water(self, sensor_data, water_type):
        """Tìm phương pháp tốt nhất cho loại nước này"""
        if not self.trial_history:
            return None
        
        best_score = -1
        best_method = None
        
        for trial in self.trial_history:
            similarity = self.calculate_sensor_similarity(sensor_data, trial['sensor_before'])
            
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
            
            vec1 = np.array(vec1, dtype=np.float32)
            vec2 = np.array(vec2, dtype=np.float32)
            
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

# ================== SENSOR CALIBRATION SYSTEM ==================
class SensorCalibrationSystem:
    """Hệ thống calibration và validation cho sensors"""
    
    def __init__(self, warm_up_seconds=60, window_size=5, stability_threshold=0.05):
        self.warm_up_seconds = warm_up_seconds
        self.window_size = window_size
        self.stability_threshold = stability_threshold
        
        self.system_start_time = datetime.now()
        self.is_warmed_up = False
        
        self.buffers = {
            'ph': deque(maxlen=window_size),
            'TDS': deque(maxlen=window_size),
            'turbidity': deque(maxlen=window_size),
            'VOC': deque(maxlen=window_size)
        }
        
        self.stats = {
            'total_readings': 0,
            'rejected_warmup': 0,
            'rejected_outlier': 0,
            'rejected_unstable': 0,
            'accepted_readings': 0
        }
        
        self.sensor_ranges = {
            'ph': (0.0, 14.0),
            'TDS': (0.0, 2000.0),
            'turbidity': (0.0, 100.0),
            'VOC': (0.0, 10.0)
        }
        
    def is_in_warmup(self):
        elapsed = (datetime.now() - self.system_start_time).total_seconds()
        return elapsed < self.warm_up_seconds
    
    def validate_reading(self, sensor_data):
        self.stats['total_readings'] += 1
        
        if self.is_in_warmup():
            elapsed = (datetime.now() - self.system_start_time).total_seconds()
            remaining = self.warm_up_seconds - elapsed
            self.stats['rejected_warmup'] += 1
            logger.debug(f"Warm-up period: {remaining:.0f}s remaining")
            return False, f"warmup({remaining:.0f}s)", None
        
        if not self.is_warmed_up:
            self.is_warmed_up = True
            logger.info("Sensor warm-up completed!")
        
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
        
        for key in ['ph', 'TDS', 'turbidity', 'VOC']:
            if key in sensor_data:
                self.buffers[key].append(sensor_data[key])
        
        if all(len(self.buffers[key]) >= self.window_size for key in ['ph', 'TDS', 'turbidity', 'VOC']):
            unstable_sensors = []
            
            for key in ['ph', 'TDS', 'turbidity', 'VOC']:
                values = list(self.buffers[key])
                mean = np.mean(values)
                std = np.std(values)
                
                cv = std / (mean + 1e-8) if mean != 0 else 0
                
                if cv > self.stability_threshold:
                    unstable_sensors.append(f"{key}(CV={cv:.3f})")
            
            if unstable_sensors:
                self.stats['rejected_unstable'] += 1
                logger.debug(f"Unstable sensors: {', '.join(unstable_sensors)}")
                return False, f"unstable:{','.join(unstable_sensors)}", None
        
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
        if self.stats['total_readings'] == 0:
            return 0.0
        return self.stats['accepted_readings'] / self.stats['total_readings']
    
    def get_stats_summary(self):
        return {
            'total': self.stats['total_readings'],
            'accepted': self.stats['accepted_readings'],
            'acceptance_rate': self.get_acceptance_rate(),
            'rejected_warmup': self.stats['rejected_warmup'],
            'rejected_outlier': self.stats['rejected_outlier'],
            'rejected_unstable': self.stats['rejected_unstable'],
            'is_warmed_up': self.is_warmed_up
        }

sensor_calibration = SensorCalibrationSystem()

# ================== DATA VALIDATION SYSTEM ==================
class DataValidationSystem:
    """Hệ thống validation cho training data"""
    
    def __init__(self):
        self.validation_stats = {
            'total_samples': 0,
            'invalid_samples': 0,
            'duplicate_samples': 0,
            'nothing_samples': 0,
            'valid_samples': 0
        }
    
    def validate_training_data(self, csv_path, output_path=None):
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Validating {len(df)} training samples...")
            
            self.validation_stats['total_samples'] = len(df)
            original_count = len(df)
            
            nothing_mask = df['water_label'] == 'nothing'
            nothing_samples = df[nothing_mask]
            
            valid_nothing = []
            for idx, row in nothing_samples.iterrows():
                ph_bad = row['pH'] < 4 or row['pH'] > 10
                tds_bad = row['TDS_ppm'] > 500
                turbidity_bad = row['turbidity_NTU'] > 10
                voc_bad = row['VOC_mg_L'] > 1.0
                
                if ph_bad or tds_bad or turbidity_bad or voc_bad:
                    valid_nothing.append(idx)
            
            invalid_nothing = nothing_samples[~nothing_samples.index.isin(valid_nothing)]
            if len(invalid_nothing) > 0:
                logger.warning(f"Removing {len(invalid_nothing)} invalid 'nothing' samples")
                df = df[~df.index.isin(invalid_nothing.index)]
                self.validation_stats['invalid_samples'] += len(invalid_nothing)
            
            self.validation_stats['nothing_samples'] = len(valid_nothing)
            
            sensor_cols = ['pH', 'TDS_ppm', 'turbidity_NTU', 'VOC_mg_L']
            duplicates = df.duplicated(subset=sensor_cols, keep='first')
            if duplicates.sum() > 0:
                logger.warning(f"Removing {duplicates.sum()} duplicate samples")
                df = df[~duplicates]
                self.validation_stats['duplicate_samples'] = duplicates.sum()
            
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
            
            valid_methods = set(RELAY_COMMAND_MAP.keys()) - {'OFF'}
            valid_combos = [','.join(combo) for combo in FILTER_COMBINATIONS]
            all_valid = list(valid_methods) + valid_combos
            
            invalid_methods = ~df['filter_methods'].isin(all_valid)
            if invalid_methods.sum() > 0:
                logger.warning(f"Removing {invalid_methods.sum()} samples with invalid filter methods")
                df = df[~invalid_methods]
                self.validation_stats['invalid_samples'] += invalid_methods.sum()
            
            self.validation_stats['valid_samples'] = len(df)
            
            logger.info("=" * 60)
            logger.info("TRAINING DATA VALIDATION REPORT")
            logger.info("=" * 60)
            logger.info(f"Original samples:        {original_count}")
            logger.info(f"Valid samples:           {len(df)} ({len(df)/original_count*100:.1f}%)")
            logger.info(f"Removed - Invalid:       {self.validation_stats['invalid_samples']}")
            logger.info(f"Removed - Duplicates:    {self.validation_stats['duplicate_samples']}")
            logger.info(f"Valid 'nothing' samples: {self.validation_stats['nothing_samples']}")
            logger.info("=" * 60)
            
            logger.info("\nClass Distribution:")
            water_dist = df['water_label'].value_counts()
            for label, count in water_dist.items():
                logger.info(f"  {label}: {count} samples ({count/len(df)*100:.1f}%)")
            
            logger.info("\nFilter Method Distribution:")
            method_dist = df['filter_methods'].value_counts().head(10)
            for method, count in method_dist.items():
                logger.info(f"  {method}: {count} samples ({count/len(df)*100:.1f}%)")
            
            output_path = output_path or csv_path
            df.to_csv(output_path, index=False)
            logger.info(f"\nCleaned data saved to: {output_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error validating training data: {e}")
            return None
    
    def check_if_ready_for_training(self, csv_path, min_samples=100, min_classes=3):
        try:
            df = pd.read_csv(csv_path)
            
            if len(df) < min_samples:
                return False, f"Not enough samples: {len(df)}/{min_samples}", None
            
            n_classes = df['water_label'].nunique()
            if n_classes < min_classes:
                return False, f"Not enough classes: {n_classes}/{min_classes}", None
            
            class_counts = df['water_label'].value_counts()
            min_class_count = class_counts.min()
            max_class_count = class_counts.max()
            imbalance_ratio = max_class_count / (min_class_count + 1e-8)
            
            if imbalance_ratio > 10:
                return False, f"Severe class imbalance: {imbalance_ratio:.1f}x", None
            
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

data_validator = DataValidationSystem()

# ================== NETWORK FUNCTIONS ==================
class TimeoutSession(requests.Session):
    def __init__(self, default_timeout=(2, 5)):
        super().__init__()
        self.default_timeout = default_timeout
    
    def request(self, method, url, **kwargs):
        if 'timeout' not in kwargs:
            kwargs['timeout'] = self.default_timeout
        return super().request(method, url, **kwargs)

def create_optimized_session():
    session = TimeoutSession(default_timeout=(2, 3))
    
    retry_strategy = Retry(
        total=1,
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
    """Get sensor data từ Arduino Uno"""
    try:
        response = esp32_session.get(SENSOR_DATA_URL, timeout=2)
        
        if response.status_code == 200:
            sensor_data = response.json()
            
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
            
            logger.info(f"Raw sensor data: {sensor_data}")
            logger.info(f"Processed sensor data: {processed_data}")
            
            return processed_data
            
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
    """Gửi command đến Arduino"""
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

def apply_filter_method(method_name):
    """Áp dụng filter method"""
    if method_name in RELAY_COMMAND_MAP:
        relay_set = RELAY_COMMAND_MAP[method_name]
        command_chars = "".join([ch.upper() if r in relay_set else ch.lower() for r, ch in RELAY_ORDER])
        success = send_command_to_arduino(command_chars)
        
        if success:
            global current_command_chars, current_relay_state
            with state_lock:
                current_command_chars = command_chars
                current_relay_state = method_name
            logger.info(f"Applied filter method: {method_name}")
            return True
        else:
            logger.error(f"Failed to apply filter method: {method_name}")
            return False
    else:
        logger.warning(f"Unknown filter method: {method_name}")
        return False

# ================== VIDEO STREAM FUNCTIONS ==================
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
                    ret, frame = cap.read()
                    if not ret:
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
        (FILTER_TRAINING_CSV, ["timestamp", "pH", "TDS_ppm", "turbidity_NTU", "VOC_mg_L", "water_label", "filter_methods", "performance"]),
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
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with csv_lock, open(FILTER_TRAINING_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                float(sensor_data.get('ph', 0.0)),
                float(sensor_data.get('TDS', sensor_data.get('tds', 0.0))),
                float(sensor_data.get('turbidity', 0.0)),
                float(sensor_data.get('VOC', sensor_data.get('voc', 0.0))),
                str(water_label),
                str(filter_method),
                1.0  # Performance mặc định cho positive sample
            ])
        logger.info(f"Added training data: {water_label} -> {filter_method}")
        
        # Trigger retrain check
        check_and_retrain_rf_model()
        
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
    """Load TFLite model (Model 1)"""
    try:
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            return None, None, None
            
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
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
    """Dự đoán với TFLite model"""
    data = np.asarray(normalized_input, dtype=np.float32)
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)

    in_quant = input_details[0].get('quantization', (0.0,0))
    out_quant = output_details[0].get('quantization', (0.0,0))
    in_scale, in_zero = (in_quant if isinstance(in_quant, (list,tuple)) else (0.0,0))
    out_scale, out_zero = (out_quant if isinstance(out_quant, (list,tuple)) else (0.0,0))

    try:
        with _interpreter_lock1:
            if in_scale and in_scale != 0:
                q = np.round(data / in_scale + in_zero).astype(input_details[0]['dtype'])
                interpreter.set_tensor(input_details[0]['index'], q)
            else:
                interpreter.set_tensor(input_details[0]['index'], data.astype(input_details[0]['dtype']))

            interpreter.invoke()
            raw_out = interpreter.get_tensor(output_details[0]['index']).copy()

        if out_scale and out_scale != 0:
            pred = (raw_out.astype(np.float32) - out_zero) * out_scale
        else:
            pred = raw_out.astype(np.float32)

        if pred.ndim == 1:
            pred = np.expand_dims(pred, axis=0)

        logger.debug(f"_tflite_predict raw_out = {raw_out}")
        logger.debug(f"_tflite_predict dequantized = {pred}")

        if apply_softmax:
            if not np.allclose(pred.sum(axis=1), 1.0, atol=1e-3):
                e = np.exp(pred - np.max(pred, axis=1, keepdims=True))
                pred = e / e.sum(axis=1, keepdims=True)

        return pred

    except Exception as e:
        logger.exception(f"_tflite_predict error: {e}")
        return None

def classify_water_with_model1(sensor_data):
    """Phân loại nước bằng Model 1"""
    global interp1, input1_details, output1_details, mean1, scale1, names1, _prediction_history

    if interp1 is None or input1_details is None or output1_details is None or mean1 is None or scale1 is None or names1 is None:
        logger.error("Model1 not initialized")
        return "Unknown", 0.0, None

    try:
        x = np.array([
            sensor_data.get('ph', 0.0),
            sensor_data.get('TDS', 0.0),
            sensor_data.get('turbidity', 0.0),
            sensor_data.get('VOC', 0.0)
        ], dtype=np.float32)

        logger.debug(f"[MODEL1] raw sensor: {x.tolist()}")

        m = np.asarray(mean1, dtype=np.float32)
        s = np.asarray(scale1, dtype=np.float32)

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

        probs = predict_with_model(interp1, input1_details, output1_details, x_norm.reshape(1, -1))
        if probs is None:
            logger.error("[MODEL1] Prediction returned None")
            return "Unknown", 0.0, None

        logger.debug(f"[MODEL1] probs: {probs}")

        idx = int(np.argmax(probs, axis=1)[0])
        conf = float(np.max(probs, axis=1)[0])
        label = names1[idx] if idx < len(names1) else str(idx)

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

def initialize_models():
    """Khởi tạo tất cả mô hình AI"""
    global interp1, input1_details, output1_details, mean1, scale1, names1
    
    load_yolo_model()
    
    if TF_AVAILABLE:
        interp1, input1_details, output1_details = load_tflite_model(MLP_MODEL_1_TFLITE_PATH)
        if interp1:
            mean1, scale1, names1 = load_model_params(MODEL_1_PARAMS_PATH)
            logger.info(f"Model 1 loaded successfully - Names: {names1}")

# ================== WATER SIGNATURES FUNCTIONS ==================
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
        features1 = [sensor_data1.get('ph', 0), sensor_data1.get('TDS', 0), 
                    sensor_data1.get('turbidity', 0), sensor_data1.get('VOC', 0)]
        features2 = [sensor_data2.get('ph', 0), sensor_data2.get('TDS', 0), 
                    sensor_data2.get('turbidity', 0), sensor_data2.get('VOC', 0)]
        
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

# ================== RANDOM FOREST MODEL 2 FUNCTIONS ==================
def check_and_retrain_rf_model():
    """Kiểm tra và retrain Random Forest model nếu cần"""
    try:
        if not AUTO_RETRAIN_MODEL2:
            return
        
        if rf_model2_handler.should_retrain():
            logger.info("Random Forest Model 2 needs retraining...")
            
            def retrain_worker():
                try:
                    success = rf_model2_handler.train()
                    if success:
                        logger.info("Random Forest Model 2 retrained successfully!")
                    else:
                        logger.warning("Random Forest Model 2 retraining failed")
                except Exception as e:
                    logger.error(f"Error in retrain worker: {e}")
            
            thread = threading.Thread(target=retrain_worker, daemon=True)
            thread.start()
            
    except Exception as e:
        logger.error(f"Error checking retrain: {e}")

def initialize_rf_model():
    """Khởi tạo Random Forest Model 2"""
    logger.info("Initializing Random Forest Model 2...")
    
    # Thử load model đã train
    loaded = rf_model2_handler.load()
    
    if not loaded:
        logger.info("No trained model found, attempting to train...")
        
        # Kiểm tra xem có đủ data để train không
        if os.path.exists(FILTER_TRAINING_CSV):
            df = pd.read_csv(FILTER_TRAINING_CSV)
            if len(df) >= RF_MIN_TRAINING_SAMPLES:
                logger.info(f"Found {len(df)} training samples, training model...")
                success = rf_model2_handler.train()
                if success:
                    logger.info("Random Forest Model 2 trained successfully!")
                else:
                    logger.warning("Random Forest Model 2 training failed")
            else:
                logger.info(f"Not enough training samples: {len(df)}/{RF_MIN_TRAINING_SAMPLES}")
        else:
            logger.info("No training data file found")
    
    # Log model info
    model_info = rf_model2_handler.get_info()
    logger.info(f"Random Forest Model 2 Status: {'TRAINED' if model_info['is_trained'] else 'NOT TRAINED'}")
    if model_info['is_trained']:
        logger.info(f"  - Classes: {model_info['n_classes']}")
        logger.info(f"  - Training samples: {model_info['training_samples']}")
        logger.info(f"  - Last training: {model_info['last_training']}")

# ================== TRIAL FUNCTIONS ==================
def should_skip_trial(water_label: str) -> bool:
    """Kiểm tra xem có nên bỏ qua trial không"""
    if not water_label:
        return False
    label = str(water_label).strip().lower()
    return label in SKIP_TRIAL_LABELS

def has_good_known_solution(sensor_data, water_type):
    """Kiểm tra xem có phương pháp tốt đã biết cho loại nước này chưa"""
    if not learning_system.trial_history:
        return False, None
    
    best_from_history = learning_system.get_best_method_for_water(sensor_data, water_type)
    if best_from_history and learning_system.calculate_sensor_similarity(
        sensor_data, learning_system.trial_history[-1]['sensor_before'] if learning_system.trial_history else sensor_data
    ) > 0.85:
        return True, best_from_history

    similar_sig = find_similar_water_signature(sensor_data, threshold=SENSOR_SIMILARITY_THRESHOLD)
    if similar_sig and similar_sig.get('achieved_threshold', False):
        return True, similar_sig['best_method']
    
    return False, None

def apply_filter_method_from_name(method_name):
    """Áp dụng phương pháp lọc từ tên"""
    if isinstance(method_name, str) and ',' in method_name:
        combo_methods = [m.strip() for m in method_name.split(',')]
        apply_filter_combination(combo_methods)
    else:
        apply_filter_method(method_name)

def apply_filter_combination(combo):
    """Áp dụng tổ hợp lọc"""
    relay_set = set()
    for method in combo:
        relay_set |= RELAY_COMMAND_MAP.get(method, set())
    
    command_chars = "".join([ch.upper() if r in relay_set else ch.lower() for r, ch in RELAY_ORDER])
    send_command_to_arduino(command_chars)

def start_smart_trial_v2(initial_sensor, water_type, characteristics):
    """Smart Trial phiên bản 2"""
    global trial_in_progress, current_trial_info, trial_cancel_requested

    def trial_worker():
        global trial_in_progress, current_trial_info, current_recommended_method, current_method_source
        global trial_cancel_requested

        with trial_cancel_lock:
            if trial_cancel_requested:
                trial_cancel_requested = False
                return
            trial_in_progress = True

        logger.info("=== SMART TRIAL V2 STARTED ===")
        logger.info(f"Initial water type: {water_type}")
        logger.info(f"Sensor before trial: pH={initial_sensor.get('ph'):.2f}, TDS={initial_sensor.get('TDS'):.1f}, "
                    f"Turbidity={initial_sensor.get('turbidity'):.2f}, VOC={initial_sensor.get('VOC'):.3f}")

        suggested_combos = learning_system.suggest_filter_combinations(initial_sensor, water_type)
        logger.info(f"Smart trial will test {len(suggested_combos)} combinations")

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
                with trial_cancel_lock:
                    if trial_cancel_requested:
                        logger.info("Trial CANCELLED: Clean water detected during trial!")
                        send_command_to_arduino("abcdefg")
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

                after_sensor = get_sensor_data_from_arduino()
                if not after_sensor:
                    logger.warning("No sensor data after filter, skipping this combo")
                    continue

                score = learning_system.evaluate_filter_performance(initial_sensor, after_sensor)
                logger.info(f"Combo {combo}, Score: {score:.4f}")

                trial_result = learning_system.record_trial_result(
                    sensor_before=initial_sensor,
                    sensor_after=after_sensor,
                    filter_combo=combo,
                    water_type=water_type,
                    performance_score=score
                )

                if score > best_score:
                    best_score = score
                    best_combo = combo
                    best_after_data = after_sensor
                    current_trial_info["best_method"] = ",".join(combo)
                    current_trial_info["best_score"] = round(score, 4)

        except Exception as e:
            logger.error(f"Error during smart trial: {e}")
        finally:
            trial_in_progress = False

            if best_combo and best_score > IMPROVEMENT_THRESHOLD:
                method_name = ",".join(best_combo)
                logger.info(f"TRIAL SUCCESS! Best method: {method_name} (Score: {best_score:.4f})")
                apply_filter_combination(best_combo)

                update_water_signature(
                    sensor_data=initial_sensor,
                    best_method=method_name,
                    improvement_score=best_score,
                    achieved_threshold=True
                )

                current_recommended_method = method_name
                current_method_source = "SmartTrial_Success"
                current_trial_info["status"] = "Completed"

                # Add to training data
                append_filter_training(initial_sensor, water_type, method_name)
                
            else:
                logger.warning("Trial failed - No good method found")
                send_command_to_arduino("abcdefg")
                current_recommended_method = "OFF"
                current_method_source = "TrialFailed"
                current_trial_info["status"] = "Failed"

            logger.info("=== SMART TRIAL V2 FINISHED ===")

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
    
    if ph < 6.0:
        characteristics.append("Acidic")
    elif ph > 8.5:
        characteristics.append("Alkaline")
    else:
        characteristics.append("Neutral pH")
    
    if tds < 50:
        characteristics.append("Soft Water")
    elif tds < 150:
        characteristics.append("Medium Hardness")
    elif tds < 300:
        characteristics.append("Hard Water")
    else:
        characteristics.append("Very Hard Water")
    
    if turbidity < 1.0:
        characteristics.append("Clear")
    elif turbidity < 5.0:
        characteristics.append("Slightly Turbid")
    elif turbidity < 10.0:
        characteristics.append("Turbid")
    else:
        characteristics.append("Very Turbid")
    
    if voc < 0.1:
        characteristics.append("Low VOC")
    elif voc < 0.5:
        characteristics.append("Medium VOC")
    else:
        characteristics.append("High VOC")
    
    return characteristics

# ================== DISTILLED REPRESENTATION FUNCTIONS ==================
def calculate_distilled_representation():
    """Tính toán distilled water representation"""
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
        return None

# ================== MAIN PROCESSING FUNCTION ==================
def intelligent_process_sensor_data(sensor_data):
    """
    XỬ LÝ THÔNG MINH - TÍCH HỢP RANDOM FOREST MODEL 2
    """
    global current_water_type, current_water_confidence, current_water_characteristics
    global current_ood_status, current_ood_reasons, current_recommended_method, current_method_source
    global trial_in_progress, last_processing_result, rf_model2_handler
    global last_processing_time

    if not sensor_data:
        return

    last_processing_time = time.time()
    
    logger.info(f"Processing sensor: pH={sensor_data.get('ph','?'):.2f}, "
                f"TDS={sensor_data.get('TDS','?'):.1f}, Turb={sensor_data.get('turbidity','?'):.2f}, "
                f"VOC={sensor_data.get('VOC','?'):.3f}")

    # 1. Phân tích đặc điểm nước
    current_water_characteristics = analyze_water_characteristics(sensor_data)

    # 2. Phân loại bằng Model 1
    water_type, water_confidence, _ = classify_water_with_model1(sensor_data)
    current_water_type = water_type
    current_water_confidence = water_confidence

    logger.info(f"Model 1 - Detected: '{water_type}' (conf: {water_confidence:.3f})")
    logger.info(f"Characteristics: {current_water_characteristics}")

    # ================== LOGIC QUYẾT ĐỊNH VỚI RANDOM FOREST ==================
    
    # Trường hợp 1: Nước sạch → bỏ qua hoàn toàn
    if should_skip_trial(water_type):
        logger.info(f"Detected CLEAN water ('{water_type}') → Turning OFF all filters")

        current_ood_status = False
        current_recommended_method = "OFF"
        current_method_source = "CleanWater"
        current_ood_reasons = [f"Label '{water_type}' → no filtering needed"]

        apply_filter_method("OFF")

        if trial_in_progress:
            with trial_cancel_lock:
                trial_cancel_requested = True
            logger.info("Trial cancelled due to clean water detection!")

        return
    
    # Trường hợp 2: Sử dụng Random Forest Model 2
    if rf_model2_handler.is_trained:
        # Predict với Random Forest
        filter_method, confidence, uncertainty, is_ood = rf_model2_handler.predict(sensor_data)
        
        logger.info(f"Random Forest Prediction: method={filter_method}, "
                   f"conf={confidence:.3f}, uncertainty={uncertainty:.3f}, is_ood={is_ood}")
        
        if filter_method is None:
            # Model 2 lỗi
            logger.warning("Random Forest Model 2 prediction error")
            current_ood_status = True
            current_ood_reasons = ["RF Model error"]
            current_recommended_method = "Trial needed"
            current_method_source = "RF_Error"
            
            if not trial_in_progress:
                logger.info("Starting trial due to RF model error...")
                start_smart_trial_v2(sensor_data, water_type, current_water_characteristics)
        
        elif is_ood:
            # Model 2 phát hiện OOD
            logger.info(f"Random Forest - OOD detected!")
            
            current_ood_status = True
            current_ood_reasons = [
                f"High uncertainty ({uncertainty:.3f})",
                f"Low confidence ({confidence:.3f})",
                f"Water type: {water_type}"
            ]
            current_recommended_method = "Trial needed"
            current_method_source = "RF_OOD_Detection"
            
            if not trial_in_progress:
                logger.info("Starting smart trial for OOD sample...")
                start_smart_trial_v2(sensor_data, water_type, current_water_characteristics)
        
        else:
            # Model 2 tự tin dự đoán
            logger.info(f"Random Forest - Confident prediction: {filter_method}")
            
            current_ood_status = False
            current_ood_reasons = ["Confident RF prediction"]
            current_recommended_method = filter_method
            current_method_source = "RandomForest"
            
            # Áp dụng filter method
            apply_filter_method_from_name(filter_method)
            
            # CHỈ GHI TRAINING DATA KHI KHÔNG PHẢI "OFF" hoặc "Trial needed"
            if filter_method and filter_method != "OFF" and filter_method != "Trial needed":
                logger.info(f"Saving to training data: {water_type} -> {filter_method}")
                append_filter_training(sensor_data, water_type, filter_method)
            else:
                logger.info(f"⏭Skipping training data for method: {filter_method}")
    
    else:
        # Model 2 chưa được train
        logger.warning("Random Forest Model 2 not trained yet")
        
        # Kiểm tra xem có solution đã biết không
        has_solution, known_method = has_good_known_solution(sensor_data, water_type)
        
        if has_solution:
            logger.info(f"Using known solution: {known_method}")
            current_ood_status = False
            current_ood_reasons = ["Known solution from history"]
            current_recommended_method = known_method
            current_method_source = "History"
            
            apply_filter_method_from_name(known_method)
            
            # CHỈ GHI TRAINING DATA KHI KHÔNG PHẢI "OFF"
            if known_method and known_method != "OFF" and known_method != "Trial needed":
                append_filter_training(sensor_data, water_type, known_method)
            
            # Kiểm tra xem có đủ data để train model chưa
            check_and_retrain_rf_model()
            
        else:
            # Không có solution đã biết, cần trial
            logger.info("No known solution, starting trial...")
            current_ood_status = True
            current_ood_reasons = ["No RF model & no known solution"]
            current_recommended_method = "Trial in progress"
            current_method_source = "Starting_Trial"
            
            if not trial_in_progress:
                start_smart_trial_v2(sensor_data, water_type, current_water_characteristics)

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

def intelligent_control_loop():
    """Control loop thông minh tích hợp Random Forest"""
    global last_sensor_request_time, current_sensor_data, stream_active
    global current_water_type, current_water_confidence, trial_in_progress
    
    sensor_fail_count = 0
    
    logger.info("Intelligent Control Loop started")
    
    while system_running:
        try:
            current_time = time.time()
            
            if current_time - last_sensor_request_time >= 3:
                was_streaming = stream_active
                if was_streaming:
                    stream_active = False
                    time.sleep(0.3)
                
                raw_sensor_data = get_sensor_data_from_arduino()
                
                if raw_sensor_data:
                    sensor_fail_count = 0
                    
                    with state_lock:
                        current_sensor_data = raw_sensor_data
                    
                    logger.info(f"New sensor data received, processing...")
                    
                    # Xử lý sensor data
                    intelligent_process_sensor_data(raw_sensor_data)
                    
                    logger.info(f"After processing: Type={current_water_type}, "
                              f"OOD={current_ood_status}, Method={current_recommended_method}, "
                              f"Source={current_method_source}")
                    
                    # Lưu vào CSV
                    append_sensor_data(raw_sensor_data)
                    
                else:
                    sensor_fail_count += 1
                    if sensor_fail_count >= 3:
                        logger.warning("⚠️ Sensor data unavailable")
                        sensor_fail_count = 0
                
                last_sensor_request_time = current_time
                
                if was_streaming:
                    time.sleep(0.3)
                    stream_active = True
            
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Control loop error: {e}")
            time.sleep(2)

# ================== DISPLAY FUNCTIONS ==================
def create_display_frame():
    """Tạo frame hiển thị với đầy đủ thông tin"""
    display_frame = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
    
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # ============ PHẦN HIỂN THỊ BÊN TRÁI (VIDEO) ============
    left_x = 20
    left_y = 20
    
    cv2.putText(display_frame, "ESP32-CAM STREAM", (left_x, left_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
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
        placeholder_text = "CAMERA OFFLINE" if not stream_active else "CONNECTING..."
        color = (0, 0, 255) if not stream_active else (0, 255, 255)
        cv2.putText(display_frame, placeholder_text, (left_x + 50, left_y + 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    
    middle_x = 550
    middle_y = 20
    
    cv2.putText(display_frame, "REAL-TIME SENSOR DATA", (middle_x, middle_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    with state_lock:
        sensor_info = current_sensor_data
        yolo_count = len(yolo_detections)
    
    sensor_y = middle_y + 40
    if sensor_info and len(sensor_info) > 0:
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
    
    yolo_y = sensor_y + 140
    cv2.putText(display_frame, f"OBJECTS DETECTED: {yolo_count}", 
                (middle_x, yolo_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    right_x = 850
    right_y = 20
    
    cv2.putText(display_frame, "AI WATER ANALYSIS", (right_x, right_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    ai_y = right_y + 40
    cv2.putText(display_frame, f"Water Type: {current_water_type}", 
                (right_x, ai_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.putText(display_frame, f"Confidence: {current_water_confidence:.3f}", 
                (right_x, ai_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                (0, 255, 0) if current_water_confidence > 0.7 else (0, 165, 255) if current_water_confidence > 0.5 else (0, 0, 255), 2)
    
    if current_water_characteristics:
        chars_y = ai_y + 60
        cv2.putText(display_frame, "Characteristics:", (right_x, chars_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        for i, char in enumerate(current_water_characteristics[:4]):
            cv2.putText(display_frame, f"  • {char}", (right_x, chars_y + 20 + i * 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    ood_y = ai_y + 150
    ood_color = (0, 0, 255) if current_ood_status else (0, 255, 0)
    ood_text = "OUT-OF-DISTRIBUTION (NEEDS TRIAL)" if current_ood_status else "NORMAL (KNOWN WATER)"
    cv2.putText(display_frame, ood_text, (right_x, ood_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ood_color, 1)
    
    method_y = ood_y + 30
    cv2.putText(display_frame, f"Recommended Filter: {current_recommended_method}", 
                (right_x, method_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                (0, 255, 0) if current_recommended_method != "OFF" and "Trial" not in current_recommended_method else (255, 255, 0), 2)
    
    cv2.putText(display_frame, f"Source: {current_method_source}", 
                (right_x, method_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # ============ PHẦN RANDOM FOREST MODEL 2 INFO ============
    rf_x = 550
    rf_y = 250
    
    rf_info = rf_model2_handler.get_info()
    cv2.putText(display_frame, "RANDOM FOREST MODEL 2", (rf_x, rf_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    status_color = (0, 255, 0) if rf_info['is_trained'] else (0, 0, 255)
    status_text = "TRAINED" if rf_info['is_trained'] else "NOT TRAINED"
    cv2.putText(display_frame, f"Status: {status_text}", 
                (rf_x, rf_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
    
    if rf_info['is_trained']:
        cv2.putText(display_frame, f"Classes: {rf_info['n_classes']}", 
                    (rf_x, rf_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(display_frame, f"Samples: {rf_info['training_samples']}", 
                    (rf_x, rf_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(display_frame, f"Trees: {rf_info['n_trees']}", 
                    (rf_x, rf_y + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # ============ PHẦN TRIAL INFO ============
    trial_x = 20
    trial_y = 400
    
    if trial_in_progress or current_trial_info["status"] != "Idle":
        status_color = {
            "Running": (0, 255, 255),
            "Completed": (0, 255, 0),
            "Failed": (0, 0, 255),
            "Cancelled": (255, 165, 0),
            "Idle": (128, 128, 128)
        }.get(current_trial_info["status"], (255, 255, 255))
        
        cv2.putText(display_frame, f"Status: {current_trial_info['status']}", 
                    (trial_x, trial_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        if current_trial_info["status"] == "Running":
            progress_y = trial_y + 70
            cv2.putText(display_frame, f"Progress: {current_trial_info['progress']}", 
                        (trial_x, progress_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display_frame, f"Trial: {current_trial_info['current_trial']}/{current_trial_info['total_trials']}", 
                        (trial_x, progress_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
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
    
    relay_y_pos = relay_y + 40
    for i, (relay_name, relay_char) in enumerate(RELAY_ORDER):
        is_on = current_command_chars[i].isupper() if i < len(current_command_chars) else False
        color = (0, 255, 0) if is_on else (100, 100, 100)
        state = "ON" if is_on else "OFF"
        
        col_offset = 0 if i < 4 else 150
        row_offset = (i % 4) * 25
        
        cv2.putText(display_frame, f"{relay_name} ({relay_char}): {state}", 
                    (relay_x + col_offset, relay_y_pos + row_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.putText(display_frame, f"Active Filter: {current_relay_state}", 
                (relay_x, relay_y_pos + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                (0, 255, 0) if current_relay_state != "OFF" else (255, 255, 255), 2)
    
    # ============ PHẦN SYSTEM INFO ============
    info_x = 20
    info_y = 550
    
    info_lines = [
        f"Time: {current_time_str}",
        f"Water Signatures: {len(water_signatures_cache)}",
        f"Stream: {'ACTIVE' if stream_active else 'INACTIVE'}",
        f"ESP32-CAM IP: {ESP32_CAM_IP}",
        f"Last Process: {datetime.fromtimestamp(last_processing_time).strftime('%H:%M:%S') if last_processing_time > 0 else 'Never'}",
        "Controls: S - Toggle Stream | Q - Quit",
        "Mode: AUTOMATIC TRIAL & RANDOM FOREST LEARNING"
    ]
    
    for i, line in enumerate(info_lines):
        cv2.putText(display_frame, line, (info_x, info_y + 30 + i * 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    status_bar_y = 780
    cv2.rectangle(display_frame, (0, status_bar_y), (DISPLAY_WIDTH, DISPLAY_HEIGHT), (40, 40, 40), -1)
    
    status_color = (0, 255, 0) if system_running else (0, 0, 255)
    status_text = "SYSTEM READY" if system_running else "SYSTEM STOPPED"
    
    cv2.putText(display_frame, f"Status: {status_text}", (20, status_bar_y + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    if trial_in_progress:
        cv2.putText(display_frame, "TRIAL IN PROGRESS - PLEASE WAIT...", 
                    (300, status_bar_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return display_frame

def display_loop():
    global system_running, stream_active
    
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)
    
    stream_active = True
    
    logger.info("Display system started - Press 'S' to toggle stream, 'Q' to quit")
    
    while system_running:
        try:
            display_frame = create_display_frame()
            cv2.imshow(WINDOW_NAME, display_frame)
            
            key = cv2.waitKey(100) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                logger.info("Quit requested by user")
                break
            elif key == ord('s') or key == ord('S'):
                stream_active = not stream_active
                logger.info(f"Stream {'ENABLED' if stream_active else 'DISABLED'}")
            elif key == ord('t') or key == ord('T'):
                if current_sensor_data and not trial_in_progress:
                    logger.info("Manual trial triggered")
                    start_smart_trial_v2(current_sensor_data, current_water_type, current_water_characteristics)
            elif key == ord('r') or key == ord('R'):
                # Manual retrain RF model
                logger.info("Manual retrain requested")
                check_and_retrain_rf_model()
            
        except Exception as e:
            logger.error(f"Display error: {e}")
            time.sleep(0.1)
    
    stream_active = False
    cv2.destroyAllWindows()
    system_running = False

def start_background_threads():
    threads = []
    
    t1 = threading.Thread(target=intelligent_control_loop, daemon=True)
    t1.start()
    threads.append(t1)
    
    time.sleep(1)
    
    t2 = threading.Thread(target=optimized_yolo_processing, daemon=True)
    t2.start()
    threads.append(t2)
    
    time.sleep(1)
    
    t3 = threading.Thread(target=optimized_video_stream, daemon=True)
    t3.start()
    threads.append(t3)
    
    logger.info(f"Started {len(threads)} background threads")
    return threads

def initialize_system():
    """Khởi tạo hệ thống"""
    logger.info("Initializing WIFI Water Filter System...")
    
    ensure_data_files()
    
    if os.path.exists(FILTER_TRAINING_CSV):
        logger.info("Checking training data quality...")
        data_validator.validate_training_data(FILTER_TRAINING_CSV)
    
    global water_signatures_cache, distilled_representation
    water_signatures_cache = load_water_signatures()
    distilled_representation = load_distilled_representation()
    
    initialize_models()
    
    initialize_rf_model()
    
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
    logger.info(f"Random Forest Auto-retrain: {'ENABLED' if AUTO_RETRAIN_MODEL2 else 'DISABLED'}")
    
    model_info = rf_model2_handler.get_info()
    if model_info['is_trained']:
        logger.info(f"Random Forest Model 2: {model_info['n_classes']} classes, {model_info['training_samples']} samples")
    else:
        logger.info("Random Forest Model 2: Not trained yet - will train when enough data")

def cleanup_system():
    """Dọn dẹp hệ thống khi kết thúc"""
    global system_running
    system_running = False
    
    logger.info("Turning off all relays...")
    send_command_to_arduino("abcdefg")
    
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