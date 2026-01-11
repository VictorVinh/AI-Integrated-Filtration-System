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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
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
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Thêm import cho Model 2 mới
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import joblib

# ================== CONFIGURATION ==================
# WiFi Communication
ESP32_CAM_IP = "192.168.4.1"
ESP32_BASE_URL = f"http://{ESP32_CAM_IP}"

# URLs for ESP32-CAM communication
STREAM_URL = f"{ESP32_BASE_URL}/stream"
SENSOR_DATA_URL = f"{ESP32_BASE_URL}/get_sensor"
COMMAND_URL = f"{ESP32_BASE_URL}/set_command"
STATUS_URL = f"{ESP32_BASE_URL}/status"

# Root folder config
ROOT_FOLDER = r'D:\Water Filter'

# Model paths
YOLO_MODEL_PATH = os.path.join(ROOT_FOLDER, 'my_model', 'my_model.pt')
MLP_MODEL_1_TFLITE_PATH = os.path.join(ROOT_FOLDER, 'water_project_ei', 'pretrained-model', 'model.tflite')
MODEL_1_PARAMS_PATH = os.path.join(ROOT_FOLDER, 'water_project_ei', 'model', 'parameters.json')

# Path cho Model 2: RandomForest
MODEL_2_RF_PATH = os.path.join(ROOT_FOLDER, 'filter_project_ei', 'model2_rf.joblib')
WATER_ENCODER_PATH = os.path.join(ROOT_FOLDER, 'filter_project_ei', 'water_encoder.joblib')

AUTO_RETRAIN_MODEL2 = True

# File paths
WATER_DATA_CSV = os.path.join(ROOT_FOLDER, 'water_data.csv')
SENSOR_DATA_CSV = os.path.join(ROOT_FOLDER, 'sensor_data.csv')
TRIAL_RESULTS_CSV = os.path.join(ROOT_FOLDER, 'trial_results.csv')
FILTER_TRAINING_CSV = os.path.join(ROOT_FOLDER, 'filter_training.csv')
WATER_SIGNATURES_JSON = os.path.join(ROOT_FOLDER, 'water_signatures.json')
DISTILLED_REP_CSV = os.path.join(ROOT_FOLDER, 'distilled_representation.csv')
DATA_TRAINING_READY_FLAG = os.path.join(ROOT_FOLDER, 'data_training_ready.txt')

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
        logging.FileHandler(os.path.join(ROOT_FOLDER, 'water_filter_system.log')),
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

# Model 2 variables
rf_model2 = None
water_encoder = None

class SmartTrialLearningSystem:
    def __init__(self):
        self.learning_data = []
        self.trial_history = []
        
        self.base_filters = [
            "activated_carbon",
            "coarse_filter",
            "fine_filter",
            "ro_filter",
            "ultrasonic_filter"
        ]
        
        self.filter_specialization = {
            "high_tds": ["ro_filter", "fine_filter"],
            "high_turbidity": ["coarse_filter", "fine_filter", "ultrasonic_filter"],
            "high_voc": ["activated_carbon", "ultrasonic_filter"],
            "low_ph": ["activated_carbon"],
            "high_ph": ["activated_carbon"]
        }
        
    def analyze_water_characteristics(self, sensor_data):
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
        if rf_model2 is not None:
            scores, stds, all_combos = self.get_model2_predictions(sensor_data, water_type)
            acquisition = np.array(scores) + 0.5 * np.array(stds)
            top_indices = np.argsort(acquisition)[-5:]
            suggested_combos = []
            for idx in top_indices[::-1]:
                combo_tuple = all_combos[idx]
                if all(v == 0 for v in combo_tuple):
                    suggested_combos.append(["OFF"])
                else:
                    filters = [self.base_filters[i] for i in range(len(self.base_filters)) if combo_tuple[i]]
                    suggested_combos.append(filters)
            logger.info(f"Model 2 suggested {len(suggested_combos)} combinations")
            return suggested_combos
        else:
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
    
    def get_model2_predictions(self, sensor_data, water_type):
        n_filters = len(self.base_filters)
        all_combos = list(itertools.product([0, 1], repeat=n_filters))
        scores = []
        stds = []
        sensor_vec = [sensor_data.get('ph', 0), sensor_data.get('TDS', 0), sensor_data.get('turbidity', 0), sensor_data.get('VOC', 0)]
        onehot = water_encoder.transform([[water_type]]).toarray().flatten().tolist()
        for combo_tuple in all_combos:
            indicators = list(combo_tuple)
            features = sensor_vec + onehot + indicators
            pred = rf_model2.predict([features])[0]
            tree_preds = [tree.predict([features])[0] for tree in rf_model2.estimators_]
            std = np.std(tree_preds)
            scores.append(pred)
            stds.append(std)
        return scores, stds, all_combos
    
    def evaluate_filter_performance(self, before_data, after_data):
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
    
    def record_trial_result(self, sensor_before, sensor_after, filter_combo, water_type, performance_score):
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
        try:
            sensor_before = trial_result['sensor_before']
            filter_combo = trial_result['filter_combo']
            performance = trial_result['performance_score']
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
                    performance
                ])
            logger.info(f"Added to training data: {filter_combo} (score: {performance:.3f})")
            return True
        except Exception as e:
            logger.error(f"Error adding to training data: {e}")
            return False
    
    def get_best_method_for_water(self, sensor_data, water_type):
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

class Model2BasedOODDetector:
    def __init__(self, confidence_threshold=0.6, entropy_threshold=1.5, history_size=100):
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        self.history_size = history_size
        self.prediction_history = deque(maxlen=history_size)
        self.confidence_history = deque(maxlen=history_size)
        self.stats = {
            'total_predictions': 0,
            'ood_count': 0,
            'avg_confidence': 0.0,
            'last_update': None
        }
        self.stats_file = os.path.join(ROOT_FOLDER, 'model2_ood_stats.json')
        self.load_stats()
    
    def calculate_entropy(self, probabilities):
        probs = np.clip(probabilities, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log(probs))
        return float(entropy)
    
    def detect_ood(self, model_output, sensor_data, yolo_count=0):
        try:
            max_confidence = float(np.max(model_output))
            predicted_class = int(np.argmax(model_output))
            entropy = self.calculate_entropy(model_output[0])
            top3_probs = np.sort(model_output[0])[-3:][::-1]
            prob_gap = float(top3_probs[0] - top3_probs[1])
            historical_confidence = self._get_historical_confidence()
            confidence_deviation = abs(max_confidence - historical_confidence) if historical_confidence else 0
            ood_reasons = []
            is_ood = False
            if max_confidence < self.confidence_threshold:
                ood_reasons.append(f"low_confidence({max_confidence:.3f}<{self.confidence_threshold})")
                is_ood = True
            if entropy > self.entropy_threshold:
                ood_reasons.append(f"high_entropy({entropy:.3f}>{self.entropy_threshold})")
                is_ood = True
            if prob_gap < 0.1:
                ood_reasons.append(f"ambiguous_prediction(gap={prob_gap:.3f})")
                is_ood = True
            if historical_confidence and confidence_deviation > 0.3:
                ood_reasons.append(f"unusual_confidence(deviation={confidence_deviation:.3f})")
                is_ood = True
            sensor_z_scores = self._calculate_sensor_z_scores(sensor_data)
            extreme_sensors = [f"{k}(z={v:.1f})" for k, v in sensor_z_scores.items() if abs(v) > 3]
            if extreme_sensors:
                ood_reasons.append(f"extreme_sensors:{','.join(extreme_sensors)}")
                is_ood = True
            explanation = {
                'is_ood': is_ood,
                'ood_score': float(1.0 - max_confidence),
                'reasons': ood_reasons,
                'confidence': max_confidence,
                'predicted_class': predicted_class,
                'entropy': entropy,
                'top3_probs': top3_probs.tolist(),
                'prob_gap': prob_gap,
                'historical_confidence': historical_confidence,
                'confidence_deviation': confidence_deviation,
                'sensor_z_scores': sensor_z_scores,
                'extreme_sensors': extreme_sensors,
                'yolo_count': yolo_count,
                'timestamp': datetime.now().isoformat()
            }
            self.prediction_history.append(predicted_class)
            self.confidence_history.append(max_confidence)
            self.stats['total_predictions'] += 1
            if is_ood:
                self.stats['ood_count'] += 1
            self.stats['avg_confidence'] = float(np.mean(self.confidence_history)) if self.confidence_history else 0.0
            self.stats['last_update'] = datetime.now().isoformat()
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
        if len(self.confidence_history) < 10:
            return None
        return float(np.mean(self.confidence_history))
    
    def _calculate_sensor_z_scores(self, sensor_data):
        z_scores = {}
        if len(self.prediction_history) < 20:
            return z_scores
        typical_ranges = {
            'ph': (6.0, 8.0, 7.0, 1.0),
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
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving Model2-OOD stats: {e}")
    
    def load_stats(self):
        try:
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r') as f:
                    self.stats = json.load(f)
                logger.info(f"Loaded Model2-OOD stats: {self.stats['total_predictions']} predictions")
        except Exception as e:
            logger.warning(f"Could not load Model2-OOD stats: {e}")
    
    def get_ood_rate(self):
        if self.stats['total_predictions'] == 0:
            return 0.0
        return self.stats['ood_count'] / self.stats['total_predictions']

class SensorCalibrationSystem:
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
        else:
            if not self.is_warmed_up:
                self.is_warmed_up = True
                logger.info("Sensor warm-up completed!")
        outliers = []
        for key, value in sensor_data.items():
            if key in self.sensor_ranges:
                min_val, max_val = self.sensor_ranges[key]
                if not (min_val <= value <= max_val):
                    outliers.append(key)
        if outliers:
            self.stats['rejected_outlier'] += 1
            reason = f"outliers: {','.join(outliers)}"
            logger.debug(f"Rejected: {reason}")
            return False, reason, None
        corrected_data = {}
        for key in ['ph', 'TDS', 'turbidity', 'VOC']:
            value = sensor_data.get(key, 0)
            self.buffers[key].append(value)
            corrected_data[key] = float(np.mean(self.buffers[key])) if len(self.buffers[key]) > 1 else value
        unstable = []
        for key in corrected_data:
            if len(self.buffers[key]) == self.window_size:
                cv = np.std(self.buffers[key]) / (np.mean(self.buffers[key]) + 1e-8)
                if cv > self.stability_threshold:
                    unstable.append(key)
        if unstable:
            self.stats['rejected_unstable'] += 1
            reason = f"unstable: {','.join(unstable)}"
            logger.debug(f"Rejected: {reason}")
            return False, reason, None
        self.stats['accepted_readings'] += 1
        return True, "valid", corrected_data

sensor_calibrator = SensorCalibrationSystem()

def get_sensor_data_from_arduino():
    try:
        response = requests.get(SENSOR_DATA_URL, timeout=5)
        if response.status_code == 200:
            raw_data = response.json()
            is_valid, reason, validated_data = sensor_calibrator.validate_reading(raw_data)
            if is_valid:
                logger.debug("Valid sensor data received")
                return validated_data
            else:
                logger.warning(f"Invalid sensor data: {reason}")
                return None
        else:
            logger.warning(f"Failed to get sensor data: HTTP {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error getting sensor data: {e}")
        return None

def send_command_to_arduino(command_chars):
    try:
        response = requests.get(f"{COMMAND_URL}?cmd={command_chars}", timeout=5)
        if response.status_code == 200:
            logger.info(f"Command sent: {command_chars}")
            with state_lock:
                global current_command_chars
                current_command_chars = command_chars
            return True
        else:
            logger.warning(f"Failed to send command: HTTP {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error sending command: {e}")
        return False

def apply_filter_combination(combo):
    if combo == ["OFF"]:
        send_command_to_arduino("abcdefg")
        return "OFF"
    else:
        relay_set = set()
        for method in combo:
            relay_set |= RELAY_COMMAND_MAP.get(method, set())
        command_chars = "".join([
            ch.upper() if r in relay_set else ch.lower()
            for r, ch in RELAY_ORDER
        ])
        send_command_to_arduino(f"CHARS:{command_chars}")
        return ",".join(combo)

def start_smart_trial_v2(initial_sensor, water_type, characteristics):
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

    thread = threading.Thread(target=trial_worker, daemon=True)
    thread.start()
    return True

def get_recommended_filter(sensor_data, water_type):
    if rf_model2 is None:
        return None
    scores, _, all_combos = learning_system.get_model2_predictions(sensor_data, water_type)
    best_idx = np.argmax(scores)
    combo_tuple = all_combos[best_idx]
    if all(v == 0 for v in combo_tuple):
        return "OFF"
    filters = [learning_system.base_filters[i] for i in range(len(learning_system.base_filters)) if combo_tuple[i]]
    return ",".join(filters)

def get_model2_output(sensor_data, water_type):
    if rf_model2 is None:
        return None
    scores, _, _ = learning_system.get_model2_predictions(sensor_data, water_type)
    probs = _softmax(np.array(scores))
    return np.array([probs])

def intelligent_control_loop():
    global last_sensor_request_time, current_sensor_data, stream_active
    global current_water_type, current_water_confidence, trial_in_progress
    
    sensor_fail_count = 0
    
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
                    
                    characteristics = learning_system.analyze_water_characteristics(raw_sensor_data)
                    
                    water_type, water_confidence, _ = classify_water_with_model1(raw_sensor_data)
                    
                    global current_water_type, current_water_confidence, current_water_characteristics
                    current_water_type = water_type
                    current_water_confidence = water_confidence
                    current_water_characteristics = characteristics
                    
                    logger.info(f"Detected: {water_type} (Confidence: {water_confidence:.2f})")
                    
                    if not trial_in_progress:
                        model_output = get_model2_output(raw_sensor_data, water_type)
                        if model_output is not None:
                            ood_detector = Model2BasedOODDetector()
                            is_ood, _ = ood_detector.detect_ood(model_output, raw_sensor_data)
                            global current_ood_status
                            current_ood_status = is_ood
                        else:
                            is_ood = True
                        
                        if not is_ood:
                            best_method = get_recommended_filter(raw_sensor_data, water_type)
                            if best_method:
                                logger.info(f"Applying recommended method: {best_method}")
                                apply_filter_combination(best_method.split(',') if best_method != "OFF" else ["OFF"])
                                current_recommended_method = best_method
                                current_method_source = "Model2 Recommendation"
                        else:
                            logger.info(f"OOD detected: {water_type} -> Starting smart trial")
                            start_smart_trial_v2(raw_sensor_data, water_type, characteristics)
                            
                    append_sensor_data(raw_sensor_data)
                    
                else:
                    sensor_fail_count += 1
                    if sensor_fail_count >= 3:
                        logger.warning("Sensor data unavailable")
                        sensor_fail_count = 0
                
                last_sensor_request_time = current_time
                
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

def train_model2():
    try:
        df = pd.read_csv(FILTER_TRAINING_CSV)
        if 'timestamp' not in df.columns:
            df.columns = ['timestamp', 'ph', 'TDS', 'turbidity', 'VOC', 'water_type', 'filter_combo', 'performance']
        
        df = df.dropna()
        df[['ph', 'TDS', 'turbidity', 'VOC', 'performance']] = df[['ph', 'TDS', 'turbidity', 'VOC', 'performance']].astype(float)
        
        global water_encoder
        water_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        water_encoder.fit(df[['water_type']])
        onehot_water = water_encoder.transform(df[['water_type']])
        onehot_df = pd.DataFrame(onehot_water, columns=water_encoder.get_feature_names_out(['water_type']))
        
        base_filters = learning_system.base_filters
        for f in base_filters:
            df[f] = df['filter_combo'].apply(lambda x: 1 if f in str(x).split(',') else 0)
        
        X = pd.concat([df[['ph', 'TDS', 'turbidity', 'VOC']], onehot_df, df[base_filters]], axis=1)
        y = df['performance']
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)
        rf.fit(X, y)
        
        joblib.dump(rf, MODEL_2_RF_PATH)
        joblib.dump(water_encoder, WATER_ENCODER_PATH)
        
        logger.info("Model 2 (RandomForest) retrained successfully!")
        return rf
    except Exception as e:
        logger.error(f"Error training Model 2: {e}")
        return None

def validate_and_clean_training_data():
    logger.info("Starting training data validation...")
    
    try:
        df = pd.read_csv(FILTER_TRAINING_CSV)
        df = df.dropna()
        df.to_csv(FILTER_TRAINING_CSV + ".cleaned", index=False)
        cleaned_df = df
    except Exception as e:
        logger.error(f"Error reading/cleaning CSV: {e}")
        return False
    
    if cleaned_df is not None and not cleaned_df.empty:
        n_classes = len(cleaned_df['water_type'].unique()) if 'water_type' in cleaned_df.columns else 0
        n_methods = len(cleaned_df['filter_combo'].unique()) if 'filter_combo' in cleaned_df.columns else 0
        
        stats = {
            'total_samples': len(cleaned_df),
            'n_classes': n_classes,
            'n_methods': n_methods
        }
        is_ready = stats['total_samples'] >= 100 and stats['n_classes'] >= 3
        
        if is_ready:
            logger.info("Data is READY for Model 2 training!")
            logger.info(f"{stats['total_samples']} samples, {stats['n_classes']} classes, {stats['n_methods']} methods")
            
            import shutil
            shutil.copy(FILTER_TRAINING_CSV + ".cleaned", FILTER_TRAINING_CSV)
            logger.info(f"Cleaned data applied to {FILTER_TRAINING_CSV}")
            
            return True
        else:
            logger.warning(f"Data NOT ready: Not enough samples or classes")
            logger.info(f"Current stats: {stats}")
            return False
    else:
        logger.warning("No data or empty DataFrame after cleaning")
        return False

def create_display_frame():
    display_frame = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
    
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
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
                (0, 255, 0) if current_recommended_method != "OFF" else (255, 255, 0), 2)
    
    cv2.putText(display_frame, f"Source: {current_method_source}", 
                (right_x, method_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
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
    
    info_x = 20
    info_y = 550
    
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
    
    status_bar_y = 780
    status_color = (0, 255, 0) if system_running and test_wifi_connection() else (0, 0, 255)
    status_text = "SYSTEM READY" if system_running and test_wifi_connection() else "SYSTEM ERROR"
    
    cv2.rectangle(display_frame, (0, status_bar_y), (DISPLAY_WIDTH, DISPLAY_HEIGHT), (40, 40, 40), -1)
    
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

def initialize_models():
    global yolo_model, interp1, input1_details, output1_details, mean1, scale1, names1
    global rf_model2, water_encoder
    
    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    if TF_AVAILABLE:
        interp1 = tf.lite.Interpreter(model_path=MLP_MODEL_1_TFLITE_PATH)
        interp1.allocate_tensors()
        input1_details = interp1.get_input_details()[0]
        output1_details = interp1.get_output_details()[0]
        
        try:
            with open(MODEL_1_PARAMS_PATH, 'r') as f:
                params1 = json.load(f)
            
            scaler_model = params1.get('scaler_parameters', {}).get('model', {})
            mean1 = np.array(scaler_model.get('mean', [0.0] * 4), dtype=np.float32)
            scale1 = np.array(scaler_model.get('scale', [1.0] * 4), dtype=np.float32)
            
            class_names_dict = params1.get('class_names', {})
            names1 = class_names_dict.get('classes', ['Unknown'] * 4) 
            
            logger.info(f"Model 1 loaded successfully: mean={mean1}, scale={scale1}, classes={names1}")
        except Exception as e:
            logger.warning(f"Failed to load Model 1 params: {e}. Using default values.")
            mean1 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            scale1 = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
            names1 = ['bestwater', 'hardwater', 'nothing', 'rainwater'] 
    
    if AUTO_RETRAIN_MODEL2 and validate_and_clean_training_data():
        rf_model2 = train_model2()
    elif os.path.exists(MODEL_2_RF_PATH) and os.path.exists(WATER_ENCODER_PATH):
        rf_model2 = joblib.load(MODEL_2_RF_PATH)
        water_encoder = joblib.load(WATER_ENCODER_PATH)
        logger.info("Loaded Model 2 (RandomForest)")
    else:
        logger.warning("Model 2 not loaded - will use fallback rules for suggestions")

def classify_water_with_model1(sensor_data):
    try:
        features = np.array([sensor_data.get('ph', 0), sensor_data.get('TDS', 0), 
                             sensor_data.get('turbidity', 0), sensor_data.get('VOC', 0)], dtype=np.float32)
        normalized = (features - mean1) / scale1
        
        with _interpreter_lock1:
            interp1.set_tensor(input1_details['index'], normalized.reshape(1, -1))
            interp1.invoke()
            output = interp1.get_tensor(output1_details['index'])[0]
        
        probs = _softmax(output)
        label_idx = np.argmax(probs)
        confidence = probs[label_idx]
        label = names1[label_idx]
        
        return label, float(confidence), probs.tolist()
    except Exception as e:
        logger.error(f"Error classifying water: {e}")
        return "Unknown", 0.0, []

def initialize_system():
    logger.info("Initializing WIFI Water Filter System...")
    
    ensure_data_files()
    
    if os.path.exists(FILTER_TRAINING_CSV):
        logger.info("Checking training data quality...")
        validate_and_clean_training_data()
    
    global water_signatures_cache, distilled_representation
    water_signatures_cache = load_water_signatures()
    distilled_representation = load_distilled_representation()
    
    initialize_models()
    
    model2_ood_detector = Model2BasedOODDetector()
    model2_ood_detector.load_stats()
    logger.info(f"Model2-OOD: {model2_ood_detector.stats['total_predictions']} predictions, "
                f"{model2_ood_detector.get_ood_rate()*100:.1f}% OOD rate")
    
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
    global system_running
    system_running = False
    
    logger.info("Turning off all relays...")
    send_command_to_arduino("abcdefg")
    
    model2_ood_detector = Model2BasedOODDetector()
    model2_ood_detector.save_stats()
    
    logger.info("System cleanup completed")

def load_water_signatures():
    try:
        with open(WATER_SIGNATURES_JSON, 'r') as f:
            return json.load(f)
    except:
        return {}

def load_distilled_representation():
    try:
        return pd.read_csv(DISTILLED_REP_CSV)
    except:
        return None

def append_sensor_data(sensor_data):
    with csv_lock, open(SENSOR_DATA_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat()] + list(sensor_data.values()))

def optimized_yolo_processing():
    while system_running:
        with frame_lock:
            if latest_frame is not None:
                results = yolo_model(latest_frame)
                annotated_frame = results[0].plot()
                global latest_frame_with_boxes, yolo_detections
                latest_frame_with_boxes = annotated_frame
                yolo_detections = [result.names[int(cls)] for result in results for cls in result.boxes.cls]
        time.sleep(1)  # Process every second

def optimized_video_stream():
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    while system_running:
        if stream_active:
            try:
                response = session.get(STREAM_URL, stream=True)
                if response.status_code == 200:
                    bytes_data = bytes()
                    for chunk in response.iter_content(chunk_size=1024):
                        bytes_data += chunk
                        a = bytes_data.find(b'\xff\xd8')
                        b = bytes_data.find(b'\xff\xd9')
                        if a != -1 and b != -1:
                            jpg = bytes_data[a:b+2]
                            bytes_data = bytes_data[b+2:]
                            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            with frame_lock:
                                global latest_frame
                                latest_frame = frame
            except Exception as e:
                logger.error(f"Video stream error: {e}")
                time.sleep(1)

def test_wifi_connection():
    try:
        requests.get(STATUS_URL, timeout=2)
        return True
    except:
        return False

def ensure_data_files():
    for file in [WATER_DATA_CSV, SENSOR_DATA_CSV, TRIAL_RESULTS_CSV, FILTER_TRAINING_CSV, WATER_SIGNATURES_JSON, DISTILLED_REP_CSV]:
        if not os.path.exists(file):
            open(file, 'w').close()

def update_water_signature(sensor_data, best_method, score, achieved_threshold):
    try:
        signatures = load_water_signatures()
        key = tuple(sensor_data.values())
        signatures[str(key)] = {
            'best_method': best_method,
            'score': score,
            'achieved_threshold': achieved_threshold
        }
        with open(WATER_SIGNATURES_JSON, 'w') as f:
            json.dump(signatures, f, indent=4)
    except Exception as e:
        logger.error(f"Error updating water signature: {e}")

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