# collect_water_data.py
import requests
import csv
import time
from datetime import datetime
import os
import sys

# ================== CONFIGURATION ==================
ESP32_CAM_IP = "192.168.4.1"  # Địa chỉ IP của ESP32-CAM
SENSOR_DATA_URL = f"http://{ESP32_CAM_IP}/get_sensor"
WATER_DATA_CSV = "water_data.csv"
LABEL = "nothing"
COLLECTION_INTERVAL = 3  # Thu thập dữ liệu mỗi 5 giây
def get_sensor_data():
    """Lấy dữ liệu cảm biến từ ESP32-CAM"""
    try:
        response = requests.get(SENSOR_DATA_URL, timeout=5)
        
        if response.status_code == 200:
            sensor_data = response.json()
            
            # Xử lý dữ liệu
            processed_data = {}
            
            # Map dữ liệu từ ESP32 sang định dạng CSV
            if 'ph' in sensor_data:
                processed_data['pH'] = float(sensor_data['ph'])
            elif 'pH' in sensor_data:
                processed_data['pH'] = float(sensor_data['pH'])
            else:
                processed_data['pH'] = 0.0
                
            if 'tds' in sensor_data:
                processed_data['TDS_ppm'] = float(sensor_data['tds'])
            elif 'TDS' in sensor_data:
                processed_data['TDS_ppm'] = float(sensor_data['TDS'])
            else:
                processed_data['TDS_ppm'] = 0.0
                
            if 'turbidity' in sensor_data:
                processed_data['turbidity_NTU'] = float(sensor_data['turbidity'])
            elif 'turbidity_NTU' in sensor_data:
                processed_data['turbidity_NTU'] = float(sensor_data['turbidity_NTU'])
            else:
                processed_data['turbidity_NTU'] = 0.0
                
            if 'voc' in sensor_data:
                processed_data['VOC_mg_L'] = float(sensor_data['voc'])
            elif 'VOC' in sensor_data:
                processed_data['VOC_mg_L'] = float(sensor_data['VOC'])
            elif 'VOC_mg_L' in sensor_data:
                processed_data['VOC_mg_L'] = float(sensor_data['VOC_mg_L'])
            else:
                processed_data['VOC_mg_L'] = 0.0
            
            return processed_data
        else:
            print(f"Lỗi HTTP: {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:

        return None
    except Exception as e:

        return None

def init_csv_file():
    """Khởi tạo file CSV nếu chưa tồn tại"""
    if not os.path.exists(WATER_DATA_CSV):
        with open(WATER_DATA_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'pH', 'TDS_ppm', 'turbidity_NTU', 'VOC_mg_L', 'label'])
        print(f"{WATER_DATA_CSV}")
    else:
        print(f"{WATER_DATA_CSV}")

def save_sensor_data(sensor_data, label):
    """Lưu dữ liệu cảm biến vào CSV"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(WATER_DATA_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                sensor_data.get('pH', 0.0),
                sensor_data.get('TDS_ppm', 0.0),
                sensor_data.get('turbidity_NTU', 0.0),
                sensor_data.get('VOC_mg_L', 0.0),
                label
            ])
        
        print(f"{timestamp} - pH: {sensor_data.get('pH', 0.0):.2f}, "
              f"TDS: {sensor_data.get('TDS_ppm', 0.0):.1f} ppm, "
              f"Turbidity: {sensor_data.get('turbidity_NTU', 0.0):.2f} NTU, "
              f"VOC: {sensor_data.get('VOC_mg_L', 0.0):.3f} mg/L, "
              f"Label: {label}")
        
        return True
        
    except Exception as e:
        return False

def check_label():
    """Kiểm tra label đã được cấu hình chưa"""
    if LABEL == "your_label_here":
        return False
    return True

def main():
    """Hàm chính thu thập dữ liệu"""
    
    # Kiểm tra label
    if not check_label():
        sys.exit(1)
    
    # Khởi tạo file CSV
    init_csv_file()
    
    # Đếm số lượng dữ liệu đã thu thập
    count = 0
    
    try:
        while True:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ")
            
            # Lấy dữ liệu từ ESP32
            sensor_data = get_sensor_data()
            
            if sensor_data:
                # Lưu dữ liệu với label đã cấu hình
                if save_sensor_data(sensor_data, LABEL):
                    count += 1
                    print(f"{count}")
            
            # Chờ trước khi thu thập tiếp
            print(f"{COLLECTION_INTERVAL}")
            time.sleep(COLLECTION_INTERVAL)
            
    except Exception as e:
        print(f"{e}")

if __name__ == "__main__":
    main()