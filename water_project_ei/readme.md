
# Water Quality Classification Project

Project này chứa model Machine Learning để phân loại chất lượng nước.

## Cấu trúc thư mục
- **model/**: Chứa các file cấu hình, tên lớp và tham số chuẩn hóa (Scaling).
  - `parameters.json`: Chứa Mean và Std Dev dùng để chuẩn hóa dữ liệu Raw từ cảm biến. **Bắt buộc dùng file này trên thiết bị nhúng**.
- **pretrained-model/**:
  - `model.tflite`: Model đã được tối ưu hóa để chạy trên thiết bị (Edge Impulse, ESP32, Arduino...).

## Hướng dẫn sử dụng trên thiết bị
1. Đọc giá trị raw từ cảm biến (pH, TDS, Turbidity, VOC).
2. **QUAN TRỌNG**: Sử dụng thông số trong `parameters.json` để chuẩn hóa:
   `input = (raw_value - mean) / scale`
3. Đưa giá trị input đã chuẩn hóa vào `model.tflite`.
4. Lấy index của output có xác suất cao nhất -> map với `class_names.json`.
