#include "esp_camera.h"
#include "WiFi.h"
#include "WiFiClient.h"
#include "WebServer.h"
#include "ESPmDNS.h"
#include "Arduino_JSON.h"
#include <Arduino.h>

#define CAMERA_MODEL_AI_THINKER

#if defined(CAMERA_MODEL_AI_THINKER)
  #define PWDN_GPIO_NUM     32
  #define RESET_GPIO_NUM    -1
  #define XCLK_GPIO_NUM      0
  #define SIOD_GPIO_NUM     26
  #define SIOC_GPIO_NUM     27
  #define Y9_GPIO_NUM       35
  #define Y8_GPIO_NUM       34
  #define Y7_GPIO_NUM       39
  #define Y6_GPIO_NUM       36
  #define Y5_GPIO_NUM       21
  #define Y4_GPIO_NUM       19
  #define Y3_GPIO_NUM       18
  #define Y2_GPIO_NUM        5
  #define VSYNC_GPIO_NUM    25
  #define HREF_GPIO_NUM     23
  #define PCLK_GPIO_NUM     22
#else
  #error "Camera model not selected"
#endif

const char* ap_ssid = "ESP32-CAM-Water-Filter";
const char* ap_password = "12345678";

#define SERIAL_BAUDRATE 115200  
#define SERIAL_TIMEOUT 100
#define STREAM_INTERVAL_MS 200  // Giảm FPS xuống 5 FPS để giảm tải

WebServer server(80);
bool apStarted = false;

// Biến toàn cục
String currentSensorData = "{}";
String currentCommand = "";
unsigned long lastSensorUpdate = 0;
bool sensorDataAvailable = false;

camera_fb_t *fb = NULL;

#define JPEG_QUALITY 10        // Chất lượng vừa phải
#define FRAME_SIZE FRAMESIZE_QVGA  // Giữ QVGA, không quá nhỏ

void setupCamera();
void startAccessPoint();
void handleRoot();
void handleStream();
void handleStatus();
void handleSensorData();
void handleGetSensor();
void handleSetCommand();
void handleGetCommand();

void setup() {
  Serial.begin(SERIAL_BAUDRATE);
  Serial.setTimeout(SERIAL_TIMEOUT);
  delay(1000);
  
  Serial.println("=== ESP32-CAM Water Filter System ===");
  Serial.println("Initializing...");

  setupCamera();
  startAccessPoint();
  
  if (apStarted) {
    server.on("/", handleRoot);
    server.on("/stream", handleStream);
    server.on("/status", HTTP_GET, handleStatus);
    server.on("/sensor_data", HTTP_POST, handleSensorData);
    server.on("/get_sensor", HTTP_GET, handleGetSensor);
    server.on("/set_command", HTTP_POST, handleSetCommand);
    server.on("/get_command", HTTP_GET, handleGetCommand);
    
    server.begin();
    Serial.println("HTTP server started");
    Serial.println("MODE: AP_WIFI_COMMUNICATION");
    
    Serial.print("Connect to WiFi: ");
    Serial.println(ap_ssid);
    Serial.print("Then visit: http://");
    Serial.println(WiFi.softAPIP());
  }
  
  Serial.println("Ready for WiFi communication");
}

void loop() {
  if (apStarted) {
    server.handleClient();
  }
  
  // Xử lý command từ serial nếu có
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command == "HEALTH_CHECK") {
      Serial.println("HEALTH:OK|HEAP:" + String(esp_get_free_heap_size()) + 
                    "|CLIENTS:" + String(WiFi.softAPgetStationNum()));
    }
    else if (command == "STATUS") {
      Serial.println("STATUS:AP_MODE|SSID:" + String(ap_ssid) + 
                    "|IP:" + WiFi.softAPIP().toString() +
                    "|CLIENTS:" + String(WiFi.softAPgetStationNum()));
    }
  }
  
  delay(10);  // Giảm delay để xử lý request nhanh hơn
}

void setupCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 10000000;
  config.pixel_format = PIXFORMAT_JPEG;
  
  config.frame_size = FRAME_SIZE;
  config.jpeg_quality = JPEG_QUALITY;
  config.fb_count = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }
  
  sensor_t *s = esp_camera_sensor_get();
  if (s != NULL) {
    s->set_framesize(s, FRAME_SIZE);
    s->set_quality(s, JPEG_QUALITY);
  }
  
  Serial.println("Camera initialized successfully");
}

void startAccessPoint() {
  Serial.printf("Starting Access Point: %s\n", ap_ssid);
  
  WiFi.mode(WIFI_AP);
  WiFi.setSleep(false);
  WiFi.setTxPower(WIFI_POWER_19_5dBm);
  
  IPAddress local_ip(192, 168, 4, 1);
  IPAddress gateway(192, 168, 4, 1);
  IPAddress subnet(255, 255, 255, 0);
  
  bool result = WiFi.softAPConfig(local_ip, gateway, subnet);
  if (!result) {
    Serial.println("AP Config failed, but continuing...");
  }
  
  result = WiFi.softAP(ap_ssid, ap_password, 1, 0, 8);
  
  if (!result) {
    Serial.println("AP Setup failed.");
    apStarted = false;
    return;
  }
  
  apStarted = true;
  delay(500);
  
  Serial.print("AP IP address: ");
  Serial.println(WiFi.softAPIP());
  Serial.println("Access Point started successfully!");
}

void handleSensorData() {
  if (server.method() == HTTP_POST) {
    String newData = server.arg("plain");
    
    Serial.print("Received sensor data: ");
    Serial.println(newData);
    
    currentSensorData = newData;
    lastSensorUpdate = millis();
    sensorDataAvailable = true;
    
    server.send(200, "application/json", "{\"status\":\"received\"}");
  } else {
    server.send(405, "text/plain", "Method Not Allowed");
  }
}

void handleGetSensor() {
  if (sensorDataAvailable && currentSensorData.length() > 0 && currentSensorData != "{}") {
    server.send(200, "application/json", currentSensorData);
  } else {
    server.send(200, "application/json", "{\"error\":\"No sensor data available\"}");
  }
}

void handleSetCommand() {
  if (server.method() == HTTP_POST) {
    String newCommand = server.arg("plain");
    
    Serial.print("Received command: ");
    Serial.println(newCommand);
    
    currentCommand = newCommand;
    
    server.send(200, "application/json", "{\"status\":\"command_received\"}");
  } else {
    server.send(405, "text/plain", "Method Not Allowed");
  }
}

void handleGetCommand() {
  if (currentCommand.length() > 0) {
    server.send(200, "text/plain", currentCommand);
    currentCommand = "";  // Clear command after sending
  } else {
    server.send(200, "text/plain", "No command");
  }
}

void handleRoot() {
  String html = "<html><head><title>ESP32-CAM Water Filter</title>";
  html += "<meta http-equiv='refresh' content='5'>";
  html += "</head><body>";
  html += "<h1>ESP32-CAM Water Filter System</h1>";
  html += "<img src='/stream' style='width: 640px;'/>";
  html += "<p><a href='/status'>System Status</a></p>";
  html += "<p>Auto-refresh every 5 seconds</p>";
  html += "</body></html>";
  
  server.send(200, "text/html", html);
}

void handleStream() {
  WiFiClient client = server.client();
  
  if (!client.connected()) {
    return;
  }
  
  // Kiểm tra số lượng client - giới hạn để tránh quá tải
  int clientCount = WiFi.softAPgetStationNum();
  if (clientCount > 2) {
    Serial.println("Too many clients, rejecting stream");
    client.println("HTTP/1.1 503 Service Unavailable");
    client.println("Content-Type: text/plain");
    client.println();
    client.println("Too many clients connected");
    client.stop();
    return;
  }
  
  // Gửi HTTP response header
  String response = "HTTP/1.1 200 OK\r\n";
  response += "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n";
  response += "Access-Control-Allow-Origin: *\r\n";
  response += "Cache-Control: no-cache\r\n";
  response += "Connection: keep-alive\r\n";
  response += "\r\n";
  client.print(response);
  
  unsigned long lastFrameTime = 0;
  unsigned long streamStartTime = millis();
  
  // Stream liên tục, không giới hạn thời gian
  while (client.connected()) {
    // Xử lý các request khác trong khi stream
    server.handleClient();
    
    unsigned long currentTime = millis();
    
    // Chỉ gửi frame mỗi STREAM_INTERVAL_MS
    if (currentTime - lastFrameTime >= STREAM_INTERVAL_MS) {
      fb = esp_camera_fb_get();
      if (!fb) {
        Serial.println("Camera capture failed");
        if (fb) {
          esp_camera_fb_return(fb);
        }
        delay(50);
        continue;
      }

      // Gửi frame boundary và header
      client.print("--frame\r\n");
      client.print("Content-Type: image/jpeg\r\n");
      client.print("Content-Length: " + String(fb->len) + "\r\n");
      client.print("\r\n");
      
      // Gửi frame data
      client.write(fb->buf, fb->len);
      client.print("\r\n");
      
      esp_camera_fb_return(fb);
      fb = NULL;
      
      lastFrameTime = currentTime;
    }
    
    delay(10);  // Nhường CPU cho các task khác
  }
  
  client.stop();
  Serial.println("Stream ended");
}

void handleStatus() {
  String json = "{";
  json += "\"status\":\"online\",";
  json += "\"mode\":\"access_point\",";
  json += "\"ap_ssid\":\"" + String(ap_ssid) + "\",";
  json += "\"ap_ip\":\"" + WiFi.softAPIP().toString() + "\",";
  json += "\"clients_connected\":" + String(WiFi.softAPgetStationNum()) + ",";
  json += "\"free_heap\":" + String(esp_get_free_heap_size()) + ",";
  json += "\"sensor_data_available\":" + String(sensorDataAvailable ? "true" : "false") + ",";
  json += "\"last_sensor_update\":" + String(lastSensorUpdate);
  json += "}";
  
  server.send(200, "application/json", json);
}