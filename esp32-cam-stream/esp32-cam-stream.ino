#include "esp_camera.h"
#include "WiFi.h"
#include "ESPAsyncWebServer.h"
#include "AsyncTCP.h"
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

AsyncWebServer server(80);
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
void handleRoot(AsyncWebServerRequest *request);
void handleStream(AsyncWebServerRequest *request);
void handleStatus(AsyncWebServerRequest *request);
void handleSensorData(AsyncWebServerRequest *request);
void handleGetSensor(AsyncWebServerRequest *request);
void handleSetCommand(AsyncWebServerRequest *request);
void handleGetCommand(AsyncWebServerRequest *request);

void streamTask(void *param){
  AsyncWebServerRequest *request = (AsyncWebServerRequest*)param;
  AsyncResponseStream *response = request->beginResponseStream("multipart/x-mixed-replace; boundary=frame");
  response->addHeader("Access-Control-Allow-Origin", "*");
  response->addHeader("Cache-Control", "no-cache");
  request->send(response);
  
  unsigned long lastFrameTime = 0;
  while(request->client()->connected()){
    unsigned long currentTime = millis();
    if (currentTime - lastFrameTime >= STREAM_INTERVAL_MS) {
      fb = esp_camera_fb_get();
      if (fb){
        response->print("--frame\r\n");
        response->print("Content-Type: image/jpeg\r\n");
        response->print("Content-Length: " + String(fb->len) + "\r\n\r\n");
        response->write(fb->buf, fb->len);
        response->print("\r\n");
        esp_camera_fb_return(fb);
        fb = NULL;
      }
      lastFrameTime = currentTime;
    }
    vTaskDelay(10 / portTICK_PERIOD_MS);
  }
  vTaskDelete(NULL);
}

void setup() {
  Serial.begin(SERIAL_BAUDRATE);
  Serial.setTimeout(SERIAL_TIMEOUT);
  delay(1000);
  
  Serial.println("=== ESP32-CAM Water Filter System ===");
  Serial.println("Initializing...");

  setupCamera();
  startAccessPoint();
  
  if (apStarted) {
    server.on("/", HTTP_GET, handleRoot);
    server.on("/stream", HTTP_GET, handleStream);
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
  
  delay(10);
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
  config.xclk_freq_hz = 20000000;  // Up to 20MHz for better FPS
  config.pixel_format = PIXFORMAT_JPEG;
  
  config.frame_size = FRAME_SIZE;
  config.jpeg_quality = JPEG_QUALITY;
  config.fb_count = 2;  // Double buffer for smoother stream

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

void handleRoot(AsyncWebServerRequest *request){
  const char* html = "<html><head><title>ESP32-CAM Water Filter</title>"
                     "<meta http-equiv='refresh' content='5'>"
                     "</head><body>"
                     "<h1>ESP32-CAM Water Filter System</h1>"
                     "<img src='/stream' style='width: 640px;'/>"
                     "<p><a href='/status'>System Status</a></p>"
                     "<p>Auto-refresh every 5 seconds</p>"
                     "</body></html>";
  request->send(200, "text/html", html);
}

void handleStream(AsyncWebServerRequest *request){
  if (WiFi.softAPgetStationNum() > 2) {
    request->send(503, "text/plain", "Too many clients connected");
    return;
  }
  xTaskCreatePinnedToCore(streamTask, "stream", 8192, request, 1, NULL, 1);
}

void handleStatus(AsyncWebServerRequest *request){
  JSONVar status;
  status["status"] = "online";
  status["mode"] = "access_point";
  status["ap_ssid"] = ap_ssid;
  status["ap_ip"] = WiFi.softAPIP().toString();
  status["clients_connected"] = WiFi.softAPgetStationNum();
  status["free_heap"] = esp_get_free_heap_size();
  status["sensor_data_available"] = sensorDataAvailable ? true : false;
  status["last_sensor_update"] = lastSensorUpdate;
  
  request->send(200, "application/json", JSON.stringify(status));
}

void handleSensorData(AsyncWebServerRequest *request){
  if (request->hasArg("plain")) {
    currentSensorData = request->arg("plain");
    lastSensorUpdate = millis();
    sensorDataAvailable = true;
    Serial.print("Received sensor data: ");
    Serial.println(currentSensorData);
    request->send(200, "application/json", "{\"status\":\"received\"}");
  } else {
    request->send(400, "text/plain", "No data");
  }
}

void handleGetSensor(AsyncWebServerRequest *request){
  if (sensorDataAvailable && currentSensorData.length() > 0 && currentSensorData != "{}") {
    request->send(200, "application/json", currentSensorData);
  } else {
    request->send(200, "application/json", "{\"error\":\"No sensor data available\"}");
  }
}

void handleSetCommand(AsyncWebServerRequest *request){
  if (request->hasArg("plain")) {
    currentCommand = request->arg("plain");
    Serial.print("Received command: ");
    Serial.println(currentCommand);
    request->send(200, "application/json", "{\"status\":\"command_received\"}");
  } else {
    request->send(400, "text/plain", "No command");
  }
}

void handleGetCommand(AsyncWebServerRequest *request){
  if (currentCommand.length() > 0) {
    request->send(200, "text/plain", currentCommand);
    currentCommand = "";  // Clear after sending
  } else {
    request->send(200, "text/plain", "No command");
  }
}
