#include <WiFiS3.h>
#include <ArduinoHttpClient.h>
#include <Arduino_JSON.h>
#include <Wire.h>
#include "Adafruit_CCS811.h"

// Kết nối vào WiFi do ESP32-CAM host
char WIFI_SSID[] = "ESP32-CAM-Water-Filter";
char WIFI_PASS[] = "12345678";
const char* ESP32_SERVER = "192.168.4.1"; // IP của ESP32-CAM
const int ESP32_PORT = 80;

WiFiClient wifi;
Adafruit_CCS811 ccs;

// Sensor pins
#define PH_PIN A0
#define TDS_PIN A1
#define TURBIDITY_PIN A2
#define MAX_SEND_RETRIES 2
#define SEND_INTERVAL 3000     
#define COMMAND_INTERVAL 1000

// Relay mapping (a..g)
const int RELAY_PINS[] = {2, 3, 5, 6, 7, 8, 9};
const int RELAY_COUNT = sizeof(RELAY_PINS)/sizeof(RELAY_PINS[0]);

float phValue=0, tdsValue=0, turbidityValue=0, vocValue=0;
uint16_t eco2=0, tvoc=0;
bool manualMode=false;
unsigned long lastUpdateTime = 0;
const unsigned long UPDATE_INTERVAL = 3000;
unsigned long lastCommandCheck = 0;
const unsigned long COMMAND_CHECK_INTERVAL = 1000;

String fstr(float v, uint8_t prec=15){
    char buf[32];
    // width 0 để không ép rộng, prec = số chữ số thập phân
    dtostrf(v, 0, prec, buf);
    return String(buf);
  }

bool wifi_ok = false;
String sensorBuffer = "{}";
bool newDataAvailable = false;

// Helper functions
void setRelayByIndex(int idx, bool on) {
  if (idx<0 || idx>=RELAY_COUNT) return;
  digitalWrite(RELAY_PINS[idx], on?LOW:HIGH);
}

void turnOffAllRelays(){ 
  for (int i=0;i<RELAY_COUNT;i++) setRelayByIndex(i,false); 
}

void printRelayStatus(){
  for(int i=0;i<RELAY_COUNT;i++){
    Serial.print((i==0?"R1: ":(i==1?"R2: ":(i==2?"R4: ":(i==3?"R5: ":(i==4?"R6: ":(i==5?"R7: ":"R8: ")))))));
    Serial.print(digitalRead(RELAY_PINS[i])==LOW?"ON ":"OFF ");
  }
  Serial.println();
}

void connectWifi(){
  Serial.print("Connecting to ESP32-CAM WiFi...");
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  unsigned long start=millis();
  while (WiFi.status()!=WL_CONNECTED && millis()-start < 15000){
    delay(500);
    Serial.print(".");
  }
  if (WiFi.status()==WL_CONNECTED){
    Serial.println();
    Serial.print("WiFi Connected! IP=");
    Serial.println(WiFi.localIP());
    wifi_ok = true;
  }else{
    Serial.println();
    Serial.println("WiFi connection failed!");
    wifi_ok = false;
  }
}

void sendSensorDataToESP32(){
  if (!wifi_ok) {
    Serial.println("WiFi not connected, skipping sensor data send");
    return;
  }
  
  WiFiClient client;


  
  // Tạo HTTP POST request thủ công
  if (client.connect(ESP32_SERVER, ESP32_PORT)) {
    // Tạo JSON data
    String jsonData = "{";
    jsonData += "\"ph\":" + fstr(phValue, 15) + ",";
    jsonData += "\"tds\":" + fstr(tdsValue, 15) + ",";
    jsonData += "\"turbidity\":" + fstr(turbidityValue, 15) + ",";
    jsonData += "\"voc\":" + fstr(vocValue, 15) + ",";
    jsonData += "\"eco2\":" + String(eco2) + ",";
    jsonData += "\"tvoc\":" + String(tvoc);
    jsonData += "}";
    
    Serial.print("Sending sensor data: ");
    Serial.println(jsonData);
    
    // Gửi POST request
    client.println("POST /sensor_data HTTP/1.1");
    client.println("Host: " + String(ESP32_SERVER));
    client.println("Content-Type: application/json");
    client.println("Connection: close");
    client.print("Content-Length: ");
    client.println(jsonData.length());
    client.println();
    client.println(jsonData);
    
    // Đợi phản hồi
    unsigned long timeout = millis();
    while (client.connected() && millis() - timeout < 2000) {
      if (client.available()) {
        String line = client.readStringUntil('\n');
        if (line.startsWith("HTTP/1.1 200")) {
          Serial.println("Sensor data sent successfully");
        }
        break;
      }
    }
    
    client.stop();
  } else {
    Serial.println("Failed to connect to ESP32-CAM");
  }
}

void checkAndApplyCommand(){
  if (!wifi_ok) return;
  
  WiFiClient client;
  
  if (client.connect(ESP32_SERVER, ESP32_PORT)) {
    client.println("GET /get_command HTTP/1.1");
    client.println("Host: " + String(ESP32_SERVER));
    client.println("Connection: close");
    client.println();
    
    unsigned long timeout = millis();
    bool inBody = false;
    String command = "";
    
    while (client.connected() && millis() - timeout < 2000) {
      if (client.available()) {
        String line = client.readStringUntil('\n');
        line.trim();
        
        if (line.length() == 0) {
          inBody = true;
          continue;
        }
        
        if (inBody) {
          command = line;
          break;
        }
      }
    }
    
    client.stop();
    
    command.trim();
    if (command.length() > 0 && command != "No command") {
      Serial.print("Received command: ");
      Serial.println(command);
      
      if (command.startsWith("CHARS:")) {
        String chars = command.substring(6);
        applyCharCommand(chars);
      }
    }
  }
}

void applyCharCommand(String chars) {
  if (chars.length() >= RELAY_COUNT) {
    for (int i=0; i<RELAY_COUNT; i++) {
      char ch = chars.charAt(i);
      bool on = (ch>='A' && ch<='Z');
      setRelayByIndex(i, on);
    }
    Serial.print("Applied relay command: ");
    Serial.println(chars);
    printRelayStatus();
  }
}

void readSensors(){
  // Đọc CCS811
  if (ccs.available() && !ccs.readData()){
    eco2 = ccs.geteCO2(); 
    tvoc = ccs.getTVOC(); 
    vocValue = tvoc/1000.0;
  }
  
  // Đọc pH
  int phADC = analogRead(PH_PIN);
  phValue = (phADC * 5.0 / 1023.0) * 3.5;
  
  // Đọc TDS
  int tdsADC = analogRead(TDS_PIN);
  float voltage = tdsADC * (5.0 / 1023.0);
  tdsValue = (133.42 * voltage*voltage*voltage - 255.86*voltage*voltage + 857.39*voltage) * 0.5;
  
  // Đọc Turbidity
  int turbidityADC = analogRead(TURBIDITY_PIN);
  turbidityValue = turbidityADC * (5.0 / 1023.0);
}

void readAndBufferSensors() {
  readSensors(); 
  
  JSONVar data;
  data["ph"] = phValue;
  data["tds"] = tdsValue;
  data["turbidity"] = turbidityValue;
  data["voc"] = vocValue;
  data["eco2"] = eco2;
  data["tvoc"] = tvoc;
  data["timestamp"] = millis();
  
  sensorBuffer = JSON.stringify(data);
  newDataAvailable = true;
  
  // Gửi ngay nếu có WiFi
  if (wifi_ok && newDataAvailable) {
    sendSensorDataToESP32();
    newDataAvailable = false;
  }
}

void handleSerialCommands(){
  while (Serial.available()>0){
    String line = Serial.readStringUntil('\n');
    line.trim();
    
    if (line.length()==0) continue;
    
    if (line.startsWith("CHARS:")){
      String chars = line.substring(6);
      applyCharCommand(chars);
    }
    else if (line == "0"){ 
      manualMode=false; 
      Serial.println("AUTO mode"); 
    }
    else if (line == "1"){ 
      manualMode=true; 
      Serial.println("MANUAL mode"); 
    }
    else if (manualMode){
      char cmd = line.charAt(0);
      if (cmd=='9'){ 
        turnOffAllRelays(); 
        Serial.println("All relays OFF"); 
      }
      else if (cmd>='A' && cmd<='G'){ 
        int idx=cmd-'A'; 
        setRelayByIndex(idx,true); 
        Serial.print("Manual ON R"); 
        Serial.println(idx+1); 
      }
      else if (cmd>='a' && cmd<='g'){ 
        int idx=cmd-'a'; 
        setRelayByIndex(idx,false); 
        Serial.print("Manual OFF R"); 
        Serial.println(idx+1); 
      }
      printRelayStatus();
    } else {
      Serial.print("Auto mode - Ignored: ");
      Serial.println(line);
    }
  }
}

void sendSensorDataToESP32WithRetry() {
  if (!wifi_ok) return;
  
  for (int retry = 0; retry < MAX_SEND_RETRIES; retry++) {
    WiFiClient client;
    HttpClient http = HttpClient(client, ESP32_SERVER, ESP32_PORT);
    
    JSONVar data;
    data["ph"] = phValue;
    data["tds"] = tdsValue;
    data["turbidity"] = turbidityValue;
    data["voc"] = vocValue;
    data["eco2"] = eco2;
    data["tvoc"] = tvoc;
    data["timestamp"] = millis();
    
    String jsonString = JSON.stringify(data);
    
    http.post("/sensor_data", "application/json", jsonString);
    
    int statusCode = http.responseStatusCode();
    
    if (statusCode == 200) {
      Serial.println("Data sent OK");
      http.stop();
      return;
    }
    
    Serial.print("Send failed (retry ");
    Serial.print(retry);
    Serial.print("): ");
    Serial.println(statusCode);
    
    http.stop();
    delay(500); // Đợi trước khi retry
  }
  
  Serial.println("Failed to send data after retries");
}

void setup(){
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("=== Arduino Uno R4 WiFi - Water Filter ===");
  
  // Khởi tạo relay
  for (int i=0;i<RELAY_COUNT;i++){ 
    pinMode(RELAY_PINS[i], OUTPUT); 
    digitalWrite(RELAY_PINS[i], HIGH); // Mặc định OFF
  }
  turnOffAllRelays();
  
  // Khởi tạo CCS811
  if (!ccs.begin(0x5A)) {
    Serial.println("CCS811 not found!");
  } else {
    Serial.println("CCS811 initialized");
  }
  
  connectWifi();
  Serial.println("System ready!");
}

void loop() {
  readSensors();
  unsigned long now = millis();
  
  // Gửi sensor data mỗi 3 giây
  if (now - lastUpdateTime > SEND_INTERVAL) {
    lastUpdateTime = now;
    
    if (wifi_ok) {
      sendSensorDataToESP32();
    } else {
      // Fallback serial
      Serial.print("SENSOR_JSON:{");
      Serial.print("\"ph\":"); Serial.print(fstr(phValue, 15)); Serial.print(",");
      Serial.print("\"tds\":"); Serial.print(fstr(tdsValue, 15)); Serial.print(",");
      Serial.print("\"turbidity\":"); Serial.print(fstr(turbidityValue, 15)); Serial.print(",");
      Serial.print("\"voc\":"); Serial.print(fstr(vocValue, 15));
      Serial.println("}");
    }
  }
  
  // Kiểm tra command mỗi giây
  if (now - lastCommandCheck > COMMAND_INTERVAL) {
    lastCommandCheck = now;
    if (wifi_ok && !manualMode) {
      checkAndApplyCommand();
    }
  }
  
  // Xử lý serial commands
  handleSerialCommands();
  
  delay(50);
}