void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("✅ ESP32 พร้อมใช้งาน!");
}

void loop() {
  Serial.println("Hello from ESP32...");
  delay(1000); // รอ 1 วินาที
}
