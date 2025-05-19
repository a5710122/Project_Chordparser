#include <Arduino.h>
#include <WiFi.h>
#include <ESP_I2S.h>      // ต้องติดตั้งไลบรารี ESP_I2S (จาก Library Manager)
#include "secrets.h"      // ใส่ WiFi SSID/PASSWORD

#define LED_PIN  (GPIO_NUM_22)         // ไฟ LED แสดงสถานะ
#define I2S_WS   (GPIO_NUM_15)         // Word Select (WS)
#define I2S_SD   (GPIO_NUM_13)         // Serial Data (SD)
#define I2S_SCK  (GPIO_NUM_2)          // Serial Clock (SCK)

#define I2S_PORT (I2S_NUM_0)
#define SAMPLE_RATE     (16000)
#define BUF_LEN         (512)
#define SERVER_PORT     (9000)

static I2SClass i2s;
static int16_t i2s_buf[BUF_LEN];
WiFiServer server(SERVER_PORT);

void initWiFi() {
  Serial.println("\nConnecting to WiFi...");
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print(".");
  }
  Serial.println("\n✅ Connected!");
  Serial.print("📡 IP Address: ");
  Serial.println(WiFi.localIP());
  Serial.printf("🟢 Starting TCP server on port %d\n", SERVER_PORT);
  server.begin();
}

void initI2S() {
  i2s_data_bit_width_t i2s_bitwidth     = I2S_DATA_BIT_WIDTH_32BIT;
  i2s_mode_t           i2s_mode         = I2S_MODE_STD;
  i2s_slot_mode_t      i2s_slot         = I2S_SLOT_MODE_MONO;
  int8_t               i2s_slot_mask    = I2S_STD_SLOT_LEFT;
  i2s_rx_transform_t   i2s_rx_transform = I2S_RX_TRANSFORM_32_TO_16;

  i2s.setPins(I2S_SCK, I2S_WS, -1, I2S_SD);  // sck, ws, out, in

  if (!i2s.begin(i2s_mode, SAMPLE_RATE, i2s_bitwidth, i2s_slot, i2s_slot_mask)) {
    Serial.println("❌ Failed to initialize I2S bus!");
    return;
  }

  if (!i2s.configureRX(SAMPLE_RATE, i2s_bitwidth, i2s_slot, i2s_rx_transform)) {
    Serial.println("❌ Failed to configure I2S RX!");
    return;
  }

  Serial.println("✅ I2S Mic Initialized");
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=== I2S Mic TCP Server ===");
  pinMode(LED_PIN, OUTPUT);
  initWiFi();
  initI2S();
}

void loop() {
  WiFiClient client = server.available();
  if (client) {
    Serial.println("✅ Client connected");
    client.println("ready");
    while (client.connected()) {
      gpio_set_level(LED_PIN, 1);
      size_t bytes_read = i2s.readBytes((char*)i2s_buf, sizeof(i2s_buf));
      gpio_set_level(LED_PIN, 0);
      
      Serial.print("Read bytes: ");
      Serial.println(bytes_read);  // <-- log จำนวน byte ที่อ่านได้

      if (bytes_read > 0) {
        Serial.print("Sample[0]: ");
        Serial.println(i2s_buf[0]);  // <-- log ค่าจริงจากไมค์
        client.write((const char *)i2s_buf, bytes_read);
      } else {
        Serial.println("❌ No bytes read from I2S");
        delay(100);
      }
    }
    Serial.println("❌ Client disconnected");
  }
}

