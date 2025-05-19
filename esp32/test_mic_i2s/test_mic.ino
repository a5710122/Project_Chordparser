#include <driver/i2s.h>

#define I2S_WS 15     // Word Select (L/R)
#define I2S_SD 32     // Serial Data (from Mic)
#define I2S_SCK 14    // Serial Clock

#define SAMPLE_RATE     16000
#define I2S_BUFFER_SIZE 1024

void setupI2SMic() {
  const i2s_config_t i2s_config = {
    .mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,  // รับแค่ข้างเดียว
    .communication_format = I2S_COMM_FORMAT_I2S_MSB,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 4,
    .dma_buf_len = 256,
    .use_apll = false
  };

  const i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
  };

  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pin_config);
  i2s_zero_dma_buffer(I2S_NUM_0);
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  setupI2SMic();
  Serial.println("Ready to capture I2S audio...");
}

void loop() {
  int16_t samples[I2S_BUFFER_SIZE];
  size_t bytes_read;

  // อ่านเสียงจาก I2S (ขนาด buffer = 1024 sample)
  i2s_read(I2S_NUM_0, &samples, sizeof(samples), &bytes_read, portMAX_DELAY);

  int sample_count = bytes_read / sizeof(int16_t);

  // ส่งแต่ละ sample ผ่าน Serial
  for (int i = 0; i < sample_count; i++) {
    Serial.println(samples[i]);
  }
}
