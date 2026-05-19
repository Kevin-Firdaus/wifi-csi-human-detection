#include "WiFi.h"
#include "esp_wifi.h"

#define AP_SSID  "ESP32_CSI_AP"
#define AP_PASS  "csi12345"

void csi_callback(void *ctx, wifi_csi_info_t *info) {
  if (!info || !info->buf) return;

  int8_t *buf = info->buf;
  int len = info->len;

  Serial.printf("CSI len=%d RSSI=%d noise=%d | ", 
    len, info->rx_ctrl.rssi, info->rx_ctrl.noise_floor);

  // Print semua nilai, bukan hanya 10 pertama
  for (int i = 0; i < len; i++) {
    Serial.printf("%d ", buf[i]);
  }
  Serial.println();
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASS);

  Serial.print("RX AP started. IP: ");
  Serial.println(WiFi.softAPIP());

  wifi_csi_acquire_config_t csi_config = {};
  esp_wifi_set_csi_config(&csi_config);
  esp_wifi_set_csi_rx_cb(csi_callback, NULL);
  esp_wifi_set_csi(true);

  Serial.println("CSI callback registered. Waiting for TX...");
}

void loop() {
  static uint8_t last_count = 0;
  uint8_t count = WiFi.softAPgetStationNum();
  if (count != last_count) {
    Serial.printf("Stations connected: %d\n", count);
    last_count = count;
  }
  delay(500);
}
