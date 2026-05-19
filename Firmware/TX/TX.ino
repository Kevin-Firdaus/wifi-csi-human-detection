#include <WiFi.h>
#include "lwip/sockets.h"
#include "lwip/inet.h"
#include "lwip/ip_addr.h"
#include "ping/ping_sock.h"

// Ganti sesuai AP yang akan dibuat RX
#define SSID_RX     "ESP32_CSI_AP"
#define PASS_RX     "csi12345"
#define RX_IP       "192.168.4.1"   // default IP SoftAP ESP32
#define PING_INTERVAL_MS  100

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("TX: Connecting to RX AP...");
  WiFi.mode(WIFI_STA);
  WiFi.begin(SSID_RX, PASS_RX);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nTX: Connected!");
  Serial.print("TX IP: ");
  Serial.println(WiFi.localIP());
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    // Kirim ping ke RX
    esp_ping_config_t config = ESP_PING_DEFAULT_CONFIG();
    ip_addr_t target_ip;
    ipaddr_aton(RX_IP, &target_ip);
    config.target_addr = target_ip;
    config.count = 1;
    config.interval_ms = PING_INTERVAL_MS;

    esp_ping_handle_t ping;
    esp_ping_new_session(&config, NULL, &ping);
    esp_ping_start(ping);

    delay(PING_INTERVAL_MS);
  } else {
    Serial.println("TX: Disconnected, reconnecting...");
    WiFi.reconnect();
    delay(1000);
  }
}
