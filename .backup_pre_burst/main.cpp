#include <Arduino.h>
#include <WiFi.h>

#include "app_config.h"
#include "camera_service.h"
#include "camera_runtime.h"
#include "device_telemetry.h"
#include "motion_detector.h"
#include "secrets.h"
#include "web_server.h"

namespace {

  constexpr unsigned long kConfigPollMs = 10000;
  constexpr unsigned long kWifiReconnectMs = 10000;

  unsigned long lastMotionCheck = 0;
  unsigned long lastConfigCheck = 0;
  unsigned long lastWifiReconnectAttempt = 0;

  void connectToWifi() {
    WiFi.mode(WIFI_STA);
    Serial.print("Connecting to ");
    Serial.println(WIFI_SSID);
    WiFi.setAutoReconnect(true);
    WiFi.persistent(false);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    while (WiFi.status() != WL_CONNECTED) {
      Serial.print(".");
      delay(100);
    }

    Serial.println("\nConnected");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
  }

  void maintainWifi(unsigned long now) {
    if (WiFi.status() == WL_CONNECTED || now - lastWifiReconnectAttempt < kWifiReconnectMs) {
      return;
    }

    lastWifiReconnectAttempt = now;
    Serial.println("Wi-Fi disconnected; reconnecting...");
    WiFi.reconnect();
  }

}  // namespace

void setup() {
  Serial.begin(115200);

  mountStorage();
  loadConfig();
  initializeCameraRuntime(cfg.motionEnabled);
  connectToWifi();
  initializeDeviceTelemetry();

  if (!initializeCamera()) {
    while (true) {
      delay(1000);
    }
  }

  startWebServer();
}

void loop() {
  handleWebClient();

  const unsigned long now = millis();
  maintainWifi(now);

  if (WiFi.status() == WL_CONNECTED &&
      getCameraOperatingMode() == CameraOperatingMode::Motion && !isCameraStreamActive() &&
      now - lastMotionCheck >= static_cast<unsigned long>(cfg.motionCheckMs)) {
    lastMotionCheck = now;
    checkMotion();
  }

  if (isConfigStorageReady() && now - lastConfigCheck >= kConfigPollMs) {
    lastConfigCheck = now;
    const Config previousConfig = cfg;
    loadConfig();
    if (cfg.motionEnabled != previousConfig.motionEnabled) {
      if (cfg.motionEnabled) {
        armMotionDetection();
      } else {
        disarmMotionDetection();
      }
    } else if (getCameraOperatingMode() == CameraOperatingMode::Motion &&
               (cfg.motionThreshold != previousConfig.motionThreshold ||
                cfg.motionSensitivity != previousConfig.motionSensitivity ||
                cfg.motionCheckMs != previousConfig.motionCheckMs ||
                cfg.jpegQuality != previousConfig.jpegQuality ||
                cfg.framesize != previousConfig.framesize)) {
      resetMotionReference();
    }
    const bool captureSettingsChanged =
        cfg.jpegQuality != previousConfig.jpegQuality || cfg.framesize != previousConfig.framesize;
    if (captureSettingsChanged) {
      markRegisterDumpPending();
      if (getCameraOperatingMode() == CameraOperatingMode::Idle) applyCameraSettings();
    }
  }
}