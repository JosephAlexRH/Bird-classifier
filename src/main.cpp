#include <Arduino.h>
#include <WiFi.h>

#include "app_config.h"
#include "camera_service.h"
#include "camera_runtime.h"
#include "device_telemetry.h"
#include "motion_detector.h"
#include "upload_buffer.h"
#include "web_server.h"

namespace {

  constexpr unsigned long kConfigPollMs = 10000;
  constexpr unsigned long kWifiReconnectMs = 10000;
  constexpr unsigned long kBufferFlushMs = 2000;

  unsigned long lastMotionCheck = 0;
  unsigned long lastConfigCheck = 0;
  unsigned long lastWifiReconnectAttempt = 0;
  unsigned long lastBufferFlush = 0;

  void connectToWifi() {
    if (cfg.wifiSsid[0] == '\0') {
      Serial.println("Wi-Fi not configured; set wifi_ssid in config.json or provide include/secrets.h");
      return;
    }
    WiFi.mode(WIFI_STA);
    Serial.print("Connecting to ");
    Serial.println(cfg.wifiSsid);
    WiFi.setAutoReconnect(true);
    WiFi.persistent(false);
    WiFi.begin(cfg.wifiSsid, cfg.wifiPassword);

    while (WiFi.status() != WL_CONNECTED) {
      Serial.print(".");
      delay(100);
    }

    Serial.println("\nConnected");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
  }

  void maintainWifi(unsigned long now) {
    if (cfg.wifiSsid[0] == '\0' || WiFi.status() == WL_CONNECTED ||
        now - lastWifiReconnectAttempt < kWifiReconnectMs) {
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
  initializeUploadBuffer();
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

  // With the SD buffer available, motion capture keeps running through Wi-Fi drops.
  const bool wifiUp = WiFi.status() == WL_CONNECTED;
  if ((wifiUp || (cfg.bufferUploadsToSd && sdReady)) &&
      getCameraOperatingMode() == CameraOperatingMode::Motion && !isCameraStreamActive() &&
      now - lastMotionCheck >= static_cast<unsigned long>(cfg.motionCheckMs)) {
    lastMotionCheck = now;
    checkMotion();
  }

  // One buffered frame per pass so motion checks run between deliveries.
  if (wifiUp && !isCameraStreamActive() && now - lastBufferFlush >= kBufferFlushMs) {
    lastBufferFlush = now;
    flushOneBufferedUpload();
  }

  if (isConfigStorageReady() && now - lastConfigCheck >= kConfigPollMs) {
    lastConfigCheck = now;
    const Config previousConfig = cfg;
    loadConfig();
    const bool wifiSettingsChanged = strcmp(cfg.wifiSsid, previousConfig.wifiSsid) != 0 ||
                                     strcmp(cfg.wifiPassword, previousConfig.wifiPassword) != 0;
    if (wifiSettingsChanged) {
      Serial.println("Wi-Fi configuration changed; reconnecting");
      WiFi.disconnect();
      connectToWifi();
    }
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
                 cfg.framesize != previousConfig.framesize ||
                 cfg.motionRoiX != previousConfig.motionRoiX ||
                 cfg.motionRoiY != previousConfig.motionRoiY ||
                 cfg.motionRoiWidth != previousConfig.motionRoiWidth ||
                 cfg.motionRoiHeight != previousConfig.motionRoiHeight ||
                 cfg.motionExposureSettleFrames != previousConfig.motionExposureSettleFrames ||
                 cfg.motionExposureSettleTolerance != previousConfig.motionExposureSettleTolerance ||
                 cfg.motionExposureShiftTolerance != previousConfig.motionExposureShiftTolerance ||
                 cfg.motionBlockColumns != previousConfig.motionBlockColumns ||
                 cfg.motionBlockRows != previousConfig.motionBlockRows ||
                 cfg.motionRatioThreshold != previousConfig.motionRatioThreshold ||
                 cfg.motionClipLow != previousConfig.motionClipLow ||
                 cfg.motionClipHigh != previousConfig.motionClipHigh ||
                 cfg.motionMaxClippedFraction != previousConfig.motionMaxClippedFraction ||
                 cfg.motionMinValidBlocks != previousConfig.motionMinValidBlocks ||
                 cfg.motionMaxChangedFraction != previousConfig.motionMaxChangedFraction)) {
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