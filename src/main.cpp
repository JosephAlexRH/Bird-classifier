#include <Arduino.h>
#include <WiFi.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <driver/periph_ctrl.h>

#include "app_config.h"
#include "camera_service.h"
#include "camera_runtime.h"
#include "device_telemetry.h"
#include "motion_detector.h"
#include "upload_buffer.h"
#include "web_server.h"

namespace {

  constexpr unsigned long kWifiReconnectMs = 10000;
  constexpr unsigned long kBufferFlushMs = 2000;
  constexpr uint32_t kMotionTaskStackBytes = 8192;
  constexpr UBaseType_t kMotionTaskPriority = 1;

  unsigned long lastWifiReconnectAttempt = 0;
  volatile bool cameraReady = false;
  TaskHandle_t motionTaskHandle = nullptr;

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
    // DIAGNOSTIC: cap transmit power to cut the current spikes from Wi-Fi TX.
    // If the crashes stop with this, the board's power supply is the problem.
    WiFi.setTxPower(WIFI_POWER_8_5dBm);

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

  void motionTask(void *parameter) {
    (void)parameter;
    while (!cameraReady) {
      vTaskDelay(pdMS_TO_TICKS(100));
    }

    bool stackWatermarkLogged = false;
    unsigned long lastBufferFlush = 0;
    while (cameraReady) {
      int delayMs = 100;
      {
        ConfigMutexGuard configLock;
        if (configLock.locked()) {
          const bool wifiUp = WiFi.status() == WL_CONNECTED;
          if ((wifiUp || (cfg.bufferUploadsToSd && sdReady)) && isFirmwareMotionOwner() &&
              getCameraOperatingMode() == CameraOperatingMode::Motion &&
              !isCameraStreamActive()) {
            checkMotion();
          }

          const unsigned long now = millis();
          if (isFirmwareMotionOwner() && wifiUp && !isCameraStreamActive() &&
              now - lastBufferFlush >= kBufferFlushMs) {
            lastBufferFlush = now;
            flushOneBufferedUpload();
          }

          if (!stackWatermarkLogged) {
            const UBaseType_t remainingStackBytes = uxTaskGetStackHighWaterMark(nullptr);
            Serial.printf("Motion task running on core %d with %u-byte stack; "
                          "minimum free stack after first loop: %u bytes\n",
                          xPortGetCoreID(), static_cast<unsigned int>(kMotionTaskStackBytes),
                          static_cast<unsigned int>(remainingStackBytes));
            stackWatermarkLogged = true;
          }

          delayMs = constrain(cfg.motionCheckMs, 100, 60000);
        }
      }
      vTaskDelay(pdMS_TO_TICKS(delayMs));
    }

    motionTaskHandle = nullptr;
    vTaskDelete(nullptr);
  }

  bool startMotionTask() {
    if (motionTaskHandle != nullptr) return true;
    const BaseType_t result =
        xTaskCreatePinnedToCore(motionTask, "motion", kMotionTaskStackBytes, nullptr,
                                kMotionTaskPriority, &motionTaskHandle, ARDUINO_RUNNING_CORE);
    if (result == pdPASS) return true;

    motionTaskHandle = nullptr;
    Serial.printf("Failed to create motion task: %ld\n", static_cast<long>(result));
    return false;
  }

}  // namespace

void setup() {
  // A panic reboot (rst:0xc SW_CPU_RESET) only resets the CPUs, timers, SPI
  // and UART. The camera's I2S0 DMA and the LEDC XCLK keep running from the
  // previous boot and can keep writing frames into RAM the new boot has
  // already handed to the Wi-Fi driver. Power the sensor down and reset both
  // peripherals as early as possible.
  pinMode(32, OUTPUT);  // camera PWDN (kPowerDownPin); HIGH = powered down
  digitalWrite(32, HIGH);
  periph_module_reset(PERIPH_I2S0_MODULE);
  periph_module_reset(PERIPH_LEDC_MODULE);

  Serial.begin(115200);
  Serial.printf("Reset reason: %d\n", static_cast<int>(esp_reset_reason()));

  if (!initializeConfigMutex()) {
    Serial.println("Could not initialize configuration mutex");
    while (true) delay(1000);
  }

  mountStorage();
  // Persistent configuration is read once at boot. HTTP updates mutate cfg in
  // RAM and saveConfig() persists that snapshot without a background reload.
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
  cameraReady = true;
  if (!startMotionTask()) {
    setCameraRuntimeError("Could not start motion task");
  }
}

void loop() {
  handleWebClient();

  const unsigned long now = millis();
  maintainWifi(now);
}