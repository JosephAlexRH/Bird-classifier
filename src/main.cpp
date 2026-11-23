#include <Arduino.h>
#include <WiFi.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <driver/periph_ctrl.h>

#include "app_config.h"
#include "board_pins.h"
#include "camera_service.h"
#include "camera_runtime.h"
#include "device_health.h"
#include "device_telemetry.h"
#include "motion_detector.h"
#include "status_led.h"
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
    // Capped below the 19.5 dBm default to soften Wi-Fi TX current spikes.
    // 8.5 dBm was too weak to associate at range; 15 dBm is a middle ground.
    WiFi.setTxPower(WIFI_POWER_15dBm);

    // Don't hang boot forever: give up after 20 s and let auto-reconnect keep
    // trying in the background while the camera and web server start.
    const unsigned long started = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - started < 20000) {
      Serial.print(".");
      delay(100);
    }
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println();
      logEvent("wifi: not connected after 20 s (status %d); continuing, driver keeps retrying",
               static_cast<int>(WiFi.status()));
      return;
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

    // setAutoReconnect(true) already retries in the driver. Calling reconnect()
    // on top of it only produces "sta is connecting, return error", so just log.
    lastWifiReconnectAttempt = now;
    Serial.printf("Wi-Fi disconnected (status %d); driver is retrying\n",
                  static_cast<int>(WiFi.status()));  // reasons go to the event log
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
  //
  // On the ESP32-S3 the camera is fed by the LCD_CAM peripheral, not I2S0, and
  // GPIO 32 belongs to the octal flash/PSRAM bus, so it must never be driven.
#if BIRDCAM_CAM_HAS_PWDN
  pinMode(kCamPwdnPin, OUTPUT);  // HIGH = sensor powered down
  digitalWrite(kCamPwdnPin, HIGH);
#endif
#if CONFIG_IDF_TARGET_ESP32S3
  periph_module_reset(PERIPH_LCD_CAM_MODULE);
  periph_module_reset(PERIPH_GDMA_MODULE);
#else
  periph_module_reset(PERIPH_I2S0_MODULE);
#endif
  periph_module_reset(PERIPH_LEDC_MODULE);
#if BIRDCAM_HAS_RGB_LED
  neopixelWrite(kRgbLedPin, 0, 0, 0);  // otherwise it keeps a random power-up colour
#endif

  // 160 MHz is plenty for motion checks and uploads and runs cooler than 240.
  setCpuFrequencyMhz(160);
  Serial.begin(115200);
  initializeDeviceHealth();

  if (!initializeConfigMutex()) {
    Serial.println("Could not initialize configuration mutex");
    while (true) delay(1000);
  }

  mountStorage();
  // Persistent configuration is read once at boot. HTTP updates mutate cfg in
  // RAM and saveConfig() persists that snapshot without a background reload.
  loadConfig();
  startStatusLed();  // needs cfg (status_led) and the reset reason
  initializeUploadBuffer();
  initializeCameraRuntime(cfg.motionEnabled);
  connectToWifi();
  initializeDeviceTelemetry();

  if (!initializeCamera()) {
    // Stay reachable long enough to report the failure, then retry from scratch.
    logEvent("camera: initialization failed; rebooting in 60 s");
    statusLedNoteCameraFailed();
    setCameraRuntimeError("Camera initialization failed");
    startWebServer();
    startDeviceHealthMonitor();
    const unsigned long failedAt = millis();
    while (millis() - failedAt < 60000) {
      handleWebClient();
      delay(10);
    }
    rebootDevice("camera initialization failed");
  }

  startWebServer();
  cameraReady = true;
  if (!startMotionTask()) {
    logEvent("motion: could not start motion task");
    setCameraRuntimeError("Could not start motion task");
  }
  startDeviceHealthMonitor();
}

void loop() {
  handleWebClient();
  maintainCameraPower(getCameraOperatingMode() == CameraOperatingMode::Idle &&
                      !isCameraStreamActive());

  const unsigned long now = millis();
  maintainWifi(now);
}