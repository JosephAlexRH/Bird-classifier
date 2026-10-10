#include "motion_detector.h"

#include <Arduino.h>
#include <ArduinoJson.h>
#include <HTTPClient.h>
#include <esp_camera.h>
#include <img_converters.h>

#include "app_config.h"
#include "camera_runtime.h"
#include "camera_service.h"
#include "image_storage.h"

namespace {

constexpr int kMotionWidth = 80;
constexpr int kMotionHeight = 60;
constexpr int kMotionPixels = kMotionWidth * kMotionHeight;
constexpr int kMotionRgb565Bytes = kMotionPixels * 2;
constexpr unsigned long kUploadFailureCooldownMs = 15000;

volatile bool motionFlag = false;
uint8_t referenceFrame[kMotionPixels];
uint8_t decodedMotionFrame[kMotionRgb565Bytes];
bool referenceFrameReady = false;
unsigned long lastUploadFailure = 0;
bool uploadCoolingDown = false;
uint32_t motionEventCount = 0;
unsigned long lastMotionMs = 0;
float lastChangedFraction = 0.0f;
bool lastMotionCaptureSuccessful = false;
bool lastMotionUploadSuccessful = false;

bool postFrame(const CapturedJpeg &capture) {
  camera_fb_t *frameBuffer = capture.frameBuffer;
  setCameraRuntimeState(CameraRuntimeState::Uploading);
  lastMotionUploadSuccessful = false;
  HTTPClient http;
  char url[64];
  snprintf(url, sizeof(url), "http://%s:%d/upload", cfg.serverIp, cfg.serverPort);

  http.begin(url);
  http.addHeader("Content-Type", "image/jpeg");
  const String metadata = buildCameraMetadata(capture.metadata, "motion");
  http.addHeader("X-ESP32-Camera-Metadata", metadata);
  // Keep the single-threaded web server responsive when the receiver is down.
  http.setConnectTimeout(1500);
  http.setTimeout(3000);

  const int httpCode = http.POST(frameBuffer->buf, frameBuffer->len);
  if (httpCode <= 0) {
    Serial.printf("POST failed: %s\n", http.errorToString(httpCode).c_str());
    recordClassificationResult(false, false, 0.0f);
    http.end();
    return false;
  }

  Serial.printf("POST %s → %d\n", url, httpCode);
  if (httpCode < 200 || httpCode >= 300) {
    Serial.println("Upload response was not successful; image will not be saved to SD");
    recordClassificationResult(false, false, 0.0f);
    http.end();
    return false;
  }

  lastMotionUploadSuccessful = true;
  acknowledgeRegisterDumpDelivery(capture.metadata);

  JsonDocument response;
  const DeserializationError error = deserializeJson(response, http.getStream());
  if (error || !response["classification"].is<JsonObject>()) {
    Serial.printf("No valid classification in upload response%s%s\n", error ? ": " : "",
                  error ? error.c_str() : "");
    recordClassificationResult(false, false, 0.0f);
    http.end();
    return true;
  }

  const JsonObject classification = response["classification"];
  if (!classification["is_bird"].is<bool>() || !classification["confidence"].is<float>()) {
    Serial.println("Classification lacks boolean is_bird or numeric confidence");
    recordClassificationResult(false, false, 0.0f);
    http.end();
    return true;
  }

  const bool isBird = classification["is_bird"];
  const float confidence = classification["confidence"];
  recordClassificationResult(true, isBird, confidence);
  Serial.printf("Classification: is_bird=%s, confidence=%.3f\n", isBird ? "true" : "false",
                confidence);

  if (cfg.saveBirdsToSd && isBird && confidence >= cfg.birdConfidenceThreshold) {
    const char *captureId = response["capture_id"] | nullptr;
    saveBirdImage(frameBuffer, captureId);
  } else if (isBird && confidence < cfg.birdConfidenceThreshold) {
    Serial.printf("Bird confidence below SD threshold %.3f; image not saved\n",
                  cfg.birdConfidenceThreshold);
  }
  http.end();
  return true;
}

uint8_t motionPixel(int index) {
  const int byteIndex = index * 2;
  const uint16_t rgb565 = static_cast<uint16_t>(decodedMotionFrame[byteIndex]) |
                          (static_cast<uint16_t>(decodedMotionFrame[byteIndex + 1]) << 8);
  const uint8_t red = static_cast<uint8_t>(((rgb565 >> 11) & 0x1f) * 255 / 31);
  const uint8_t green = static_cast<uint8_t>(((rgb565 >> 5) & 0x3f) * 255 / 63);
  const uint8_t blue = static_cast<uint8_t>((rgb565 & 0x1f) * 255 / 31);
  return static_cast<uint8_t>((77U * red + 150U * green + 29U * blue) >> 8);
}

}  // namespace

void checkMotion() {
  if (!cfg.motionEnabled || getCameraOperatingMode() != CameraOperatingMode::Motion ||
      isCameraStreamActive()) {
    return;
  }

  bool modeChanged = false;
  if (!configureMotionCapture(modeChanged)) {
    setCameraRuntimeError("Could not configure motion capture mode");
    return;
  }
  if (modeChanged) {
    // A buffer already in flight can still use the previous JPEG mode.
    if (!drainCameraBuffer()) {
      setCameraRuntimeError("Could not drain stale motion frame");
      return;
    }
  }

  camera_fb_t *frameBuffer = acquireCameraFrame();
  if (!isCompleteJpeg(frameBuffer) || frameBuffer->width != 160 || frameBuffer->height != 120) {
    if (frameBuffer != nullptr) {
      esp_camera_fb_return(frameBuffer);
    }
    setCameraRuntimeError("Motion JPEG frame capture failed");
    return;
  }

  if (!jpg2rgb565(frameBuffer->buf, frameBuffer->len, decodedMotionFrame, JPG_SCALE_2X)) {
    esp_camera_fb_return(frameBuffer);
    setCameraRuntimeError("Motion JPEG decode failed");
    return;
  }
  esp_camera_fb_return(frameBuffer);

  if (!referenceFrameReady) {
    for (int index = 0; index < kMotionPixels; ++index) {
      referenceFrame[index] = motionPixel(index);
    }
    referenceFrameReady = true;
    clearCameraRuntimeError();
    setCameraRuntimeState(isMotionUploadCoolingDown() ? CameraRuntimeState::UploadCooldown
                                                      : CameraRuntimeState::MotionArmed);
    return;
  }

  clearCameraRuntimeError();

  int changedPixels = 0;
  for (int index = 0; index < kMotionPixels; ++index) {
    const uint8_t currentPixel = motionPixel(index);

    if (abs(static_cast<int>(currentPixel) - static_cast<int>(referenceFrame[index])) >
        cfg.motionThreshold) {
      ++changedPixels;
    }

    referenceFrame[index] =
        static_cast<uint8_t>(0.8f * referenceFrame[index] + 0.2f * currentPixel);
  }

  const float changedFraction = static_cast<float>(changedPixels) / kMotionPixels;
  lastChangedFraction = changedFraction;
  if (changedFraction <= cfg.motionSensitivity) {
    setCameraRuntimeState(isMotionUploadCoolingDown() ? CameraRuntimeState::UploadCooldown
                                                      : CameraRuntimeState::MotionArmed);
    return;
  }

  motionFlag = true;
  ++motionEventCount;
  lastMotionMs = millis();
  lastMotionCaptureSuccessful = false;
  lastMotionUploadSuccessful = false;
  setCameraRuntimeState(CameraRuntimeState::MotionTriggered);
  Serial.printf("Motion detected: %.1f%% pixels changed\n", changedFraction * 100);

  const unsigned long now = millis();
  if (uploadCoolingDown && now - lastUploadFailure < kUploadFailureCooldownMs) {
    Serial.println("Receiver unavailable; skipping upload during retry cooldown");
    setCameraRuntimeState(CameraRuntimeState::UploadCooldown);
    return;
  }
  uploadCoolingDown = false;

  CapturedJpeg capture;
  setCameraRuntimeState(CameraRuntimeState::Capturing);
  if (captureJpeg(capture)) {
    lastMotionCaptureSuccessful = true;
    if (!postFrame(capture)) {
      lastUploadFailure = millis();
      uploadCoolingDown = true;
      setCameraRuntimeState(CameraRuntimeState::UploadCooldown);
    } else {
      clearCameraRuntimeError();
    }
    esp_camera_fb_return(capture.frameBuffer);
  } else {
    setCameraRuntimeError("Motion JPEG capture failed");
  }
  resetMotionReference();
}

void armMotionDetection() {
  cfg.motionEnabled = true;
  selectCameraOperatingMode(CameraOperatingMode::Motion);
  resetMotionReference();
  clearCameraRuntimeError();
  setCameraRuntimeState(CameraRuntimeState::MotionArming);
}

void disarmMotionDetection() {
  cfg.motionEnabled = false;
  selectCameraOperatingMode(CameraOperatingMode::Idle);
  resetMotionReference();
  clearCameraRuntimeError();
  setCameraRuntimeState(CameraRuntimeState::Idle);
}

void resetMotionReference() {
  referenceFrameReady = false;
  const CameraRuntimeState state = getCameraRuntimeState();
  if (getCameraOperatingMode() == CameraOperatingMode::Motion && !isCameraStreamActive() &&
      state != CameraRuntimeState::Error && state != CameraRuntimeState::UploadCooldown) {
    setCameraRuntimeState(CameraRuntimeState::MotionArming);
  }
}

bool consumeMotionFlag() {
  const bool wasSet = motionFlag;
  motionFlag = false;
  return wasSet;
}

bool isMotionFlagSet() {
  return motionFlag;
}

bool isReferenceFrameReady() {
  return referenceFrameReady;
}

uint32_t getMotionEventCount() { return motionEventCount; }

unsigned long getLastMotionMs() { return lastMotionMs; }

float getLastChangedFraction() { return lastChangedFraction; }

bool wasLastMotionCaptureSuccessful() { return lastMotionCaptureSuccessful; }

bool wasLastMotionUploadSuccessful() { return lastMotionUploadSuccessful; }

bool isMotionUploadCoolingDown() {
  return uploadCoolingDown && millis() - lastUploadFailure < kUploadFailureCooldownMs;
}