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
#include "upload_buffer.h"

#include <WiFi.h>

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
int consecutiveTriggers = 0;
bool exposureSettled = false;
int exposureSettleCount = 0;
float exposureSettleMinimum = 0.0f;
float exposureSettleMaximum = 0.0f;
bool currentExposureProductValid = false;
float currentExposureProduct = 0.0f;
bool referenceExposureProductValid = false;
float referenceExposureProduct = 0.0f;
float lastExposureShift = 0.0f;
float lastGlobalRatio = 1.0f;
int lastValidBlocks = 0;
int lastClippedBlocks = 0;
int lastChangedBlocks = 0;
const char *lastDecision = "not_started";
float blockRatios[kMaxMotionBlocks];
float blockReferenceMeans[kMaxMotionBlocks];
float blockCurrentMeans[kMaxMotionBlocks];

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
  // With wait_for_classification off the receiver stores the JPEG, queues its
  // own inference, and answers at once so the detector can re-arm sooner.
  const bool deferClassification = !cfg.waitForClassification;
  if (deferClassification) http.addHeader("X-ESP32-Classification", "deferred");
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

  if (deferClassification) {
    // No decision comes back in this mode, so there is nothing to save to SD.
    recordClassificationResult(false, false, 0.0f);
    http.end();
    return true;
  }

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

void copyCurrentToReference(const ExposureGainSample &exposure) {
  for (int index = 0; index < kMotionPixels; ++index) {
    referenceFrame[index] = motionPixel(index);
  }
  referenceFrameReady = true;
  referenceExposureProductValid = exposure.valid;
  referenceExposureProduct = exposure.valid ? exposure.product : 0.0f;
}

void restartExposureSettling() {
  exposureSettled = false;
  exposureSettleCount = 0;
  exposureSettleMinimum = 0.0f;
  exposureSettleMaximum = 0.0f;
  lastDecision = "settling_exposure";
  const CameraRuntimeState state = getCameraRuntimeState();
  if (getCameraOperatingMode() == CameraOperatingMode::Motion && !isCameraStreamActive() &&
      state != CameraRuntimeState::Error && state != CameraRuntimeState::UploadCooldown) {
    setCameraRuntimeState(CameraRuntimeState::MotionArming);
  }
}

bool updateExposureSettling(const ExposureGainSample &exposure) {
  currentExposureProductValid = exposure.valid;
  currentExposureProduct = exposure.valid ? exposure.product : 0.0f;
  if (!exposure.valid) {
    restartExposureSettling();
    lastDecision = "invalid_exposure";
    return false;
  }

  if (exposureSettleCount == 0) {
    exposureSettleMinimum = exposure.product;
    exposureSettleMaximum = exposure.product;
    exposureSettleCount = 1;
  } else {
    const float candidateMinimum = min(exposureSettleMinimum, exposure.product);
    const float candidateMaximum = max(exposureSettleMaximum, exposure.product);
    const float range = candidateMaximum / candidateMinimum - 1.0f;
    if (range <= cfg.motionExposureSettleTolerance) {
      exposureSettleMinimum = candidateMinimum;
      exposureSettleMaximum = candidateMaximum;
      ++exposureSettleCount;
    } else {
      exposureSettleMinimum = exposure.product;
      exposureSettleMaximum = exposure.product;
      exposureSettleCount = 1;
    }
  }

  exposureSettled = exposureSettleCount >= cfg.motionExposureSettleFrames;
  lastDecision = exposureSettled ? "exposure_settled" : "settling_exposure";
  return exposureSettled;
}

float relativeExposureShift(float first, float second) {
  const float denominator = max(first, second);
  return denominator > 0.0f ? fabsf(first - second) / denominator : 1.0f;
}

float median(float *values, int count) {
  for (int index = 1; index < count; ++index) {
    const float value = values[index];
    int position = index;
    while (position > 0 && values[position - 1] > value) {
      values[position] = values[position - 1];
      --position;
    }
    values[position] = value;
  }
  if ((count & 1) != 0) return values[count / 2];
  return 0.5f * (values[count / 2 - 1] + values[count / 2]);
}

bool isSupportedMotionFrame(const camera_fb_t *frameBuffer) {
  if (frameBuffer == nullptr) return false;
  return (frameBuffer->width == 160 && frameBuffer->height == 120) ||
         (frameBuffer->width == 320 && frameBuffer->height == 240) ||
         (frameBuffer->width == 640 && frameBuffer->height == 480);
}

bool decodeMotionFrame(const camera_fb_t *frameBuffer) {
  if (frameBuffer->width == 160 && frameBuffer->height == 120) {
    return jpg2rgb565(frameBuffer->buf, frameBuffer->len, decodedMotionFrame, JPG_SCALE_2X);
  }
  if (frameBuffer->width == 320 && frameBuffer->height == 240) {
    return jpg2rgb565(frameBuffer->buf, frameBuffer->len, decodedMotionFrame, JPG_SCALE_4X);
  }
  if (frameBuffer->width == 640 && frameBuffer->height == 480) {
    return jpg2rgb565(frameBuffer->buf, frameBuffer->len, decodedMotionFrame, JPG_SCALE_8X);
  }
  return false;
}

void effectiveMotionRoi(int &x, int &y, int &width, int &height) {
  x = constrain(static_cast<int>(floorf(cfg.motionRoiX * kMotionWidth)), 0, kMotionWidth - 1);
  y = constrain(static_cast<int>(floorf(cfg.motionRoiY * kMotionHeight)), 0, kMotionHeight - 1);
  const int right = constrain(static_cast<int>(ceilf((cfg.motionRoiX + cfg.motionRoiWidth) *
                                                       kMotionWidth)),
                              x + 1, kMotionWidth);
  const int bottom = constrain(static_cast<int>(ceilf((cfg.motionRoiY + cfg.motionRoiHeight) *
                                                        kMotionHeight)),
                               y + 1, kMotionHeight);
  width = right - x;
  height = bottom - y;
}

bool analyzeBlocks(float &changedFraction) {
  int roiX = 0;
  int roiY = 0;
  int roiWidth = 0;
  int roiHeight = 0;
  effectiveMotionRoi(roiX, roiY, roiWidth, roiHeight);
  const int columns = min(cfg.motionBlockColumns, roiWidth);
  const int rows = min(cfg.motionBlockRows, roiHeight);
  int validBlocks = 0;
  int clippedBlocks = 0;

  for (int row = 0; row < rows; ++row) {
    const int top = roiY + row * roiHeight / rows;
    const int bottom = roiY + (row + 1) * roiHeight / rows;
    for (int column = 0; column < columns; ++column) {
      const int left = roiX + column * roiWidth / columns;
      const int right = roiX + (column + 1) * roiWidth / columns;
      uint32_t referenceSum = 0;
      uint32_t currentSum = 0;
      int referenceClipped = 0;
      int currentClipped = 0;
      int pixels = 0;
      for (int y = top; y < bottom; ++y) {
        for (int x = left; x < right; ++x) {
          const int index = y * kMotionWidth + x;
          const uint8_t reference = referenceFrame[index];
          const uint8_t current = motionPixel(index);
          referenceSum += reference;
          currentSum += current;
          if (reference <= cfg.motionClipLow || reference >= cfg.motionClipHigh) ++referenceClipped;
          if (current <= cfg.motionClipLow || current >= cfg.motionClipHigh) ++currentClipped;
          ++pixels;
        }
      }

      const float referenceClippedFraction = static_cast<float>(referenceClipped) / pixels;
      const float currentClippedFraction = static_cast<float>(currentClipped) / pixels;
      const float referenceMean = static_cast<float>(referenceSum) / pixels;
      if (referenceClippedFraction > cfg.motionMaxClippedFraction ||
          currentClippedFraction > cfg.motionMaxClippedFraction || referenceMean <= 1.0f) {
        ++clippedBlocks;
        continue;
      }
      const float currentMean = static_cast<float>(currentSum) / pixels;
      blockReferenceMeans[validBlocks] = referenceMean;
      blockCurrentMeans[validBlocks] = currentMean;
      blockRatios[validBlocks] = currentMean / referenceMean;
      ++validBlocks;
    }
  }

  lastValidBlocks = validBlocks;
  lastClippedBlocks = clippedBlocks;
  lastChangedBlocks = 0;
  changedFraction = 0.0f;
  if (validBlocks < cfg.motionMinValidBlocks) return false;

  lastGlobalRatio = median(blockRatios, validBlocks);
  if (lastGlobalRatio <= 0.0f || !isfinite(lastGlobalRatio)) return false;
  for (int block = 0; block < validBlocks; ++block) {
    const float ratioDeviation = fabsf(blockCurrentMeans[block] /
                                           (blockReferenceMeans[block] * lastGlobalRatio) -
                                       1.0f);
    const float luminanceResidual =
        fabsf(blockCurrentMeans[block] - lastGlobalRatio * blockReferenceMeans[block]);
    if (ratioDeviation > cfg.motionRatioThreshold && luminanceResidual > cfg.motionThreshold) {
      ++lastChangedBlocks;
    }
  }
  changedFraction = static_cast<float>(lastChangedBlocks) / validBlocks;
  return true;
}

}  // namespace

void checkMotion() {
  if (!isFirmwareMotionOwner()) return;
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
    // A frame-size transition restarts AEC/AGC convergence. Keep an existing
    // background when requested, but never compare against a transition frame.
    restartExposureSettling();
  }

  // The queued frame was grabbed just after the previous check and is a whole
  // check interval old. A mode change has already drained it.
  if (!modeChanged && !discardQueuedCameraFrame()) {
    setCameraRuntimeError("Could not discard stale motion frame");
    return;
  }

  camera_fb_t *frameBuffer = acquireCameraFrame();
  if (!isCompleteJpeg(frameBuffer) || !isSupportedMotionFrame(frameBuffer)) {
    if (frameBuffer != nullptr) {
      esp_camera_fb_return(frameBuffer);
    }
    setCameraRuntimeError("Motion JPEG frame capture failed");
    return;
  }

  if (!decodeMotionFrame(frameBuffer)) {
    esp_camera_fb_return(frameBuffer);
    setCameraRuntimeError("Motion JPEG decode failed");
    return;
  }
  // Read AEC/AGC while this exact framebuffer is retained. This is a post-frame,
  // non-atomic sensor observation, so several consecutive products must agree.
  const ExposureGainSample exposure = sampleExposureGain();
  esp_camera_fb_return(frameBuffer);

  if (!exposureSettled && !updateExposureSettling(exposure)) {
    clearCameraRuntimeError();
    setCameraRuntimeState(CameraRuntimeState::MotionArming);
    return;
  }

  currentExposureProductValid = exposure.valid;
  currentExposureProduct = exposure.valid ? exposure.product : 0.0f;
  if (!exposure.valid) {
    restartExposureSettling();
    lastDecision = "invalid_exposure";
    clearCameraRuntimeError();
    return;
  }

  if (!referenceFrameReady || !referenceExposureProductValid) {
    copyCurrentToReference(exposure);
    lastDecision = "reference_acquired";
    clearCameraRuntimeError();
    setCameraRuntimeState(isMotionUploadCoolingDown() ? CameraRuntimeState::UploadCooldown
                                                      : CameraRuntimeState::MotionArmed);
    return;
  }

  clearCameraRuntimeError();

  lastExposureShift = relativeExposureShift(exposure.product, referenceExposureProduct);
  if (lastExposureShift > cfg.motionExposureShiftTolerance) {
    copyCurrentToReference(exposure);
    consecutiveTriggers = 0;
    lastChangedFraction = 0.0f;
    lastDecision = "exposure_shift";
    setCameraRuntimeState(isMotionUploadCoolingDown() ? CameraRuntimeState::UploadCooldown
                                                      : CameraRuntimeState::MotionArmed);
    return;
  }

  float changedFraction = 0.0f;
  if (!analyzeBlocks(changedFraction)) {
    copyCurrentToReference(exposure);
    consecutiveTriggers = 0;
    lastChangedFraction = 0.0f;
    lastDecision = "insufficient_valid_blocks";
    setCameraRuntimeState(isMotionUploadCoolingDown() ? CameraRuntimeState::UploadCooldown
                                                      : CameraRuntimeState::MotionArmed);
    return;
  }
  lastChangedFraction = changedFraction;
  if (changedFraction > cfg.motionMaxChangedFraction) {
    copyCurrentToReference(exposure);
    consecutiveTriggers = 0;
    lastDecision = "lighting_change";
    setCameraRuntimeState(isMotionUploadCoolingDown() ? CameraRuntimeState::UploadCooldown
                                                      : CameraRuntimeState::MotionArmed);
    return;
  }

  if (changedFraction <= cfg.motionSensitivity) {
    // Keep the image and its exposure product from the same accepted quiet frame.
    copyCurrentToReference(exposure);
    consecutiveTriggers = 0;
    lastDecision = "quiet";
    setCameraRuntimeState(isMotionUploadCoolingDown() ? CameraRuntimeState::UploadCooldown
                                                      : CameraRuntimeState::MotionArmed);
    return;
  }

  lastDecision = "motion";
  motionFlag = true;
  ++motionEventCount;
  lastMotionMs = millis();
  lastMotionCaptureSuccessful = false;
  lastMotionUploadSuccessful = false;
  setCameraRuntimeState(CameraRuntimeState::MotionTriggered);
  Serial.printf("Motion detected: %d/%d blocks changed (%.1f%%), global ratio %.3f\n",
                lastChangedBlocks, lastValidBlocks, changedFraction * 100, lastGlobalRatio);

  const bool canBuffer = cfg.bufferUploadsToSd && sdReady;
  bool offline = WiFi.status() != WL_CONNECTED || isMotionUploadCoolingDown();
  if (offline && !canBuffer) {
    Serial.println("Receiver unavailable; skipping upload during retry cooldown");
    setCameraRuntimeState(CameraRuntimeState::UploadCooldown);
    return;
  }
  if (!offline) uploadCoolingDown = false;

  // One framebuffer means the burst is capture -> upload -> capture; the sensor
  // stays in the full-resolution mode for every frame after the first.
  const int burstCount = constrain(cfg.burstCount, 1, kMaxBurstCount);
  for (int shot = 0; shot < burstCount; ++shot) {
    if (shot > 0) {
      delay(constrain(cfg.burstIntervalMs, 0, kMaxBurstIntervalMs));
      // Drop the frame queued while the previous one was uploading.
      if (!discardQueuedCameraFrame()) {
        setCameraRuntimeError("Could not discard stale burst frame");
        break;
      }
    }

    CapturedJpeg capture;
    setCameraRuntimeState(CameraRuntimeState::Capturing);
    if (!captureJpeg(capture)) {
      setCameraRuntimeError("Motion JPEG capture failed");
      break;
    }
    lastMotionCaptureSuccessful = true;
    bool delivered = false;
    if (!offline) {
      delivered = postFrame(capture);
      if (!delivered) {
        noteMotionUploadFailure();
        offline = true;  // The rest of this burst goes straight to SD.
      }
    }
    const bool buffered =
        !delivered && bufferUpload(capture.frameBuffer, buildCameraMetadata(capture.metadata, "motion"));
    esp_camera_fb_return(capture.frameBuffer);
    if (!delivered && !buffered) break;
    clearCameraRuntimeError();
    Serial.printf("Burst frame %d/%d %s\n", shot + 1, burstCount,
                  delivered ? "uploaded" : "buffered to SD");
  }

  ++consecutiveTriggers;
  if (!cfg.keepReferenceAfterTrigger || consecutiveTriggers >= cfg.maxConsecutiveTriggers) {
    // Either the old behaviour was requested, or the scene has stayed different
    // for long enough that it is the new background (moved feeder, new light).
    if (cfg.keepReferenceAfterTrigger) {
      Serial.printf("%d consecutive triggers; relearning background\n", consecutiveTriggers);
    }
    resetMotionReference();
  } else if (getCameraRuntimeState() != CameraRuntimeState::Error) {
    setCameraRuntimeState(isMotionUploadCoolingDown() ? CameraRuntimeState::UploadCooldown
                                                      : CameraRuntimeState::MotionArmed);
  }
}

void armMotionDetection() {
  if (!isFirmwareMotionOwner()) {
    cfg.motionEnabled = false;
    selectCameraOperatingMode(CameraOperatingMode::Idle);
    resetMotionReference();
    clearCameraRuntimeError();
    setCameraRuntimeState(CameraRuntimeState::Idle);
    return;
  }
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
  const CameraRuntimeState state = getCameraRuntimeState();
  referenceFrameReady = false;
  referenceExposureProductValid = false;
  referenceExposureProduct = 0.0f;
  currentExposureProductValid = false;
  currentExposureProduct = 0.0f;
  lastExposureShift = 0.0f;
  lastGlobalRatio = 1.0f;
  lastValidBlocks = 0;
  lastClippedBlocks = 0;
  lastChangedBlocks = 0;
  lastChangedFraction = 0.0f;
  consecutiveTriggers = 0;
  restartExposureSettling();
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

void noteMotionUploadFailure() {
  lastUploadFailure = millis();
  uploadCoolingDown = true;
}

int getConsecutiveMotionTriggers() { return consecutiveTriggers; }

int getMotionExposureSettleCount() { return exposureSettleCount; }

bool isMotionExposureSettled() { return exposureSettled; }

bool isMotionExposureProductValid() { return currentExposureProductValid; }

float getMotionCurrentExposureProduct() { return currentExposureProduct; }

float getMotionReferenceExposureProduct() { return referenceExposureProduct; }

float getMotionLastExposureShift() { return lastExposureShift; }

float getMotionLastGlobalRatio() { return lastGlobalRatio; }

int getMotionLastValidBlocks() { return lastValidBlocks; }

int getMotionLastClippedBlocks() { return lastClippedBlocks; }

int getMotionLastChangedBlocks() { return lastChangedBlocks; }

const char *getMotionLastDecision() { return lastDecision; }

void getMotionRoiPixels(int &x, int &y, int &width, int &height) {
  effectiveMotionRoi(x, y, width, height);
}

bool isMotionUploadCoolingDown() {
  return uploadCoolingDown && millis() - lastUploadFailure < kUploadFailureCooldownMs;
}