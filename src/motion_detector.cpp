#include "motion_detector.h"

#include <Arduino.h>
#include <ArduinoJson.h>
#include <HTTPClient.h>
#include <esp_camera.h>
#include <img_converters.h>
#include <esp_heap_caps.h>

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
constexpr unsigned long kCaptureFailureBackoffMs = 1500;
constexpr uint32_t kCameraRecoveryFailureThreshold = 3;

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
unsigned long lastCaptureFailureMs = 0;
unsigned long lastCaptureDurationMs = 0;
unsigned long lastSuccessfulCaptureMs = 0;
uint32_t captureFailureCount = 0;
uint32_t consecutiveCaptureFailures = 0;
uint32_t cameraRecoveryAttempts = 0;
uint32_t cameraRecoverySuccesses = 0;
const char *lastCaptureOperation = "not_started";
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

void noteCaptureFailure(const char *operation, const String &message, unsigned long startedMs) {
  lastMotionCaptureSuccessful = false;
  lastCaptureFailureMs = millis();
  lastCaptureDurationMs = lastCaptureFailureMs - startedMs;
  ++captureFailureCount;
  ++consecutiveCaptureFailures;
  lastCaptureOperation = operation;
  lastDecision = operation;
  setCameraRuntimeError(message);
  Serial.printf("Motion capture failure during %s after %lu ms (%lu consecutive): %s\n",
                operation, lastCaptureDurationMs,
                static_cast<unsigned long>(consecutiveCaptureFailures), message.c_str());
}

void noteCaptureSuccess(const char *operation, unsigned long startedMs) {
  const unsigned long now = millis();
  lastCaptureDurationMs = now - startedMs;
  lastSuccessfulCaptureMs = now;
  consecutiveCaptureFailures = 0;
  lastCaptureOperation = operation;
}

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

// Reads the width and height from the JPEG's own SOF header. jpg2rgb565 sizes
// its output from this header, not from frameBuffer->width/height.
bool readJpegDimensions(const uint8_t *data, size_t length, uint16_t &width, uint16_t &height) {
  if (length < 4 || data[0] != 0xff || data[1] != 0xd8) return false;
  size_t index = 2;
  while (index + 9 < length) {
    if (data[index] != 0xff) return false;
    const uint8_t marker = data[index + 1];
    if (marker == 0xff) {
      ++index;
      continue;
    }
    if (marker == 0x01 || (marker >= 0xd0 && marker <= 0xd8)) {
      index += 2;
      continue;
    }
    if (marker == 0xda || marker == 0xd9) return false;  // scan data before any SOF
    const size_t segmentLength = (static_cast<size_t>(data[index + 2]) << 8) | data[index + 3];
    if (segmentLength < 2) return false;
    if (marker >= 0xc0 && marker <= 0xcf && marker != 0xc4 && marker != 0xc8 &&
        marker != 0xcc) {
      height = (static_cast<uint16_t>(data[index + 5]) << 8) | data[index + 6];
      width = (static_cast<uint16_t>(data[index + 7]) << 8) | data[index + 8];
      return true;
    }
    index += 2 + segmentLength;
  }
  return false;
}

void checkInternalHeap(const char *where) {
  if (!heap_caps_check_integrity(MALLOC_CAP_INTERNAL, true)) {
    Serial.printf("HEAP CORRUPTION detected after %s\n", where);
  }
}

uint32_t rejectedMotionJpegCount = 0;

bool decodeMotionFrame(const camera_fb_t *frameBuffer) {
  // decodedMotionFrame holds exactly 80x60 RGB565. If the JPEG header is larger
  // than the size the driver labelled the buffer with (a stale frame after a
  // size change, or a corrupted header), jpg2rgb565 would write past the end of
  // this static array into neighbouring RAM. Refuse instead.
  uint16_t jpegWidth = 0;
  uint16_t jpegHeight = 0;
  if (!readJpegDimensions(frameBuffer->buf, frameBuffer->len, jpegWidth, jpegHeight) ||
      jpegWidth != frameBuffer->width || jpegHeight != frameBuffer->height) {
    ++rejectedMotionJpegCount;
    Serial.printf("Rejected motion JPEG #%lu: header %ux%u, buffer labelled %ux%u, %u bytes\n",
                  static_cast<unsigned long>(rejectedMotionJpegCount), jpegWidth, jpegHeight,
                  static_cast<unsigned int>(frameBuffer->width),
                  static_cast<unsigned int>(frameBuffer->height),
                  static_cast<unsigned int>(frameBuffer->len));
    return false;
  }
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
  if (isMotionCaptureBackoffActive()) return;

  const unsigned long analysisStartedMs = millis();
  if (consecutiveCaptureFailures >= kCameraRecoveryFailureThreshold) {
    ++cameraRecoveryAttempts;
    lastCaptureOperation = "camera_reinitialize";
    Serial.printf("Reinitializing camera after %lu consecutive capture failures\n",
                  static_cast<unsigned long>(consecutiveCaptureFailures));
    if (!reinitializeActiveCamera()) {
      noteCaptureFailure("camera_reinitialize", "Camera reinitialization failed",
                         analysisStartedMs);
      return;
    }

    ++cameraRecoverySuccesses;
    consecutiveCaptureFailures = 0;
    lastCaptureDurationMs = millis() - analysisStartedMs;
    lastCaptureOperation = "camera_reinitialized";
    resetMotionReference();
    clearCameraRuntimeError();
    setCameraRuntimeState(CameraRuntimeState::MotionArming);
    Serial.printf("Camera reinitialized in %lu ms; motion reference will be relearned\n",
                  lastCaptureDurationMs);
    return;
  }

  bool modeChanged = false;
  if (!configureMotionCapture(modeChanged)) {
    noteCaptureFailure("configure", "Could not configure motion capture mode", analysisStartedMs);
    return;
  }
  if (modeChanged) {
    // A buffer already in flight can still use the previous JPEG mode.
    if (!drainCameraBuffer()) {
      noteCaptureFailure("mode_change_drain", "Could not drain stale motion frame",
                         analysisStartedMs);
      return;
    }
    // A frame-size transition restarts AEC/AGC convergence. Keep an existing
    // background when requested, but never compare against a transition frame.
    restartExposureSettling();
  }

  // The queued frame was grabbed just after the previous check and is a whole
  // check interval old. A mode change has already drained it.
  if (!modeChanged && !cameraUsesLatestFrameGrab() && !discardQueuedCameraFrame()) {
    noteCaptureFailure("analysis_discard", "Could not discard stale motion frame",
                       analysisStartedMs);
    return;
  }

  camera_fb_t *frameBuffer = acquireCameraFrame();
  if (!isCompleteJpeg(frameBuffer) || !isSupportedMotionFrame(frameBuffer)) {
    if (frameBuffer != nullptr) {
      releaseCameraFrame(frameBuffer);
    }
    noteCaptureFailure("analysis_capture", "Motion JPEG frame capture failed", analysisStartedMs);
    return;
  }

  if (!decodeMotionFrame(frameBuffer)) {
    releaseCameraFrame(frameBuffer);
    noteCaptureFailure("analysis_decode", "Motion JPEG decode failed", analysisStartedMs);
    return;
  }
  // Read AEC/AGC while this exact framebuffer is retained. This is a post-frame,
  // non-atomic sensor observation, so several consecutive products must agree.
  const ExposureGainSample exposure = sampleExposureGain();
  releaseCameraFrame(frameBuffer);
  checkInternalHeap("motion decode");  // DIAGNOSTIC
  noteCaptureSuccess("analysis", analysisStartedMs);

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

  // The OV5640 uses a latest-frame queue. Copy each JPEG to owned PSRAM and
  // return the driver framebuffer before network or SD I/O so capture can continue.
  const int burstCount = constrain(cfg.burstCount, 1, kMaxBurstCount);
  for (int shot = 0; shot < burstCount; ++shot) {
    const unsigned long burstStartedMs = millis();
    if (shot > 0) {
      delay(constrain(cfg.burstIntervalMs, 0, kMaxBurstIntervalMs));
      // One-buffer OV2640 mode can contain the frame queued during upload.
      if (!cameraUsesLatestFrameGrab() && !discardQueuedCameraFrame()) {
        noteCaptureFailure("burst_discard", "Could not discard stale burst frame",
                           burstStartedMs);
        break;
      }
    }

    CapturedJpeg capture;
    setCameraRuntimeState(CameraRuntimeState::Capturing);
    if (!captureJpeg(capture)) {
      noteCaptureFailure("burst_capture", "Motion JPEG capture failed", burstStartedMs);
      break;
    }

    camera_fb_t ownedFrame = *capture.frameBuffer;
    ownedFrame.buf = static_cast<uint8_t *>(psramFound() ? ps_malloc(ownedFrame.len)
                                                         : malloc(ownedFrame.len));
    if (ownedFrame.buf == nullptr) {
      releaseCameraFrame(capture.frameBuffer);
      noteCaptureFailure("burst_copy", "Not enough memory to retain motion JPEG",
                         burstStartedMs);
      break;
    }
    memcpy(ownedFrame.buf, capture.frameBuffer->buf, ownedFrame.len);
    releaseCameraFrame(capture.frameBuffer);
    capture.frameBuffer = &ownedFrame;
    noteCaptureSuccess("burst_capture", burstStartedMs);
    lastMotionCaptureSuccessful = true;
    bool delivered = false;
    if (!offline) {
      delivered = postFrame(capture);
      checkInternalHeap("burst upload");  // DIAGNOSTIC
      if (!delivered) {
        noteMotionUploadFailure();
        offline = true;  // The rest of this burst goes straight to SD.
      }
    }
    const bool buffered =
        !delivered && bufferUpload(capture.frameBuffer, buildCameraMetadata(capture.metadata, "motion"));
    free(ownedFrame.buf);
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
  ConfigMutexGuard configLock;
  if (!configLock.locked()) return;
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
  ConfigMutexGuard configLock;
  if (!configLock.locked()) return;
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

bool isMotionCaptureBackoffActive() {
  return consecutiveCaptureFailures > 0 &&
         millis() - lastCaptureFailureMs < kCaptureFailureBackoffMs;
}

uint32_t getMotionCaptureFailureCount() { return captureFailureCount; }

uint32_t getConsecutiveMotionCaptureFailures() { return consecutiveCaptureFailures; }

uint32_t getMotionCameraRecoveryAttempts() { return cameraRecoveryAttempts; }

uint32_t getMotionCameraRecoverySuccesses() { return cameraRecoverySuccesses; }

unsigned long getLastMotionCaptureDurationMs() { return lastCaptureDurationMs; }

unsigned long getLastSuccessfulMotionCaptureMs() { return lastSuccessfulCaptureMs; }

const char *getLastMotionCaptureOperation() { return lastCaptureOperation; }

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