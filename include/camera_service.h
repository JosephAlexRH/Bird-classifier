#pragma once

#include <Arduino.h>
#include <esp_camera.h>

#include "app_config.h"
#include "device_telemetry.h"

struct CameraSettingsSnapshot {
  int framesize;
  int pixelFormat;
  int jpegQuality;
  int brightness;
  int contrast;
  int saturation;
  int sharpness;
  int denoise;
  int specialEffect;
  int whiteBalanceMode;
  int autoWhiteBalance;
  int autoWhiteBalanceGain;
  int autoExposure;
  int autoExposureDsp;
  int exposureLevel;
  int exposureValue;
  int autoGain;
  int gain;
  int gainCeiling;
  int badPixelCorrection;
  int whitePixelCorrection;
  int rawGamma;
  int lensCorrection;
  int horizontalMirror;
  int verticalFlip;
  int downsizeCrop;
  int colorBar;
  bool scale;
  bool binning;
};

struct Ov2640RegisterSnapshot {
  bool supported;
  bool exposureValid;
  bool exposureConsistent;
  uint8_t exposureReadAttempts;
  uint8_t aecHigh;
  uint8_t aecMiddle;
  uint8_t aecLow;
  uint32_t exposure;
  uint32_t aecDriverLimitLines;
  bool aeSaturatedValid;
  bool aeSaturated;
  bool gainValid;
  uint8_t gainRaw;
  float nominalGainMultiplier;
  bool exposureGainProductValid;
  float exposureGainProduct;
  bool dspRawControlsValid;
  uint8_t dspControlCc;
  uint8_t dspControlCd;
  uint8_t dspControlCe;
};

// A lightweight post-frame AEC/AGC observation for motion gating. The OV2640
// registers are not an atomic exposure-start snapshot, so callers compare this
// relative product only between frames captured in the same sensor mode.
struct ExposureGainSample {
  bool valid;
  uint32_t exposureLines;
  uint8_t gainRaw;
  float nominalGainMultiplier;
  float product;
};

struct CameraMetadataSnapshot {
  bool sensorAvailable;
  uint16_t sensorPid;
  uint8_t sensorVersion;
  uint8_t manufacturerHigh;
  uint8_t manufacturerLow;
  uint8_t sensorAddress;
  int xclkHz;
  CameraSettingsSnapshot driverCache;
  Ov2640RegisterSnapshot observed;
  DeviceTelemetrySnapshot device;

  int requestedFramesize;
  int requestedJpegQuality;
  bool requestedMotionEnabled;
  MotionOwner requestedMotionOwner;
  int requestedMotionThreshold;
  float requestedMotionSensitivity;
  int requestedMotionCheckMs;
  float requestedMotionRoiX;
  float requestedMotionRoiY;
  float requestedMotionRoiWidth;
  float requestedMotionRoiHeight;
  bool requestedSaveBirdsToSd;
  float requestedBirdConfidenceThreshold;

  bool hasFrame;
  size_t frameWidth;
  size_t frameHeight;
  size_t frameBytes;
  int framePixelFormat;
  int64_t frameTimestampUs;
  int64_t registerReadBootUs;
  uint64_t frameNumberSinceBoot;
  uint32_t settingsRevision;
  bool settingsAgeValid;
  uint64_t msSinceSettingsChange;

  int frameBufferCount;
  int grabMode;
  bool registerDumpValid;
  bool includeRegisterDumpPayload;
  String registerDumpSha256;
  uint32_t persistentSettingsRevision;
};

struct CapturedJpeg {
  camera_fb_t *frameBuffer;
  CameraMetadataSnapshot metadata;
};

bool initializeCamera();
void applyCameraSettings();
bool drainCameraBuffer();
bool discardQueuedCameraFrame();
bool prepareJpegCapture(bool drainQueuedFrame = false);
bool configureMotionCapture(bool &changed);
camera_fb_t *acquireCameraFrame();
bool captureJpeg(CapturedJpeg &capture, bool drainQueuedFrame = false);
bool isCompleteJpeg(const camera_fb_t *frameBuffer);
CameraMetadataSnapshot sampleCameraMetadata(const camera_fb_t *frameBuffer = nullptr);
ExposureGainSample sampleExposureGain();
String buildCameraMetadata(const CameraMetadataSnapshot &snapshot, const char *source);
void markRegisterDumpPending();
void acknowledgeRegisterDumpDelivery(const CameraMetadataSnapshot &snapshot);