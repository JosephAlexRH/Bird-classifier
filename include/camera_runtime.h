#pragma once

#include <Arduino.h>

enum class CameraOperatingMode {
  Idle,
  Motion,
};

enum class CameraRuntimeState {
  Idle,
  MotionArming,
  MotionArmed,
  MotionTriggered,
  Capturing,
  Uploading,
  UploadCooldown,
  Streaming,
  Error,
};

void initializeCameraRuntime(bool motionEnabled);
void selectCameraOperatingMode(CameraOperatingMode mode);
CameraOperatingMode getCameraOperatingMode();
const char *cameraOperatingModeName(CameraOperatingMode mode);

void setCameraRuntimeState(CameraRuntimeState state);
CameraRuntimeState getCameraRuntimeState();
const char *cameraRuntimeStateName(CameraRuntimeState state);
bool isCameraStreamActive();

void setCameraRuntimeError(const String &message);
void clearCameraRuntimeError();
const String &getCameraRuntimeError();