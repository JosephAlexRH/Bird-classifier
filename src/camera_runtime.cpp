#include "camera_runtime.h"

namespace {

CameraOperatingMode selectedMode = CameraOperatingMode::Idle;
CameraRuntimeState runtimeState = CameraRuntimeState::Idle;
String lastError;

}  // namespace

void initializeCameraRuntime(bool motionEnabled) {
  selectedMode = motionEnabled ? CameraOperatingMode::Motion : CameraOperatingMode::Idle;
  runtimeState = motionEnabled ? CameraRuntimeState::MotionArming : CameraRuntimeState::Idle;
  lastError = "";
}

void selectCameraOperatingMode(CameraOperatingMode mode) {
  selectedMode = mode;
  if (runtimeState != CameraRuntimeState::Streaming) {
    runtimeState = mode == CameraOperatingMode::Motion ? CameraRuntimeState::MotionArming
                                                       : CameraRuntimeState::Idle;
  }
}

CameraOperatingMode getCameraOperatingMode() { return selectedMode; }

const char *cameraOperatingModeName(CameraOperatingMode mode) {
  switch (mode) {
    case CameraOperatingMode::Motion: return "motion";
    case CameraOperatingMode::Idle: return "idle";
  }
  return "idle";
}

void setCameraRuntimeState(CameraRuntimeState state) { runtimeState = state; }

CameraRuntimeState getCameraRuntimeState() { return runtimeState; }

const char *cameraRuntimeStateName(CameraRuntimeState state) {
  switch (state) {
    case CameraRuntimeState::Idle: return "idle";
    case CameraRuntimeState::MotionArming: return "motion_arming";
    case CameraRuntimeState::MotionArmed: return "motion_armed";
    case CameraRuntimeState::MotionTriggered: return "motion_triggered";
    case CameraRuntimeState::Capturing: return "capturing";
    case CameraRuntimeState::Uploading: return "uploading";
    case CameraRuntimeState::UploadCooldown: return "upload_cooldown";
    case CameraRuntimeState::Streaming: return "streaming";
    case CameraRuntimeState::Error: return "error";
  }
  return "error";
}

bool isCameraStreamActive() { return runtimeState == CameraRuntimeState::Streaming; }

void setCameraRuntimeError(const String &message) {
  lastError = message;
  runtimeState = CameraRuntimeState::Error;
}

void clearCameraRuntimeError() { lastError = ""; }

const String &getCameraRuntimeError() { return lastError; }