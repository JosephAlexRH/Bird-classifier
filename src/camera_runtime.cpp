#include "camera_runtime.h"

#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>

namespace {

CameraOperatingMode selectedMode = CameraOperatingMode::Idle;
CameraRuntimeState runtimeState = CameraRuntimeState::Idle;
String lastError;
StaticSemaphore_t runtimeMutexStorage;
SemaphoreHandle_t runtimeMutex = nullptr;

bool lockRuntime() {
  if (runtimeMutex == nullptr) {
    runtimeMutex = xSemaphoreCreateRecursiveMutexStatic(&runtimeMutexStorage);
  }
  return runtimeMutex != nullptr &&
         xSemaphoreTakeRecursive(runtimeMutex, portMAX_DELAY) == pdTRUE;
}

void unlockRuntime() {
  if (runtimeMutex != nullptr) xSemaphoreGiveRecursive(runtimeMutex);
}

class RuntimeMutexGuard {
 public:
  RuntimeMutexGuard() : locked_(lockRuntime()) {}
  ~RuntimeMutexGuard() {
    if (locked_) unlockRuntime();
  }
  bool locked() const { return locked_; }

 private:
  bool locked_;
};

}  // namespace

void initializeCameraRuntime(bool motionEnabled) {
  RuntimeMutexGuard runtimeLock;
  if (!runtimeLock.locked()) return;
  selectedMode = motionEnabled ? CameraOperatingMode::Motion : CameraOperatingMode::Idle;
  runtimeState = motionEnabled ? CameraRuntimeState::MotionArming : CameraRuntimeState::Idle;
  lastError = "";
}

void selectCameraOperatingMode(CameraOperatingMode mode) {
  RuntimeMutexGuard runtimeLock;
  if (!runtimeLock.locked()) return;
  selectedMode = mode;
  if (runtimeState != CameraRuntimeState::Streaming) {
    runtimeState = mode == CameraOperatingMode::Motion ? CameraRuntimeState::MotionArming
                                                       : CameraRuntimeState::Idle;
  }
}

CameraOperatingMode getCameraOperatingMode() {
  RuntimeMutexGuard runtimeLock;
  return runtimeLock.locked() ? selectedMode : CameraOperatingMode::Idle;
}

const char *cameraOperatingModeName(CameraOperatingMode mode) {
  switch (mode) {
    case CameraOperatingMode::Motion: return "motion";
    case CameraOperatingMode::Idle: return "idle";
  }
  return "idle";
}

void setCameraRuntimeState(CameraRuntimeState state) {
  RuntimeMutexGuard runtimeLock;
  if (runtimeLock.locked()) runtimeState = state;
}

CameraRuntimeState getCameraRuntimeState() {
  RuntimeMutexGuard runtimeLock;
  return runtimeLock.locked() ? runtimeState : CameraRuntimeState::Error;
}

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

bool isCameraStreamActive() {
  RuntimeMutexGuard runtimeLock;
  return runtimeLock.locked() && runtimeState == CameraRuntimeState::Streaming;
}

void setCameraRuntimeError(const String &message) {
  RuntimeMutexGuard runtimeLock;
  if (!runtimeLock.locked()) return;
  lastError = message;
  runtimeState = CameraRuntimeState::Error;
}

void clearCameraRuntimeError() {
  RuntimeMutexGuard runtimeLock;
  if (runtimeLock.locked()) lastError = "";
}

String getCameraRuntimeError() {
  RuntimeMutexGuard runtimeLock;
  return runtimeLock.locked() ? lastError : String("runtime mutex unavailable");
}