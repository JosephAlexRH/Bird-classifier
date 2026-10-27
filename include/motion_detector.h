#pragma once

#include <Arduino.h>

void checkMotion();
void armMotionDetection();
void disarmMotionDetection();
void resetMotionReference();
bool consumeMotionFlag();
bool isMotionFlagSet();
bool isReferenceFrameReady();
uint32_t getMotionEventCount();
unsigned long getLastMotionMs();
float getLastChangedFraction();
bool wasLastMotionCaptureSuccessful();
bool wasLastMotionUploadSuccessful();
bool isMotionCaptureBackoffActive();
uint32_t getMotionCaptureFailureCount();
uint32_t getConsecutiveMotionCaptureFailures();
uint32_t getMotionCameraRecoveryAttempts();
uint32_t getMotionCameraRecoverySuccesses();
unsigned long getLastMotionCaptureDurationMs();
unsigned long getLastSuccessfulMotionCaptureMs();
const char *getLastMotionCaptureOperation();
bool isMotionUploadCoolingDown();
void noteMotionUploadFailure();
int getConsecutiveMotionTriggers();
int getMotionExposureSettleCount();
bool isMotionExposureSettled();
bool isMotionExposureProductValid();
float getMotionCurrentExposureProduct();
float getMotionReferenceExposureProduct();
float getMotionLastExposureShift();
float getMotionLastGlobalRatio();
int getMotionLastValidBlocks();
int getMotionLastClippedBlocks();
int getMotionLastChangedBlocks();
const char *getMotionLastDecision();
void getMotionRoiPixels(int &x, int &y, int &width, int &height);