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