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