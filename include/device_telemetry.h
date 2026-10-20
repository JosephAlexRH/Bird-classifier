#pragma once

#include <Arduino.h>

struct DeviceTelemetrySnapshot {
  String cameraId;
  String bootId;
  bool timeSynchronized;
  String deviceTimeUtc;
  int64_t sampledAtBootUs;
  float chipTemperatureC;
  uint32_t freeHeapBytes;
  uint32_t minimumFreeHeapBytes;
  uint32_t largestFreeHeapBlockBytes;
  uint32_t psramTotalBytes;
  uint32_t psramFreeBytes;
  uint32_t psramMinimumFreeBytes;
  uint32_t psramLargestFreeBlockBytes;
};

void initializeDeviceTelemetry();
DeviceTelemetrySnapshot sampleDeviceTelemetry();
const String &getCameraId();
const String &getBootId();