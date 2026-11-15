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

// Hub-supplied time, for sites with no internet. Call with the
// server_time_unix_ms value from a heartbeat reply. Does nothing once internet
// time (SNTP) has synchronised; otherwise sets the clock, and afterwards only
// corrects it when it has drifted from the hub by more than two seconds.
void applyHubTime(int64_t hubUnixMs);

// "ntp", "hub" or "none": where the device clock currently comes from.
const char *deviceTimeSource();