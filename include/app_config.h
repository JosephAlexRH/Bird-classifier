#pragma once

#include <Arduino.h>

enum class MotionOwner {
  Pi,
  Firmware,
};

struct Config {
  char wifiSsid[33];
  char wifiPassword[65];
  int motionThreshold;
  float motionSensitivity;
  int motionCheckMs;
  bool motionEnabled;
  MotionOwner motionOwner;
  int jpegQuality;
  int framesize;
  char serverIp[16];
  int serverPort;
  bool saveBirdsToSd;
  float birdConfidenceThreshold;
  bool waitForClassification;
  int burstCount;
  int burstIntervalMs;
  bool keepReferenceAfterTrigger;
  int maxConsecutiveTriggers;
  bool bufferUploadsToSd;
  float motionRoiX;
  float motionRoiY;
  float motionRoiWidth;
  float motionRoiHeight;
  int motionExposureSettleFrames;
  float motionExposureSettleTolerance;
  float motionExposureShiftTolerance;
  int motionBlockColumns;
  int motionBlockRows;
  float motionRatioThreshold;
  int motionClipLow;
  int motionClipHigh;
  float motionMaxClippedFraction;
  int motionMinValidBlocks;
  float motionMaxChangedFraction;
};

constexpr int kMaxBurstCount = 10;
constexpr int kMaxBurstIntervalMs = 5000;
constexpr int kMaxConsecutiveTriggersLimit = 100;
constexpr int kMaxMotionBlocks = 64;
constexpr int kMaxMotionSettleFrames = 20;

extern Config cfg;
extern bool sdReady;

const char *motionOwnerName(MotionOwner owner);
bool isFirmwareMotionOwner();

void mountStorage();
bool isConfigStorageReady();
const char* getConfigStorageName();
bool isConfigFileHashValid();
const String& getConfigFileSha256();
void loadConfig();
void saveConfig();