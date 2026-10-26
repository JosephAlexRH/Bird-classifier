#pragma once

#include <Arduino.h>

enum class MotionOwner {
  Pi,
  Firmware,
};

constexpr uint16_t kOv2640SensorKey = 0x2642;
constexpr uint16_t kOv5640SensorKey = 0x5640;
constexpr int kMinimumCameraXclkHz = 6000000;
constexpr int kOv2640MaximumXclkHz = 24000000;
constexpr int kOv5640MaximumXclkHz = 27000000;

struct CameraSensorProfile {
  int framesize;
  int jpegQuality;
  int xclkHz;
};

struct Config {
  char wifiSsid[33];
  char wifiPassword[65];
  int motionThreshold;
  float motionSensitivity;
  int motionCheckMs;
  bool motionEnabled;
  MotionOwner motionOwner;
  CameraSensorProfile ov2640;
  CameraSensorProfile ov5640;
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
const CameraSensorProfile *getCameraSensorProfile(uint16_t sensorKey);
CameraSensorProfile *getMutableCameraSensorProfile(uint16_t sensorKey);
bool selectCameraSensorProfile(uint16_t sensorKey);
uint16_t getActiveCameraSensorKey();
const char *cameraSensorKeyName(uint16_t sensorKey);
int cameraSensorMaximumFramesize(uint16_t sensorKey);
int cameraSensorMaximumXclkHz(uint16_t sensorKey);
const char *configFramesizeName(int framesize);
bool parseConfigFramesize(const char *name, int &framesize);

void mountStorage();
bool isConfigStorageReady();
const char* getConfigStorageName();
bool isConfigFileHashValid();
const String& getConfigFileSha256();
void invalidateConfigFileHash();
void loadConfig();
void saveConfig();