#pragma once

#include <Arduino.h>

struct Config {
  int motionThreshold;
  float motionSensitivity;
  int motionCheckMs;
  bool motionEnabled;
  int jpegQuality;
  int framesize;
  char serverIp[16];
  int serverPort;
  bool saveBirdsToSd;
  float birdConfidenceThreshold;
};

extern Config cfg;
extern bool sdReady;

void mountStorage();
bool isConfigStorageReady();
const char* getConfigStorageName();
bool isConfigFileHashValid();
const String& getConfigFileSha256();
void loadConfig();
void saveConfig();