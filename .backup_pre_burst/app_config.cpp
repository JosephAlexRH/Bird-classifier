#include "app_config.h"

#include <Arduino.h>
#include <ArduinoJson.h>
#include <FS.h>
#include <SD_MMC.h>
#include <SPIFFS.h>
#include <mbedtls/sha256.h>

namespace {

constexpr char kConfigPath[] = "/config.json";

fs::FS *configStorage = nullptr;
const char *configStorageName = "none";
String configFileSha256;
bool configFileHashValid = false;

String bytesToHex(const uint8_t *bytes, size_t length) {
  static const char digits[] = "0123456789abcdef";
  String result;
  result.reserve(length * 2);
  for (size_t index = 0; index < length; ++index) {
    result += digits[bytes[index] >> 4];
    result += digits[bytes[index] & 0x0f];
  }
  return result;
}

bool hashConfigFile() {
  configFileHashValid = false;
  configFileSha256 = "";
  if (configStorage == nullptr) return false;

  File file = configStorage->open(kConfigPath, FILE_READ);
  if (!file) return false;
  mbedtls_sha256_context context;
  mbedtls_sha256_init(&context);
  mbedtls_sha256_starts_ret(&context, 0);
  uint8_t buffer[256];
  while (file.available()) {
    const size_t length = file.read(buffer, sizeof(buffer));
    if (length == 0) break;
    mbedtls_sha256_update_ret(&context, buffer, length);
  }
  file.close();
  uint8_t digest[32];
  const int result = mbedtls_sha256_finish_ret(&context, digest);
  mbedtls_sha256_free(&context);
  if (result != 0) return false;
  configFileSha256 = bytesToHex(digest, sizeof(digest));
  configFileHashValid = true;
  return true;
}

}  // namespace

Config cfg = {
    20,
    0.05f,
    500,
    true,
    10,
    8,  // FRAMESIZE_VGA
    "192.168.7.170",
    5000,
    true,
    0.80f,
};

bool sdReady = false;

void mountStorage() {
  // One-bit mode avoids conflicts with the AI Thinker ESP32-CAM pins.
  if (SD_MMC.begin("/sdcard", true) && SD_MMC.cardType() != CARD_NONE) {
    sdReady = true;
    configStorage = &SD_MMC;
    configStorageName = "SD card";
    Serial.println("SD mounted; configuration storage: SD card");
    return;
  }

  SD_MMC.end();
  Serial.println("No usable SD card; trying internal SPIFFS configuration");

  if (SPIFFS.begin(true)) {
    configStorage = &SPIFFS;
    configStorageName = "SPIFFS";
    Serial.println("Configuration storage: internal SPIFFS");
    return;
  }

  Serial.println("SPIFFS mount failed — using compiled default config");
}

bool isConfigStorageReady() {
  return configStorage != nullptr;
}

const char *getConfigStorageName() {
  return configStorageName;
}

bool isConfigFileHashValid() { return configFileHashValid; }

const String &getConfigFileSha256() { return configFileSha256; }

void loadConfig() {
  configFileHashValid = false;
  configFileSha256 = "";
  if (!isConfigStorageReady()) {
    return;
  }

  File file = configStorage->open(kConfigPath, FILE_READ);
  if (!file) {
    Serial.printf("%s not found in %s; using current values\n", kConfigPath, configStorageName);
    return;
  }

  StaticJsonDocument<512> document;
  const DeserializationError error = deserializeJson(document, file);
  file.close();

  if (error) {
    Serial.printf("Config parse error: %s — keeping current values\n", error.c_str());
    return;
  }

  hashConfigFile();

  if (document.containsKey("motion_threshold")) {
    cfg.motionThreshold = document["motion_threshold"];
  }
  if (document.containsKey("motion_sensitivity")) {
    cfg.motionSensitivity = document["motion_sensitivity"];
  }
  if (document.containsKey("motion_check_ms")) {
    cfg.motionCheckMs = document["motion_check_ms"];
  }
  if (document.containsKey("motion_enabled")) {
    cfg.motionEnabled = document["motion_enabled"];
  }
  if (document.containsKey("jpeg_quality")) {
    cfg.jpegQuality = document["jpeg_quality"];
  }
  if (document.containsKey("framesize")) {
    cfg.framesize = document["framesize"];
  }
  if (document.containsKey("server_ip")) {
    strlcpy(cfg.serverIp, document["server_ip"] | cfg.serverIp, sizeof(cfg.serverIp));
  }
  if (document.containsKey("server_port")) {
    cfg.serverPort = document["server_port"];
  }
  if (document.containsKey("save_birds_to_sd")) {
    cfg.saveBirdsToSd = document["save_birds_to_sd"];
  }
  if (document.containsKey("bird_confidence_threshold")) {
    cfg.birdConfidenceThreshold = document["bird_confidence_threshold"];
  }

  Serial.printf("Config loaded from %s:\n", configStorageName);
  Serial.printf("  motion_enabled:     %s\n", cfg.motionEnabled ? "true" : "false");
  Serial.printf("  motion_threshold:   %d\n", cfg.motionThreshold);
  Serial.printf("  motion_sensitivity: %.2f\n", cfg.motionSensitivity);
  Serial.printf("  motion_check_ms:    %d\n", cfg.motionCheckMs);
  Serial.printf("  jpeg_quality:       %d\n", cfg.jpegQuality);
  Serial.printf("  framesize:          %d\n", cfg.framesize);
  Serial.printf("  server_ip:          %s\n", cfg.serverIp);
  Serial.printf("  server_port:        %d\n", cfg.serverPort);
  Serial.printf("  save_birds_to_sd:   %s\n", cfg.saveBirdsToSd ? "true" : "false");
  Serial.printf("  bird_confidence:    %.2f\n", cfg.birdConfidenceThreshold);
}

void saveConfig() {
  configFileHashValid = false;
  configFileSha256 = "";
  if (!isConfigStorageReady()) {
    return;
  }

  StaticJsonDocument<512> document;
  document["motion_threshold"] = cfg.motionThreshold;
  document["motion_sensitivity"] = cfg.motionSensitivity;
  document["motion_check_ms"] = cfg.motionCheckMs;
  document["motion_enabled"] = cfg.motionEnabled;
  document["jpeg_quality"] = cfg.jpegQuality;
  document["framesize"] = cfg.framesize;
  document["server_ip"] = cfg.serverIp;
  document["server_port"] = cfg.serverPort;
  document["save_birds_to_sd"] = cfg.saveBirdsToSd;
  document["bird_confidence_threshold"] = cfg.birdConfidenceThreshold;

  File file = configStorage->open(kConfigPath, FILE_WRITE);
  if (!file) {
    Serial.printf("Failed to open %s in %s for writing\n", kConfigPath, configStorageName);
    return;
  }

  serializeJsonPretty(document, file);
  file.close();
  hashConfigFile();
  Serial.printf("Config saved to %s\n", configStorageName);
}