#include "app_config.h"

#include <Arduino.h>
#include <ArduinoJson.h>
#include <FS.h>
#include <SD_MMC.h>
#include <SPIFFS.h>
#include <mbedtls/sha256.h>

#if __has_include("secrets.h")
#include "secrets.h"
#define HAS_COMPILED_WIFI_SECRETS 1
#else
#define HAS_COMPILED_WIFI_SECRETS 0
#endif

namespace {

constexpr char kConfigPath[] = "/config.json";

fs::FS *configStorage = nullptr;
const char *configStorageName = "none";
String configFileSha256;
bool configFileHashValid = false;

void applyCompiledWifiFallback() {
  cfg.wifiSsid[0] = '\0';
  cfg.wifiPassword[0] = '\0';
#if HAS_COMPILED_WIFI_SECRETS
  if (WIFI_SSID[0] != '\0') {
    strlcpy(cfg.wifiSsid, WIFI_SSID, sizeof(cfg.wifiSsid));
    strlcpy(cfg.wifiPassword, WIFI_PASSWORD, sizeof(cfg.wifiPassword));
  }
#endif
}

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

void sanitizeMotionConfig() {
  if (!isfinite(cfg.motionRoiX) || !isfinite(cfg.motionRoiY) ||
      !isfinite(cfg.motionRoiWidth) || !isfinite(cfg.motionRoiHeight) ||
      cfg.motionRoiX < 0.0f || cfg.motionRoiY < 0.0f || cfg.motionRoiWidth <= 0.0f ||
      cfg.motionRoiHeight <= 0.0f || cfg.motionRoiX + cfg.motionRoiWidth > 1.00001f ||
      cfg.motionRoiY + cfg.motionRoiHeight > 1.00001f ||
      ceilf(cfg.motionRoiWidth * 80.0f) < 16.0f ||
      ceilf(cfg.motionRoiHeight * 60.0f) < 12.0f) {
    cfg.motionRoiX = 0.0f;
    cfg.motionRoiY = 0.0f;
    cfg.motionRoiWidth = 1.0f;
    cfg.motionRoiHeight = 1.0f;
  }
  cfg.motionExposureSettleFrames = constrain(cfg.motionExposureSettleFrames, 2,
                                              kMaxMotionSettleFrames);
  if (!isfinite(cfg.motionExposureSettleTolerance)) cfg.motionExposureSettleTolerance = 0.05f;
  cfg.motionExposureSettleTolerance = constrain(cfg.motionExposureSettleTolerance, 0.0f, 1.0f);
  if (!isfinite(cfg.motionExposureShiftTolerance)) cfg.motionExposureShiftTolerance = 0.10f;
  cfg.motionExposureShiftTolerance =
      constrain(cfg.motionExposureShiftTolerance, cfg.motionExposureSettleTolerance, 1.0f);
  cfg.motionBlockColumns = constrain(cfg.motionBlockColumns, 1, 16);
  cfg.motionBlockRows = constrain(cfg.motionBlockRows, 1, 16);
  if (cfg.motionBlockColumns * cfg.motionBlockRows > kMaxMotionBlocks) {
    cfg.motionBlockColumns = 8;
    cfg.motionBlockRows = 6;
  }
  if (!isfinite(cfg.motionRatioThreshold)) cfg.motionRatioThreshold = 0.15f;
  cfg.motionRatioThreshold = constrain(cfg.motionRatioThreshold, 0.0f, 4.0f);
  cfg.motionClipLow = constrain(cfg.motionClipLow, 0, 254);
  cfg.motionClipHigh = constrain(cfg.motionClipHigh, 1, 255);
  if (cfg.motionClipLow >= cfg.motionClipHigh) {
    cfg.motionClipLow = 8;
    cfg.motionClipHigh = 247;
  }
  if (!isfinite(cfg.motionMaxClippedFraction)) cfg.motionMaxClippedFraction = 0.25f;
  cfg.motionMaxClippedFraction = constrain(cfg.motionMaxClippedFraction, 0.0f, 1.0f);
  cfg.motionMinValidBlocks = constrain(cfg.motionMinValidBlocks, 1,
                                       cfg.motionBlockColumns * cfg.motionBlockRows);
  if (!isfinite(cfg.motionMaxChangedFraction) ||
      cfg.motionMaxChangedFraction <= cfg.motionSensitivity ||
      cfg.motionMaxChangedFraction > 1.0f) {
    cfg.motionMaxChangedFraction = cfg.motionSensitivity < 0.45f
                                       ? 0.45f
                                       : min(1.0f, cfg.motionSensitivity + 0.10f);
  }
}

}  // namespace

Config cfg = {
    "",    // loaded from config.json, or include/secrets.h when present
    "",    // never written to the serial log or returned by the config API
    20,
    0.05f,
    200,
    true,
    10,
    8,  // FRAMESIZE_VGA
    "192.168.7.170",
    5000,
    true,
    0.80f,
    true,  // wait for the receiver's classification before re-arming
    3,     // JPEGs uploaded per motion trigger
    150,   // pause between burst JPEGs, ms
    true,  // keep the pre-trigger background so a subject that stays keeps triggering
    10,    // consecutive triggers before the background is relearned anyway
    true,  // write undeliverable motion JPEGs to /pending on SD and upload later
    0.0f,  // normalized motion ROI x
    0.0f,  // normalized motion ROI y
    1.0f,  // normalized motion ROI width
    1.0f,  // normalized motion ROI height
    4,     // consecutive stable exposure/gain products required before comparison
    0.05f, // maximum product range while settling
    0.10f, // maximum product shift between a reference and comparison frame
    8,     // motion block columns
    6,     // motion block rows
    0.15f, // block ratio deviation from the global median
    8,     // low luminance clipping boundary
    247,   // high luminance clipping boundary
    0.25f, // maximum clipped-pixel fraction in either version of a block
    12,    // minimum valid blocks required for a decision
    0.45f, // broad-change upper bound (lighting/scene change rather than motion)
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
    if (cfg.wifiSsid[0] == '\0') applyCompiledWifiFallback();
    return;
  }

  File file = configStorage->open(kConfigPath, FILE_READ);
  if (!file) {
    Serial.printf("%s not found in %s; using current values\n", kConfigPath, configStorageName);
    if (cfg.wifiSsid[0] == '\0') applyCompiledWifiFallback();
    return;
  }

  StaticJsonDocument<2048> document;
  const DeserializationError error = deserializeJson(document, file);
  file.close();

  if (error) {
    Serial.printf("Config parse error: %s — keeping current values\n", error.c_str());
    if (cfg.wifiSsid[0] == '\0') applyCompiledWifiFallback();
    return;
  }

  hashConfigFile();

  const char *ssid = document["wifi_ssid"] | "";
  const char *password = document["wifi_password"] | "";
  if (ssid[0] != '\0' && strlen(ssid) <= 32 && strlen(password) <= 64) {
    strlcpy(cfg.wifiSsid, ssid, sizeof(cfg.wifiSsid));
    strlcpy(cfg.wifiPassword, password, sizeof(cfg.wifiPassword));
  } else {
    if (ssid[0] != '\0') {
      Serial.println("Invalid Wi-Fi credentials in config.json; using compiled fallback if available");
    }
    applyCompiledWifiFallback();
  }

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
  if (document.containsKey("wait_for_classification")) {
    cfg.waitForClassification = document["wait_for_classification"];
  }
  if (document.containsKey("burst_count")) {
    cfg.burstCount = constrain(document["burst_count"].as<int>(), 1, kMaxBurstCount);
  }
  if (document.containsKey("burst_interval_ms")) {
    cfg.burstIntervalMs = constrain(document["burst_interval_ms"].as<int>(), 0, kMaxBurstIntervalMs);
  }
  if (document.containsKey("keep_reference_after_trigger")) {
    cfg.keepReferenceAfterTrigger = document["keep_reference_after_trigger"];
  }
  if (document.containsKey("max_consecutive_triggers")) {
    cfg.maxConsecutiveTriggers =
        constrain(document["max_consecutive_triggers"].as<int>(), 1, kMaxConsecutiveTriggersLimit);
  }
  if (document.containsKey("buffer_uploads_to_sd")) {
    cfg.bufferUploadsToSd = document["buffer_uploads_to_sd"];
  }
  if (document["motion_roi"].is<JsonObject>()) {
    const JsonObject roi = document["motion_roi"];
    cfg.motionRoiX = roi["x"] | cfg.motionRoiX;
    cfg.motionRoiY = roi["y"] | cfg.motionRoiY;
    cfg.motionRoiWidth = roi["width"] | cfg.motionRoiWidth;
    cfg.motionRoiHeight = roi["height"] | cfg.motionRoiHeight;
  }
  if (document.containsKey("motion_exposure_settle_frames")) {
    cfg.motionExposureSettleFrames =
        constrain(document["motion_exposure_settle_frames"].as<int>(), 2, kMaxMotionSettleFrames);
  }
  if (document.containsKey("motion_exposure_settle_tolerance")) {
    cfg.motionExposureSettleTolerance =
        constrain(document["motion_exposure_settle_tolerance"].as<float>(), 0.0f, 1.0f);
  }
  if (document.containsKey("motion_exposure_shift_tolerance")) {
    cfg.motionExposureShiftTolerance =
        constrain(document["motion_exposure_shift_tolerance"].as<float>(), 0.0f, 1.0f);
  }
  if (document.containsKey("motion_block_columns")) {
    cfg.motionBlockColumns = constrain(document["motion_block_columns"].as<int>(), 1, 16);
  }
  if (document.containsKey("motion_block_rows")) {
    cfg.motionBlockRows = constrain(document["motion_block_rows"].as<int>(), 1, 16);
  }
  if (cfg.motionBlockColumns * cfg.motionBlockRows > kMaxMotionBlocks) {
    cfg.motionBlockColumns = 8;
    cfg.motionBlockRows = 6;
  }
  if (document.containsKey("motion_ratio_threshold")) {
    cfg.motionRatioThreshold = constrain(document["motion_ratio_threshold"].as<float>(), 0.0f, 4.0f);
  }
  if (document.containsKey("motion_clip_low")) {
    cfg.motionClipLow = constrain(document["motion_clip_low"].as<int>(), 0, 254);
  }
  if (document.containsKey("motion_clip_high")) {
    cfg.motionClipHigh = constrain(document["motion_clip_high"].as<int>(), 1, 255);
  }
  if (cfg.motionClipLow >= cfg.motionClipHigh) {
    cfg.motionClipLow = 8;
    cfg.motionClipHigh = 247;
  }
  if (document.containsKey("motion_max_clipped_fraction")) {
    cfg.motionMaxClippedFraction =
        constrain(document["motion_max_clipped_fraction"].as<float>(), 0.0f, 1.0f);
  }
  if (document.containsKey("motion_min_valid_blocks")) {
    cfg.motionMinValidBlocks =
        constrain(document["motion_min_valid_blocks"].as<int>(), 1, kMaxMotionBlocks);
  }
  if (document.containsKey("motion_max_changed_fraction")) {
    cfg.motionMaxChangedFraction =
        constrain(document["motion_max_changed_fraction"].as<float>(), 0.0f, 1.0f);
  }
  sanitizeMotionConfig();

  Serial.printf("Config loaded from %s:\n", configStorageName);
  Serial.printf("  wifi_ssid:          %s\n", cfg.wifiSsid[0] == '\0' ? "(not configured)" : cfg.wifiSsid);
  Serial.printf("  wifi_password:      %s\n", cfg.wifiPassword[0] == '\0' ? "(not set)" : "(set)");
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
  Serial.printf("  wait_for_classif.:  %s\n", cfg.waitForClassification ? "true" : "false");
  Serial.printf("  burst_count:        %d\n", cfg.burstCount);
  Serial.printf("  burst_interval_ms:  %d\n", cfg.burstIntervalMs);
  Serial.printf("  keep_reference:     %s\n", cfg.keepReferenceAfterTrigger ? "true" : "false");
  Serial.printf("  max_consec_trig.:   %d\n", cfg.maxConsecutiveTriggers);
  Serial.printf("  buffer_to_sd:       %s\n", cfg.bufferUploadsToSd ? "true" : "false");
  Serial.printf("  motion_roi:          %.3f,%.3f %.3fx%.3f\n", cfg.motionRoiX, cfg.motionRoiY,
                cfg.motionRoiWidth, cfg.motionRoiHeight);
  Serial.printf("  exposure_settle:     %d frames at %.3f\n", cfg.motionExposureSettleFrames,
                cfg.motionExposureSettleTolerance);
  Serial.printf("  exposure_shift:      %.3f\n", cfg.motionExposureShiftTolerance);
  Serial.printf("  motion_blocks:       %dx%d\n", cfg.motionBlockColumns, cfg.motionBlockRows);
  Serial.printf("  ratio_threshold:     %.3f\n", cfg.motionRatioThreshold);
  Serial.printf("  changed_bounds:      %.3f..%.3f\n", cfg.motionSensitivity,
                cfg.motionMaxChangedFraction);
}

void saveConfig() {
  configFileHashValid = false;
  configFileSha256 = "";
  if (!isConfigStorageReady()) {
    return;
  }

  StaticJsonDocument<2048> document;
  document["wifi_ssid"] = cfg.wifiSsid;
  document["wifi_password"] = cfg.wifiPassword;
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
  document["wait_for_classification"] = cfg.waitForClassification;
  document["burst_count"] = cfg.burstCount;
  document["burst_interval_ms"] = cfg.burstIntervalMs;
  document["keep_reference_after_trigger"] = cfg.keepReferenceAfterTrigger;
  document["max_consecutive_triggers"] = cfg.maxConsecutiveTriggers;
  document["buffer_uploads_to_sd"] = cfg.bufferUploadsToSd;
  JsonObject roi = document["motion_roi"].to<JsonObject>();
  roi["x"] = cfg.motionRoiX;
  roi["y"] = cfg.motionRoiY;
  roi["width"] = cfg.motionRoiWidth;
  roi["height"] = cfg.motionRoiHeight;
  document["motion_exposure_settle_frames"] = cfg.motionExposureSettleFrames;
  document["motion_exposure_settle_tolerance"] = cfg.motionExposureSettleTolerance;
  document["motion_exposure_shift_tolerance"] = cfg.motionExposureShiftTolerance;
  document["motion_block_columns"] = cfg.motionBlockColumns;
  document["motion_block_rows"] = cfg.motionBlockRows;
  document["motion_ratio_threshold"] = cfg.motionRatioThreshold;
  document["motion_clip_low"] = cfg.motionClipLow;
  document["motion_clip_high"] = cfg.motionClipHigh;
  document["motion_max_clipped_fraction"] = cfg.motionMaxClippedFraction;
  document["motion_min_valid_blocks"] = cfg.motionMinValidBlocks;
  document["motion_max_changed_fraction"] = cfg.motionMaxChangedFraction;

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