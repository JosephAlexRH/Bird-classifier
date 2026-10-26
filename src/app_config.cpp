#include "app_config.h"

#include <Arduino.h>
#include <ArduinoJson.h>
#include <FS.h>
#include <SD_MMC.h>
#include <SPIFFS.h>
#include <esp_camera.h>
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
uint16_t activeCameraSensorKey = 0;

bool loadSensorProfile(JsonVariantConst value, CameraSensorProfile &profile, uint16_t sensorKey) {
  const char *key = cameraSensorKeyName(sensorKey);
  if (!value.is<JsonObjectConst>()) {
    Serial.printf("Ignoring sensors.%s because it is not an object\n", key);
    return false;
  }

  JsonObjectConst object = value.as<JsonObjectConst>();
  CameraSensorProfile candidate = profile;
  if (object.containsKey("frame_size")) {
    JsonVariantConst frameSize = object["frame_size"];
    if (frameSize.is<const char *>()) {
      if (!parseConfigFramesize(frameSize.as<const char *>(), candidate.framesize)) {
        Serial.printf("Ignoring invalid sensors.%s.frame_size\n", key);
        return false;
      }
    } else if (frameSize.is<int>()) {
      candidate.framesize = frameSize.as<int>();
    } else {
      Serial.printf("Ignoring invalid sensors.%s.frame_size\n", key);
      return false;
    }
  }
  if (candidate.framesize < static_cast<int>(FRAMESIZE_QQVGA) ||
      candidate.framesize > cameraSensorMaximumFramesize(sensorKey)) {
    Serial.printf("Ignoring unsupported sensors.%s.frame_size\n", key);
    return false;
  }
  if (object.containsKey("jpeg_quality")) candidate.jpegQuality = object["jpeg_quality"];
  if (candidate.jpegQuality < 0 || candidate.jpegQuality > 63) {
    Serial.printf("Ignoring invalid sensors.%s.jpeg_quality\n", key);
    return false;
  }
  if (object.containsKey("xclk_hz")) candidate.xclkHz = object["xclk_hz"];
  if (candidate.xclkHz < kMinimumCameraXclkHz ||
      candidate.xclkHz > cameraSensorMaximumXclkHz(sensorKey)) {
    Serial.printf("Ignoring invalid sensors.%s.xclk_hz\n", key);
    return false;
  }
  profile = candidate;
  return true;
}

void writeSensorProfile(JsonObject object, const CameraSensorProfile &profile) {
  object["frame_size"] = configFramesizeName(profile.framesize);
  object["jpeg_quality"] = profile.jpegQuality;
  object["xclk_hz"] = profile.xclkHz;
}

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
    false,
    MotionOwner::Pi,
    {static_cast<int>(FRAMESIZE_VGA), 12, 20000000},
    {static_cast<int>(FRAMESIZE_VGA), 10, 20000000},
    10,
    static_cast<int>(FRAMESIZE_VGA),
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

const char *motionOwnerName(MotionOwner owner) {
  return owner == MotionOwner::Firmware ? "firmware" : "pi";
}

bool isFirmwareMotionOwner() { return cfg.motionOwner == MotionOwner::Firmware; }

const CameraSensorProfile *getCameraSensorProfile(uint16_t sensorKey) {
  if (sensorKey == kOv2640SensorKey) return &cfg.ov2640;
  if (sensorKey == kOv5640SensorKey) return &cfg.ov5640;
  return nullptr;
}

CameraSensorProfile *getMutableCameraSensorProfile(uint16_t sensorKey) {
  if (sensorKey == kOv2640SensorKey) return &cfg.ov2640;
  if (sensorKey == kOv5640SensorKey) return &cfg.ov5640;
  return nullptr;
}

bool selectCameraSensorProfile(uint16_t sensorKey) {
  const CameraSensorProfile *profile = getCameraSensorProfile(sensorKey);
  if (profile == nullptr) return false;
  activeCameraSensorKey = sensorKey;
  cfg.framesize = profile->framesize;
  cfg.jpegQuality = profile->jpegQuality;
  return true;
}

uint16_t getActiveCameraSensorKey() { return activeCameraSensorKey; }

const char *cameraSensorKeyName(uint16_t sensorKey) {
  if (sensorKey == kOv2640SensorKey) return "0x2642";
  if (sensorKey == kOv5640SensorKey) return "0x5640";
  return "unknown";
}

int cameraSensorMaximumFramesize(uint16_t sensorKey) {
  if (sensorKey == kOv2640SensorKey) return static_cast<int>(FRAMESIZE_UXGA);
  if (sensorKey == kOv5640SensorKey) return static_cast<int>(FRAMESIZE_QSXGA);
  return -1;
}

int cameraSensorMaximumXclkHz(uint16_t sensorKey) {
  if (sensorKey == kOv2640SensorKey) return kOv2640MaximumXclkHz;
  if (sensorKey == kOv5640SensorKey) return kOv5640MaximumXclkHz;
  return -1;
}

const char *configFramesizeName(int framesize) {
  switch (static_cast<framesize_t>(framesize)) {
    case FRAMESIZE_96X96: return "96X96";
    case FRAMESIZE_QQVGA: return "QQVGA";
    case FRAMESIZE_QCIF: return "QCIF";
    case FRAMESIZE_HQVGA: return "HQVGA";
    case FRAMESIZE_240X240: return "240X240";
    case FRAMESIZE_QVGA: return "QVGA";
    case FRAMESIZE_CIF: return "CIF";
    case FRAMESIZE_HVGA: return "HVGA";
    case FRAMESIZE_VGA: return "VGA";
    case FRAMESIZE_SVGA: return "SVGA";
    case FRAMESIZE_XGA: return "XGA";
    case FRAMESIZE_HD: return "HD";
    case FRAMESIZE_SXGA: return "SXGA";
    case FRAMESIZE_UXGA: return "UXGA";
    case FRAMESIZE_FHD: return "FHD";
    case FRAMESIZE_P_HD: return "P_HD";
    case FRAMESIZE_P_3MP: return "P_3MP";
    case FRAMESIZE_QXGA: return "QXGA";
    case FRAMESIZE_QHD: return "QHD";
    case FRAMESIZE_WQXGA: return "WQXGA";
    case FRAMESIZE_P_FHD: return "P_FHD";
    case FRAMESIZE_QSXGA: return "QSXGA";
    default: return "UNKNOWN";
  }
}

bool parseConfigFramesize(const char *name, int &framesize) {
  if (name == nullptr) return false;
  for (int value = static_cast<int>(FRAMESIZE_96X96);
       value <= static_cast<int>(FRAMESIZE_QSXGA); ++value) {
    if (strcasecmp(name, configFramesizeName(value)) == 0) {
      framesize = value;
      return true;
    }
  }
  return false;
}

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

void invalidateConfigFileHash() {
  configFileHashValid = false;
  configFileSha256 = "";
}

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

  StaticJsonDocument<3072> document;
  const DeserializationError error = deserializeJson(document, file);
  file.close();

  if (error) {
    Serial.printf("Config parse error: %s — keeping current values\n", error.c_str());
    if (cfg.wifiSsid[0] == '\0') applyCompiledWifiFallback();
    return;
  }

  hashConfigFile();

  if (document.containsKey("motion_owner")) {
    const char *owner = document["motion_owner"] | "";
    if (strcmp(owner, "firmware") == 0) {
      cfg.motionOwner = MotionOwner::Firmware;
    } else if (strcmp(owner, "pi") == 0) {
      cfg.motionOwner = MotionOwner::Pi;
    } else {
      cfg.motionOwner = MotionOwner::Pi;
      Serial.println("Invalid motion_owner in config.json; defaulting to pi");
    }
  } else {
    cfg.motionOwner = MotionOwner::Pi;
  }

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
  if (!isFirmwareMotionOwner() && cfg.motionEnabled) {
    cfg.motionEnabled = false;
    Serial.println("Ignoring motion_enabled=true because motion_owner is pi");
  }
  JsonObjectConst sensors = document["sensors"].as<JsonObjectConst>();
  if (!sensors.isNull()) {
    if (sensors.containsKey("0x2642")) {
      loadSensorProfile(sensors["0x2642"], cfg.ov2640, kOv2640SensorKey);
    }
    if (sensors.containsKey("0x5640")) {
      loadSensorProfile(sensors["0x5640"], cfg.ov5640, kOv5640SensorKey);
    }
  }
  if (sensors.isNull() && document.containsKey("jpeg_quality")) {
    const int legacyQuality = document["jpeg_quality"];
    if (legacyQuality >= 0 && legacyQuality <= 63) {
      cfg.ov2640.jpegQuality = legacyQuality;
      cfg.ov5640.jpegQuality = legacyQuality;
    } else {
      Serial.println("Ignoring invalid legacy jpeg_quality");
    }
  }
  if (sensors.isNull() && document.containsKey("framesize")) {
    const int legacyFramesize = document["framesize"];
    if (legacyFramesize >= static_cast<int>(FRAMESIZE_QQVGA) &&
        legacyFramesize <= cameraSensorMaximumFramesize(kOv2640SensorKey)) {
      cfg.ov2640.framesize = legacyFramesize;
      cfg.ov5640.framesize = legacyFramesize;
    } else {
      Serial.println("Ignoring invalid legacy framesize");
    }
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
  JsonObjectConst pipeline = document["pipeline"].as<JsonObjectConst>();
  if (!pipeline.isNull() && pipeline.containsKey("bird_confidence_threshold")) {
    cfg.birdConfidenceThreshold = pipeline["bird_confidence_threshold"];
  } else if (document.containsKey("bird_confidence_threshold")) {
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
  if (activeCameraSensorKey != 0) selectCameraSensorProfile(activeCameraSensorKey);

  Serial.printf("Config loaded from %s:\n", configStorageName);
  Serial.printf("  wifi_ssid:          %s\n", cfg.wifiSsid[0] == '\0' ? "(not configured)" : cfg.wifiSsid);
  Serial.printf("  wifi_password:      %s\n", cfg.wifiPassword[0] == '\0' ? "(not set)" : "(set)");
  Serial.printf("  motion_enabled:     %s\n", cfg.motionEnabled ? "true" : "false");
  Serial.printf("  motion_owner:       %s\n", motionOwnerName(cfg.motionOwner));
  Serial.printf("  motion_threshold:   %d\n", cfg.motionThreshold);
  Serial.printf("  motion_sensitivity: %.2f\n", cfg.motionSensitivity);
  Serial.printf("  motion_check_ms:    %d\n", cfg.motionCheckMs);
  Serial.printf("  OV2640 profile:      %s, quality %d, xclk %d Hz\n",
                configFramesizeName(cfg.ov2640.framesize), cfg.ov2640.jpegQuality,
                cfg.ov2640.xclkHz);
  Serial.printf("  OV5640 profile:      %s, quality %d, xclk %d Hz\n",
                configFramesizeName(cfg.ov5640.framesize), cfg.ov5640.jpegQuality,
                cfg.ov5640.xclkHz);
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

  StaticJsonDocument<3072> document;
  document["wifi_ssid"] = cfg.wifiSsid;
  document["wifi_password"] = cfg.wifiPassword;
  document["motion_threshold"] = cfg.motionThreshold;
  document["motion_sensitivity"] = cfg.motionSensitivity;
  document["motion_check_ms"] = cfg.motionCheckMs;
  document["motion_enabled"] = cfg.motionEnabled;
  document["motion_owner"] = motionOwnerName(cfg.motionOwner);
  JsonObject sensors = document["sensors"].to<JsonObject>();
  writeSensorProfile(sensors["0x2642"].to<JsonObject>(), cfg.ov2640);
  writeSensorProfile(sensors["0x5640"].to<JsonObject>(), cfg.ov5640);
  document["server_ip"] = cfg.serverIp;
  document["server_port"] = cfg.serverPort;
  document["save_birds_to_sd"] = cfg.saveBirdsToSd;
  JsonObject pipeline = document["pipeline"].to<JsonObject>();
  pipeline["bird_confidence_threshold"] = cfg.birdConfidenceThreshold;
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