#include "web_server.h"

#include <Arduino.h>
#include <ArduinoJson.h>
#include <WiFi.h>
#include <esp_camera.h>

#include "app_config.h"
#include "camera_runtime.h"
#include "camera_service.h"
#include "device_telemetry.h"
#include "image_storage.h"
#include "motion_detector.h"
#include "upload_buffer.h"

namespace {

constexpr size_t kMaxHeaderBytes = 4096;
constexpr size_t kMaxBodyBytes = 2048;
constexpr size_t kNetworkWriteChunkBytes = 1360;
constexpr unsigned long kRequestTimeoutMs = 3000;

WiFiServer server(80);

constexpr char kStreamBoundary[] = "esp32cam-frame";

struct HttpRequest {
  String method;
  String path;
  String body;
  int contentLength = 0;
};

void sendResponse(WiFiClient &client, int status, const char *reason, const char *contentType,
                  const String &body, const String &extraHeaders = "") {
  client.printf("HTTP/1.1 %d %s\r\n", status, reason);
  client.printf("Content-Type: %s\r\n", contentType);
  client.printf("Content-Length: %u\r\n", body.length());
  client.print("Connection: close\r\n");
  client.print("Access-Control-Allow-Origin: *\r\n");
  client.print("Access-Control-Allow-Headers: Content-Type\r\n");
  client.print("Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n");
  client.print("Cache-Control: no-store\r\n");
  if (extraHeaders.length() > 0) client.print(extraHeaders);
  client.print("\r\n");
  client.print(body);
}

void sendJson(WiFiClient &client, int status, const char *reason, const String &body) {
  sendResponse(client, status, reason, "application/json", body);
}

bool writeAll(WiFiClient &client, const uint8_t *data, size_t length) {
  size_t written = 0;
  while (written < length && client.connected()) {
    const size_t remaining = length - written;
    const size_t requested = remaining < kNetworkWriteChunkBytes ? remaining : kNetworkWriteChunkBytes;
    const size_t chunk = client.write(data + written, requested);
    if (chunk != requested) return false;
    written += requested;
    yield();
  }
  return written == length;
}

void restoreSelectedCameraMode() {
  const bool hasError = getCameraRuntimeError().length() > 0;
  if (getCameraOperatingMode() == CameraOperatingMode::Motion) {
    setCameraRuntimeState(CameraRuntimeState::MotionArming);
    resetMotionReference();
    if (hasError) setCameraRuntimeState(CameraRuntimeState::Error);
  } else {
    setCameraRuntimeState(hasError ? CameraRuntimeState::Error : CameraRuntimeState::Idle);
  }
}

void streamJpegs(WiFiClient &client) {
  setCameraRuntimeState(CameraRuntimeState::Streaming);
  clearCameraRuntimeError();

  if (!prepareJpegCapture(true)) {
    setCameraRuntimeError("Could not prepare JPEG streaming mode");
    sendJson(client, 500, "Internal Server Error", "{\"error\":\"JPEG stream setup failed\"}");
    restoreSelectedCameraMode();
    return;
  }

  client.print("HTTP/1.1 200 OK\r\n");
  client.printf("Content-Type: multipart/x-mixed-replace; boundary=%s\r\n", kStreamBoundary);
  client.print("Connection: close\r\nAccess-Control-Allow-Origin: *\r\nCache-Control: no-store\r\n\r\n");

  while (client.connected()) {
    camera_fb_t *frameBuffer = acquireCameraFrame();
    if (!isCompleteJpeg(frameBuffer)) {
      if (frameBuffer != nullptr) esp_camera_fb_return(frameBuffer);
      setCameraRuntimeError("JPEG stream frame capture failed");
      break;
    }

    char partHeader[128];
    const int headerLength = snprintf(partHeader, sizeof(partHeader),
                                      "--%s\r\nContent-Type: image/jpeg\r\n"
                                      "Content-Length: %u\r\n\r\n",
                                      kStreamBoundary,
                                      static_cast<unsigned int>(frameBuffer->len));
    const bool headerValid = headerLength > 0 && headerLength < static_cast<int>(sizeof(partHeader));
    const bool sent = headerValid &&
                      writeAll(client, reinterpret_cast<const uint8_t *>(partHeader), headerLength) &&
                      writeAll(client, frameBuffer->buf, frameBuffer->len) &&
                      writeAll(client, reinterpret_cast<const uint8_t *>("\r\n"), 2);
    esp_camera_fb_return(frameBuffer);
    if (!sent) break;
    delay(1);
  }

  restoreSelectedCameraMode();
}

bool readRequest(WiFiClient &client, HttpRequest &parsed, String &error) {
  client.setTimeout(1);
  String headers;
  const unsigned long startedAt = millis();
  while (!headers.endsWith("\r\n\r\n")) {
    while (client.available()) {
      headers += static_cast<char>(client.read());
      if (headers.length() > kMaxHeaderBytes) {
        error = "request headers too large";
        return false;
      }
      if (headers.endsWith("\r\n\r\n")) break;
    }
    if (millis() - startedAt >= kRequestTimeoutMs) {
      error = "request header timeout";
      return false;
    }
    delay(1);
  }

  const int requestLineEnd = headers.indexOf("\r\n");
  const int firstSpace = headers.indexOf(' ');
  const int secondSpace = headers.indexOf(' ', firstSpace + 1);
  if (requestLineEnd < 0 || firstSpace <= 0 || secondSpace <= firstSpace ||
      secondSpace > requestLineEnd) {
    error = "malformed request line";
    return false;
  }
  parsed.method = headers.substring(0, firstSpace);
  parsed.path = headers.substring(firstSpace + 1, secondSpace);
  const int queryStart = parsed.path.indexOf('?');
  if (queryStart >= 0) parsed.path = parsed.path.substring(0, queryStart);

  int lineStart = requestLineEnd + 2;
  while (lineStart < static_cast<int>(headers.length()) - 2) {
    const int lineEnd = headers.indexOf("\r\n", lineStart);
    if (lineEnd < 0 || lineEnd == lineStart) break;
    const String line = headers.substring(lineStart, lineEnd);
    const int colon = line.indexOf(':');
    if (colon > 0) {
      String name = line.substring(0, colon);
      name.toLowerCase();
      if (name == "content-length") {
        String value = line.substring(colon + 1);
        value.trim();
        parsed.contentLength = value.toInt();
      }
    }
    lineStart = lineEnd + 2;
  }

  if (parsed.contentLength < 0 || parsed.contentLength > static_cast<int>(kMaxBodyBytes)) {
    error = "request body too large";
    return false;
  }

  parsed.body.reserve(parsed.contentLength);
  const unsigned long bodyStartedAt = millis();
  while (parsed.body.length() < static_cast<size_t>(parsed.contentLength)) {
    while (client.available() && parsed.body.length() < static_cast<size_t>(parsed.contentLength)) {
      parsed.body += static_cast<char>(client.read());
    }
    if (millis() - bodyStartedAt >= kRequestTimeoutMs) {
      error = "request body timeout";
      return false;
    }
    delay(1);
  }
  return true;
}

String configJson() {
  JsonDocument document;
  document["wifi_ssid"] = cfg.wifiSsid;
  document["wifi_password_set"] = cfg.wifiPassword[0] != '\0';
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
  String json;
  serializeJson(document, json);
  return json;
}

bool validConfig(const JsonDocument &document, String &error) {
  if (document.containsKey("wifi_ssid")) {
    if (!document["wifi_ssid"].is<const char *>()) {
      error = "wifi_ssid must be a string";
      return false;
    }
    const char *value = document["wifi_ssid"];
    if (strlen(value) == 0 || strlen(value) > 32) {
      error = "wifi_ssid must contain 1 to 32 bytes";
      return false;
    }
  }
  if (document.containsKey("wifi_password")) {
    if (!document["wifi_password"].is<const char *>()) {
      error = "wifi_password must be a string";
      return false;
    }
    if (strlen(document["wifi_password"].as<const char *>()) > 64) {
      error = "wifi_password must contain no more than 64 bytes";
      return false;
    }
  }
  if (document.containsKey("motion_threshold")) {
    const int value = document["motion_threshold"];
    if (value < 0 || value > 255) {
      error = "motion_threshold must be between 0 and 255";
      return false;
    }
  }
  if (document.containsKey("motion_sensitivity")) {
    const float value = document["motion_sensitivity"];
    if (value < 0.0f || value > 1.0f) {
      error = "motion_sensitivity must be between 0 and 1";
      return false;
    }
  }
  if (document.containsKey("motion_check_ms")) {
    const int value = document["motion_check_ms"];
    if (value < 100 || value > 60000) {
      error = "motion_check_ms must be between 100 and 60000";
      return false;
    }
  }
  if (document.containsKey("jpeg_quality")) {
    const int value = document["jpeg_quality"];
    if (value < 0 || value > 63) {
      error = "jpeg_quality must be between 0 and 63";
      return false;
    }
  }
  if (document.containsKey("framesize")) {
    const int value = document["framesize"];
    if (value < static_cast<int>(FRAMESIZE_96X96) || value > static_cast<int>(FRAMESIZE_UXGA)) {
      error = "framesize is unsupported by the OV2640";
      return false;
    }
  }
  if (document.containsKey("server_port")) {
    const int value = document["server_port"];
    if (value < 1 || value > 65535) {
      error = "server_port must be between 1 and 65535";
      return false;
    }
  }
  if (document.containsKey("server_ip")) {
    const char *value = document["server_ip"] | "";
    if (strlen(value) == 0 || strlen(value) >= sizeof(cfg.serverIp)) {
      error = "server_ip must fit an IPv4 address";
      return false;
    }
  }
  if (document.containsKey("bird_confidence_threshold")) {
    const float value = document["bird_confidence_threshold"];
    if (value < 0.0f || value > 1.0f) {
      error = "bird_confidence_threshold must be between 0 and 1";
      return false;
    }
  }
  if (document.containsKey("wait_for_classification") &&
      !document["wait_for_classification"].is<bool>()) {
    error = "wait_for_classification must be true or false";
    return false;
  }
  if (document.containsKey("burst_count")) {
    const int value = document["burst_count"];
    if (value < 1 || value > kMaxBurstCount) {
      error = "burst_count must be between 1 and 10";
      return false;
    }
  }
  if (document.containsKey("burst_interval_ms")) {
    const int value = document["burst_interval_ms"];
    if (value < 0 || value > kMaxBurstIntervalMs) {
      error = "burst_interval_ms must be between 0 and 5000";
      return false;
    }
  }
  if (document.containsKey("keep_reference_after_trigger") &&
      !document["keep_reference_after_trigger"].is<bool>()) {
    error = "keep_reference_after_trigger must be true or false";
    return false;
  }
  if (document.containsKey("buffer_uploads_to_sd") &&
      !document["buffer_uploads_to_sd"].is<bool>()) {
    error = "buffer_uploads_to_sd must be true or false";
    return false;
  }
  if (document.containsKey("max_consecutive_triggers")) {
    const int value = document["max_consecutive_triggers"];
    if (value < 1 || value > kMaxConsecutiveTriggersLimit) {
      error = "max_consecutive_triggers must be between 1 and 100";
      return false;
    }
  }
  float roiX = cfg.motionRoiX;
  float roiY = cfg.motionRoiY;
  float roiWidth = cfg.motionRoiWidth;
  float roiHeight = cfg.motionRoiHeight;
  if (document.containsKey("motion_roi")) {
    const JsonVariantConst roiValue = document["motion_roi"];
    if (!roiValue.is<JsonObjectConst>()) {
      error = "motion_roi must be an object with x, y, width, and height";
      return false;
    }
    const JsonObjectConst roi = roiValue.as<JsonObjectConst>();
    if (!roi.containsKey("x") || !roi.containsKey("y") || !roi.containsKey("width") ||
        !roi.containsKey("height") || !roi["x"].is<float>() || !roi["y"].is<float>() ||
        !roi["width"].is<float>() || !roi["height"].is<float>()) {
      error = "motion_roi requires numeric x, y, width, and height";
      return false;
    }
    roiX = roi["x"];
    roiY = roi["y"];
    roiWidth = roi["width"];
    roiHeight = roi["height"];
    if (!isfinite(roiX) || !isfinite(roiY) || !isfinite(roiWidth) || !isfinite(roiHeight) ||
        roiX < 0.0f || roiY < 0.0f || roiWidth <= 0.0f || roiHeight <= 0.0f ||
        roiX + roiWidth > 1.00001f || roiY + roiHeight > 1.00001f ||
        ceilf(roiWidth * 80.0f) < 16.0f || ceilf(roiHeight * 60.0f) < 12.0f) {
      error = "motion_roi must be within the frame and at least 16x12 motion pixels";
      return false;
    }
  }
  const int settleFrames = document["motion_exposure_settle_frames"] |
                           cfg.motionExposureSettleFrames;
  const float settleTolerance = document["motion_exposure_settle_tolerance"] |
                                cfg.motionExposureSettleTolerance;
  const float shiftTolerance = document["motion_exposure_shift_tolerance"] |
                               cfg.motionExposureShiftTolerance;
  if (settleFrames < 2 || settleFrames > kMaxMotionSettleFrames) {
    error = "motion_exposure_settle_frames must be between 2 and 20";
    return false;
  }
  if (!isfinite(settleTolerance) || settleTolerance < 0.0f || settleTolerance > 1.0f) {
    error = "motion_exposure_settle_tolerance must be between 0 and 1";
    return false;
  }
  if (!isfinite(shiftTolerance) || shiftTolerance < settleTolerance || shiftTolerance > 1.0f) {
    error = "motion_exposure_shift_tolerance must be between settle tolerance and 1";
    return false;
  }
  const int blockColumns = document["motion_block_columns"] | cfg.motionBlockColumns;
  const int blockRows = document["motion_block_rows"] | cfg.motionBlockRows;
  if (blockColumns < 1 || blockColumns > 16 || blockRows < 1 || blockRows > 16 ||
      blockColumns * blockRows > kMaxMotionBlocks) {
    error = "motion block grid must be 1-16 per axis and no more than 64 blocks";
    return false;
  }
  const float ratioThreshold = document["motion_ratio_threshold"] | cfg.motionRatioThreshold;
  if (!isfinite(ratioThreshold) || ratioThreshold < 0.0f || ratioThreshold > 4.0f) {
    error = "motion_ratio_threshold must be between 0 and 4";
    return false;
  }
  const int clipLow = document["motion_clip_low"] | cfg.motionClipLow;
  const int clipHigh = document["motion_clip_high"] | cfg.motionClipHigh;
  if (clipLow < 0 || clipHigh > 255 || clipLow >= clipHigh) {
    error = "motion clipping limits must satisfy 0 <= low < high <= 255";
    return false;
  }
  const float maxClipped = document["motion_max_clipped_fraction"] |
                           cfg.motionMaxClippedFraction;
  if (!isfinite(maxClipped) || maxClipped < 0.0f || maxClipped > 1.0f) {
    error = "motion_max_clipped_fraction must be between 0 and 1";
    return false;
  }
  const int minValidBlocks = document["motion_min_valid_blocks"] | cfg.motionMinValidBlocks;
  if (minValidBlocks < 1 || minValidBlocks > blockColumns * blockRows) {
    error = "motion_min_valid_blocks must fit within the configured block grid";
    return false;
  }
  const float lowerChanged = document["motion_sensitivity"] | cfg.motionSensitivity;
  const float upperChanged = document["motion_max_changed_fraction"] |
                             cfg.motionMaxChangedFraction;
  if (!isfinite(upperChanged) || upperChanged <= lowerChanged || upperChanged > 1.0f) {
    error = "motion_max_changed_fraction must be greater than motion_sensitivity and at most 1";
    return false;
  }
  return true;
}

void applyConfig(const JsonDocument &document) {
  if (document.containsKey("wifi_ssid")) {
    strlcpy(cfg.wifiSsid, document["wifi_ssid"], sizeof(cfg.wifiSsid));
  }
  if (document.containsKey("wifi_password")) {
    strlcpy(cfg.wifiPassword, document["wifi_password"], sizeof(cfg.wifiPassword));
  }
  if (document.containsKey("motion_threshold")) cfg.motionThreshold = document["motion_threshold"];
  if (document.containsKey("motion_sensitivity")) cfg.motionSensitivity = document["motion_sensitivity"];
  if (document.containsKey("motion_check_ms")) cfg.motionCheckMs = document["motion_check_ms"];
  if (document.containsKey("motion_enabled")) cfg.motionEnabled = document["motion_enabled"];
  if (document.containsKey("jpeg_quality")) cfg.jpegQuality = document["jpeg_quality"];
  if (document.containsKey("framesize")) cfg.framesize = document["framesize"];
  if (document.containsKey("server_ip")) {
    strlcpy(cfg.serverIp, document["server_ip"] | cfg.serverIp, sizeof(cfg.serverIp));
  }
  if (document.containsKey("server_port")) cfg.serverPort = document["server_port"];
  if (document.containsKey("save_birds_to_sd")) cfg.saveBirdsToSd = document["save_birds_to_sd"];
  if (document.containsKey("bird_confidence_threshold")) {
    cfg.birdConfidenceThreshold = document["bird_confidence_threshold"];
  }
  if (document.containsKey("wait_for_classification")) {
    cfg.waitForClassification = document["wait_for_classification"];
  }
  if (document.containsKey("burst_count")) cfg.burstCount = document["burst_count"];
  if (document.containsKey("burst_interval_ms")) cfg.burstIntervalMs = document["burst_interval_ms"];
  if (document.containsKey("keep_reference_after_trigger")) {
    cfg.keepReferenceAfterTrigger = document["keep_reference_after_trigger"];
  }
  if (document.containsKey("max_consecutive_triggers")) {
    cfg.maxConsecutiveTriggers = document["max_consecutive_triggers"];
  }
  if (document.containsKey("buffer_uploads_to_sd")) {
    cfg.bufferUploadsToSd = document["buffer_uploads_to_sd"];
  }
  const JsonVariantConst roiValue = document["motion_roi"];
  if (roiValue.is<JsonObjectConst>()) {
    const JsonObjectConst roi = roiValue.as<JsonObjectConst>();
    cfg.motionRoiX = roi["x"];
    cfg.motionRoiY = roi["y"];
    cfg.motionRoiWidth = roi["width"];
    cfg.motionRoiHeight = roi["height"];
  }
  if (document.containsKey("motion_exposure_settle_frames")) {
    cfg.motionExposureSettleFrames = document["motion_exposure_settle_frames"];
  }
  if (document.containsKey("motion_exposure_settle_tolerance")) {
    cfg.motionExposureSettleTolerance = document["motion_exposure_settle_tolerance"];
  }
  if (document.containsKey("motion_exposure_shift_tolerance")) {
    cfg.motionExposureShiftTolerance = document["motion_exposure_shift_tolerance"];
  }
  if (document.containsKey("motion_block_columns")) {
    cfg.motionBlockColumns = document["motion_block_columns"];
  }
  if (document.containsKey("motion_block_rows")) cfg.motionBlockRows = document["motion_block_rows"];
  if (document.containsKey("motion_ratio_threshold")) {
    cfg.motionRatioThreshold = document["motion_ratio_threshold"];
  }
  if (document.containsKey("motion_clip_low")) cfg.motionClipLow = document["motion_clip_low"];
  if (document.containsKey("motion_clip_high")) cfg.motionClipHigh = document["motion_clip_high"];
  if (document.containsKey("motion_max_clipped_fraction")) {
    cfg.motionMaxClippedFraction = document["motion_max_clipped_fraction"];
  }
  if (document.containsKey("motion_min_valid_blocks")) {
    cfg.motionMinValidBlocks = document["motion_min_valid_blocks"];
  }
  if (document.containsKey("motion_max_changed_fraction")) {
    cfg.motionMaxChangedFraction = document["motion_max_changed_fraction"];
  }
}

void handleClient(WiFiClient client) {
  HttpRequest request;
  String requestError;
  if (!readRequest(client, request, requestError)) {
    const bool tooLarge = requestError.indexOf("large") >= 0;
    sendJson(client, tooLarge ? 413 : 400, tooLarge ? "Payload Too Large" : "Bad Request",
             "{\"error\":\"" + requestError + "\"}");
    client.stop();
    return;
  }

  if (request.method == "OPTIONS") {
    sendResponse(client, 204, "No Content", "text/plain", "");
  } else if (request.method == "GET" && request.path == "/stream") {
    streamJpegs(client);
  } else if (request.method == "GET" && request.path == "/capture") {
    setCameraRuntimeState(CameraRuntimeState::Capturing);
    CapturedJpeg capture;
    if (!captureJpeg(capture, true)) {
      setCameraRuntimeError("Manual JPEG capture failed");
      sendJson(client, 500, "Internal Server Error", "{\"error\":\"JPEG capture failed\"}");
    } else {
      camera_fb_t *frameBuffer = capture.frameBuffer;
      const String metadata = buildCameraMetadata(capture.metadata, "preview");
      client.print("HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\n");
      client.printf("Content-Length: %u\r\n", frameBuffer->len);
      client.print("Connection: close\r\nAccess-Control-Allow-Origin: *\r\nCache-Control: no-store\r\n");
      client.print("X-ESP32-Camera-Metadata: ");
      client.print(metadata);
      client.print("\r\n\r\n");
      const bool sent = writeAll(client, frameBuffer->buf, frameBuffer->len);
      esp_camera_fb_return(frameBuffer);
      if (sent) {
        acknowledgeRegisterDumpDelivery(capture.metadata);
        clearCameraRuntimeError();
      } else {
        setCameraRuntimeError("Manual JPEG response was interrupted");
      }
    }
    restoreSelectedCameraMode();
  } else if (request.method == "GET" && request.path == "/metadata") {
    const CameraMetadataSnapshot snapshot = sampleCameraMetadata();
    sendJson(client, 200, "OK", buildCameraMetadata(snapshot, "profile"));
    acknowledgeRegisterDumpDelivery(snapshot);
  } else if (request.method == "GET" && request.path == "/motion") {
    sendJson(client, 200, "OK",
             consumeMotionFlag() ? "{\"motion\":true}" : "{\"motion\":false}");
  } else if (request.method == "GET" && request.path == "/config") {
    sendJson(client, 200, "OK", configJson());
  } else if (request.method == "POST" && request.path == "/config") {
    JsonDocument document;
    const DeserializationError error = deserializeJson(document, request.body);
    if (error || !document.is<JsonObject>()) {
      sendJson(client, 400, "Bad Request", "{\"error\":\"Expected a JSON object\"}");
    } else {
      String validationError;
      if (!validConfig(document, validationError)) {
        sendJson(client, 422, "Unprocessable Content",
                 "{\"error\":\"" + validationError + "\"}");
      } else {
        const Config previousConfig = cfg;
        applyConfig(document);
        saveConfig();
        const bool wifiSettingsChanged = strcmp(cfg.wifiSsid, previousConfig.wifiSsid) != 0 ||
                                         strcmp(cfg.wifiPassword, previousConfig.wifiPassword) != 0;
        if (cfg.motionEnabled != previousConfig.motionEnabled) {
          if (cfg.motionEnabled) {
            armMotionDetection();
          } else {
            disarmMotionDetection();
          }
        } else if (getCameraOperatingMode() == CameraOperatingMode::Motion) {
          resetMotionReference();
        } else {
          applyCameraSettings();
        }
        if (cfg.jpegQuality != previousConfig.jpegQuality || cfg.framesize != previousConfig.framesize) {
          markRegisterDumpPending();
        }
        if (wifiSettingsChanged) {
          Serial.println("Wi-Fi configuration saved; it will be used after restart");
        }
        sendJson(client, 200, "OK", configJson());
      }
    }
  } else if (request.method == "POST" && request.path == "/mode") {
    JsonDocument document;
    const DeserializationError error = deserializeJson(document, request.body);
    const char *mode = document["mode"] | "";
    if (error || !document.is<JsonObject>() || (strcmp(mode, "motion") != 0 && strcmp(mode, "idle") != 0)) {
      sendJson(client, 422, "Unprocessable Content",
               "{\"error\":\"mode must be either motion or idle\"}");
    } else {
      if (strcmp(mode, "motion") == 0) {
        armMotionDetection();
      } else {
        disarmMotionDetection();
      }
      saveConfig();
      JsonDocument response;
      response["selected_mode"] = cameraOperatingModeName(getCameraOperatingMode());
      response["runtime_state"] = cameraRuntimeStateName(getCameraRuntimeState());
      String json;
      serializeJson(response, json);
      sendJson(client, 200, "OK", json);
    }
  } else if (request.method == "GET" && request.path == "/status") {
    JsonDocument document;
    document["selected_mode"] = cameraOperatingModeName(getCameraOperatingMode());
    document["runtime_state"] = cameraRuntimeStateName(getCameraRuntimeState());
    document["stream_active"] = isCameraStreamActive();
    document["boot_id"] = getBootId();
    const DeviceTelemetrySnapshot device = sampleDeviceTelemetry();
    document["time_sync_status"] = device.timeSynchronized ? "synchronized" : "unsynchronized";
    if (device.timeSynchronized) {
      document["device_time_utc"] = device.deviceTimeUtc;
    } else {
      document["device_time_utc"] = nullptr;
    }
    document["chip_temperature_c"] = device.chipTemperatureC;
    document["chip_temperature_qualifier"] =
        "uncalibrated_internal_die_temperature_not_ambient";
    document["free_heap_bytes"] = device.freeHeapBytes;
    document["minimum_free_heap_bytes_since_boot"] = device.minimumFreeHeapBytes;
    document["largest_free_heap_block_bytes"] = device.largestFreeHeapBlockBytes;
    document["psram_total_bytes"] = device.psramTotalBytes;
    document["psram_free_bytes"] = device.psramFreeBytes;
    document["psram_minimum_free_bytes_since_boot"] = device.psramMinimumFreeBytes;
    document["psram_largest_free_block_bytes"] = device.psramLargestFreeBlockBytes;
    document["config_file_hash_valid"] = isConfigFileHashValid();
    if (isConfigFileHashValid()) {
      document["config_file_sha256"] = getConfigFileSha256();
    } else {
      document["config_file_sha256"] = nullptr;
    }
    document["motion_flag"] = isMotionFlagSet();
    document["motion_enabled"] = cfg.motionEnabled;
    document["threshold"] = cfg.motionThreshold;
    document["sensitivity"] = cfg.motionSensitivity;
    document["ref_ready"] = isReferenceFrameReady();
    document["motion_event_count"] = getMotionEventCount();
    document["last_motion_ms"] = getLastMotionMs();
    document["last_changed_fraction"] = getLastChangedFraction();
    document["motion_exposure_settled"] = isMotionExposureSettled();
    document["motion_exposure_settle_count"] = getMotionExposureSettleCount();
    document["motion_exposure_settle_required"] = cfg.motionExposureSettleFrames;
    document["motion_exposure_product_valid"] = isMotionExposureProductValid();
    if (isMotionExposureProductValid()) {
      document["motion_current_exposure_gain_product"] = getMotionCurrentExposureProduct();
    } else {
      document["motion_current_exposure_gain_product"] = nullptr;
    }
    if (isReferenceFrameReady()) {
      document["motion_reference_exposure_gain_product"] = getMotionReferenceExposureProduct();
    } else {
      document["motion_reference_exposure_gain_product"] = nullptr;
    }
    document["motion_last_exposure_shift"] = getMotionLastExposureShift();
    document["motion_last_global_ratio"] = getMotionLastGlobalRatio();
    document["motion_last_valid_blocks"] = getMotionLastValidBlocks();
    document["motion_last_clipped_blocks"] = getMotionLastClippedBlocks();
    document["motion_last_changed_blocks"] = getMotionLastChangedBlocks();
    document["motion_last_decision"] = getMotionLastDecision();
    int roiX = 0;
    int roiY = 0;
    int roiWidth = 0;
    int roiHeight = 0;
    getMotionRoiPixels(roiX, roiY, roiWidth, roiHeight);
    JsonObject roi = document["motion_roi_pixels"].to<JsonObject>();
    roi["x"] = roiX;
    roi["y"] = roiY;
    roi["width"] = roiWidth;
    roi["height"] = roiHeight;
    document["last_motion_capture_succeeded"] = wasLastMotionCaptureSuccessful();
    document["last_motion_upload_succeeded"] = wasLastMotionUploadSuccessful();
    document["upload_cooldown_active"] = isMotionUploadCoolingDown();
    document["buffered_upload_count"] = getBufferedUploadCount();
    document["consecutive_motion_triggers"] = getConsecutiveMotionTriggers();
    if (getCameraRuntimeError().length() > 0) {
      document["last_error"] = getCameraRuntimeError();
    } else {
      document["last_error"] = nullptr;
    }
    document["sd_ready"] = sdReady;
    document["config_storage"] = getConfigStorageName();
    document["bird_storage_enabled"] = cfg.saveBirdsToSd;
    document["bird_confidence_threshold"] = cfg.birdConfidenceThreshold;
    document["last_classification_received"] = wasLastClassificationReceived();
    document["last_classification_is_bird"] = wasLastClassificationBird();
    document["last_classification_confidence"] = getLastClassificationConfidence();
    document["last_sd_save_succeeded"] = wasLastSdSaveSuccessful();
    document["last_saved_image"] = getLastSavedImagePath();
    String json;
    serializeJson(document, json);
    sendJson(client, 200, "OK", json);
  } else {
    sendJson(client, 404, "Not Found", "{\"error\":\"Endpoint not found\"}");
  }

  client.stop();
}

}  // namespace

void startWebServer() {
  server.begin();
  Serial.println("HTTP server started");
}

void handleWebClient() {
  WiFiClient client = server.available();
  if (client) handleClient(client);
}