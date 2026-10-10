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
  String json;
  serializeJson(document, json);
  return json;
}

bool validConfig(const JsonDocument &document, String &error) {
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
  return true;
}

void applyConfig(const JsonDocument &document) {
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
    document["last_motion_capture_succeeded"] = wasLastMotionCaptureSuccessful();
    document["last_motion_upload_succeeded"] = wasLastMotionUploadSuccessful();
    document["upload_cooldown_active"] = isMotionUploadCoolingDown();
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