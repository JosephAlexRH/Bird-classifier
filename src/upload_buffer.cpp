#include "upload_buffer.h"

#include <Arduino.h>
#include <FS.h>
#include <HTTPClient.h>
#include <SD_MMC.h>
#include <esp_system.h>

#include "app_config.h"
#include "camera_runtime.h"
#include "camera_service.h"
#include "motion_detector.h"

namespace {

constexpr char kPendingDirectory[] = "/pending";
constexpr uint32_t kMaxBufferedUploads = 1000;

uint32_t bufferedCount = 0;
uint32_t sequence = 0;
uint32_t sessionTag = 0;

// File::name() is the bare filename on Arduino-ESP32 2.x; tolerate a full path.
String baseName(const char *name) {
  const String value(name);
  const int slash = value.lastIndexOf('/');
  return slash >= 0 ? value.substring(slash + 1) : value;
}

String readTextFile(const String &path) {
  String text;
  File file = SD_MMC.open(path, FILE_READ);
  if (!file) return text;
  text.reserve(file.size());
  char chunk[257];
  while (file.available()) {
    const int length = file.read(reinterpret_cast<uint8_t *>(chunk), sizeof(chunk) - 1);
    if (length <= 0) break;
    chunk[length] = '\0';
    text += chunk;
  }
  file.close();
  return text;
}

void removeBuffered(const String &jpgPath, const String &jsonPath) {
  SD_MMC.remove(jpgPath);
  SD_MMC.remove(jsonPath);
  if (bufferedCount > 0) --bufferedCount;
}

}  // namespace

void initializeUploadBuffer() {
  sessionTag = esp_random();
  bufferedCount = 0;
  if (!sdReady) return;

  File directory = SD_MMC.open(kPendingDirectory);
  if (!directory || !directory.isDirectory()) return;
  for (File entry = directory.openNextFile(); entry; entry = directory.openNextFile()) {
    if (!entry.isDirectory() && baseName(entry.name()).endsWith(".jpg")) ++bufferedCount;
    entry.close();
  }
  directory.close();
  Serial.printf("SD upload buffer: %u frame(s) waiting from earlier sessions\n", bufferedCount);
}

bool bufferUpload(const camera_fb_t *frameBuffer, const String &metadata) {
  if (!cfg.bufferUploadsToSd || !sdReady || !isCompleteJpeg(frameBuffer)) return false;
  if (bufferedCount >= kMaxBufferedUploads) {
    Serial.println("SD upload buffer is full; frame dropped");
    return false;
  }
  if (!SD_MMC.exists(kPendingDirectory) && !SD_MMC.mkdir(kPendingDirectory)) {
    Serial.println("Failed to create /pending on SD card");
    return false;
  }

  char stem[48];
  snprintf(stem, sizeof(stem), "%s/%08lx_%06lu", kPendingDirectory,
           static_cast<unsigned long>(sessionTag), static_cast<unsigned long>(++sequence));
  const String jsonPath = String(stem) + ".json";
  const String partPath = String(stem) + ".part";
  const String jpgPath = String(stem) + ".jpg";

  File metadataFile = SD_MMC.open(jsonPath, FILE_WRITE);
  if (!metadataFile) return false;
  metadataFile.print(metadata);
  metadataFile.close();

  // The JPEG only gains its .jpg name once fully written, so a power cut cannot
  // leave a truncated image that the flush would try to upload.
  File image = SD_MMC.open(partPath, FILE_WRITE);
  size_t written = 0;
  if (image) {
    written = image.write(frameBuffer->buf, frameBuffer->len);
    image.close();
  }
  if (written != frameBuffer->len || !SD_MMC.rename(partPath, jpgPath)) {
    SD_MMC.remove(partPath);
    SD_MMC.remove(jsonPath);
    Serial.println("SD upload buffer write failed; frame dropped");
    return false;
  }

  ++bufferedCount;
  Serial.printf("Buffered to SD: %s (%u bytes, %u waiting)\n", jpgPath.c_str(), written,
                bufferedCount);
  return true;
}

void flushOneBufferedUpload() {
  if (bufferedCount == 0 || !sdReady || isMotionUploadCoolingDown()) return;

  File directory = SD_MMC.open(kPendingDirectory);
  if (!directory || !directory.isDirectory()) {
    bufferedCount = 0;
    return;
  }
  String jpgPath;
  for (File entry = directory.openNextFile(); entry; entry = directory.openNextFile()) {
    const String name = baseName(entry.name());
    const bool isJpeg = !entry.isDirectory() && name.endsWith(".jpg");
    entry.close();
    if (isJpeg) {
      jpgPath = String(kPendingDirectory) + "/" + name;
      break;
    }
  }
  directory.close();
  if (jpgPath.length() == 0) {
    bufferedCount = 0;
    return;
  }
  const String jsonPath = jpgPath.substring(0, jpgPath.length() - 4) + ".json";

  File image = SD_MMC.open(jpgPath, FILE_READ);
  const size_t length = image ? image.size() : 0;
  uint8_t *data = nullptr;
  if (length > 0) {
    data = static_cast<uint8_t *>(psramFound() ? ps_malloc(length) : malloc(length));
  }
  if (length > 0 && data == nullptr) {
    image.close();
    Serial.println("Not enough memory to upload a buffered frame; will retry");
    return;
  }
  const bool readOk = data != nullptr && image.read(data, length) == length;
  if (image) image.close();
  const bool valid = readOk && length >= 4 && data[0] == 0xff && data[1] == 0xd8 &&
                     data[length - 2] == 0xff && data[length - 1] == 0xd9;
  if (!valid) {
    free(data);
    Serial.printf("Discarding unreadable buffered frame %s\n", jpgPath.c_str());
    removeBuffered(jpgPath, jsonPath);
    return;
  }

  const String metadata = readTextFile(jsonPath);
  const CameraRuntimeState previousState = getCameraRuntimeState();
  setCameraRuntimeState(CameraRuntimeState::Uploading);

  HTTPClient http;
  char url[64];
  snprintf(url, sizeof(url), "http://%s:%d/upload", cfg.serverIp, cfg.serverPort);
  http.begin(url);
  http.addHeader("Content-Type", "image/jpeg");
  if (metadata.length() > 0) http.addHeader("X-ESP32-Camera-Metadata", metadata);
  // A late frame has no live framebuffer to act on, so never wait for a verdict.
  http.addHeader("X-ESP32-Classification", "deferred");
  http.addHeader("X-ESP32-Buffered", "true");
  http.setConnectTimeout(1500);
  http.setTimeout(3000);
  const int httpCode = http.POST(data, length);
  http.end();
  free(data);

  if (httpCode >= 200 && httpCode < 300) {
    removeBuffered(jpgPath, jsonPath);
    setCameraRuntimeState(previousState);
    Serial.printf("Buffered frame delivered: %s (%u still waiting)\n", jpgPath.c_str(),
                  bufferedCount);
  } else if (httpCode >= 400 && httpCode < 500) {
    // The receiver will never accept this file; do not let it block the rest.
    removeBuffered(jpgPath, jsonPath);
    setCameraRuntimeState(previousState);
    Serial.printf("Receiver rejected buffered frame %s with %d; discarded\n", jpgPath.c_str(),
                  httpCode);
  } else {
    noteMotionUploadFailure();
    setCameraRuntimeState(CameraRuntimeState::UploadCooldown);
    Serial.printf("Buffered frame upload failed (%d); %u waiting\n", httpCode, bufferedCount);
  }
}

uint32_t getBufferedUploadCount() { return bufferedCount; }
