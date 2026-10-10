#include "image_storage.h"

#include <Arduino.h>
#include <FS.h>
#include <SD_MMC.h>

#include "app_config.h"
#include "camera_service.h"

namespace {

constexpr char kBirdDirectory[] = "/birds";

bool lastClassificationReceived = false;
bool lastClassificationBird = false;
float lastClassificationConfidence = 0.0f;
bool lastSdSaveSuccessful = false;
String lastSavedImagePath;
uint32_t fallbackSequence = 0;

String safeCaptureId(const char *captureId) {
  String safe;
  if (captureId != nullptr) {
    for (const char *character = captureId; *character != '\0' && safe.length() < 80; ++character) {
      const bool allowed = (*character >= 'a' && *character <= 'z') ||
                           (*character >= 'A' && *character <= 'Z') ||
                           (*character >= '0' && *character <= '9') || *character == '-' ||
                           *character == '_';
      safe += allowed ? *character : '_';
    }
  }
  if (safe.length() == 0) {
    safe = "bird_" + String(millis()) + "_" + String(++fallbackSequence);
  }
  return safe;
}

}  // namespace

bool saveBirdImage(const camera_fb_t *frameBuffer, const char *captureId) {
  lastSdSaveSuccessful = false;
  if (!cfg.saveBirdsToSd || !sdReady || !isCompleteJpeg(frameBuffer)) {
    return false;
  }

  if (!SD_MMC.exists(kBirdDirectory) && !SD_MMC.mkdir(kBirdDirectory)) {
    Serial.println("Failed to create /birds on SD card");
    return false;
  }

  const String path = String(kBirdDirectory) + "/" + safeCaptureId(captureId) + ".jpg";
  File file = SD_MMC.open(path, FILE_WRITE);
  if (!file) {
    Serial.printf("Failed to open %s for writing\n", path.c_str());
    return false;
  }

  const size_t written = file.write(frameBuffer->buf, frameBuffer->len);
  file.flush();
  file.close();
  if (written != frameBuffer->len) {
    SD_MMC.remove(path);
    Serial.printf("Incomplete SD write for %s: %u of %u bytes\n", path.c_str(), written,
                  frameBuffer->len);
    return false;
  }

  lastSdSaveSuccessful = true;
  lastSavedImagePath = path;
  Serial.printf("Bird image saved to SD: %s (%u bytes)\n", path.c_str(), written);
  return true;
}

void recordClassificationResult(bool received, bool isBird, float confidence) {
  lastClassificationReceived = received;
  lastClassificationBird = received && isBird;
  lastClassificationConfidence = received ? confidence : 0.0f;
  lastSdSaveSuccessful = false;
}

bool wasLastClassificationReceived() { return lastClassificationReceived; }

bool wasLastClassificationBird() { return lastClassificationBird; }

float getLastClassificationConfidence() { return lastClassificationConfidence; }

bool wasLastSdSaveSuccessful() { return lastSdSaveSuccessful; }

const String &getLastSavedImagePath() { return lastSavedImagePath; }