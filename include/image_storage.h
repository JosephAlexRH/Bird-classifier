#pragma once

#include <Arduino.h>
#include <esp_camera.h>

bool saveBirdImage(const camera_fb_t *frameBuffer, const char *captureId);
void recordClassificationResult(bool received, bool isBird, float confidence);
bool wasLastClassificationReceived();
bool wasLastClassificationBird();
float getLastClassificationConfidence();
bool wasLastSdSaveSuccessful();
const String &getLastSavedImagePath();