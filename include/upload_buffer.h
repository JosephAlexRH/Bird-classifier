#pragma once

#include <Arduino.h>
#include <esp_camera.h>

// SD-card holding area for motion JPEGs that could not be delivered. Frames are
// written to /pending with their metadata and uploaded later, one per flush call,
// so motion checks keep running between retries.
void initializeUploadBuffer();
bool bufferUpload(const camera_fb_t *frameBuffer, const String &metadata);
void flushOneBufferedUpload();
uint32_t getBufferedUploadCount();
