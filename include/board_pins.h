#pragma once

#include <sdkconfig.h>  // CONFIG_IDF_TARGET_* even if included before Arduino.h

// Per-board camera, SD card and LED wiring. Exactly one BIRDCAM_BOARD_* is set
// by the PlatformIO environment (see platformio.ini).

#if defined(BIRDCAM_BOARD_FREENOVE_S3)

// Freenove ESP32-S3 WROOM CAM. Camera pins match Espressif's
// CAMERA_MODEL_ESP32S3_EYE, which is what Freenove's own CameraWebServer
// sketch selects. The camera has no PWDN or RESET line on this board, so the
// sensor cannot be hard-powered-down; idle only stops XCLK and DMA.
#define BIRDCAM_BOARD_NAME "freenove_esp32s3_wroom_cam"
#define BIRDCAM_CAM_HAS_PWDN 0
constexpr int kCamPwdnPin = -1;
constexpr int kCamResetPin = -1;
constexpr int kCamXclkPin = 15;
constexpr int kCamSiodPin = 4;
constexpr int kCamSiocPin = 5;
constexpr int kCamY9Pin = 16;
constexpr int kCamY8Pin = 17;
constexpr int kCamY7Pin = 18;
constexpr int kCamY6Pin = 12;
constexpr int kCamY5Pin = 10;
constexpr int kCamY4Pin = 8;
constexpr int kCamY3Pin = 9;
constexpr int kCamY2Pin = 11;
constexpr int kCamVsyncPin = 6;
constexpr int kCamHrefPin = 7;
constexpr int kCamPclkPin = 13;

// SD card is 1-bit SDMMC on GPIO matrix pins (Freenove Sketch_04.1).
constexpr bool kSdNeedsPins = true;
constexpr int kSdClkPin = 39;
constexpr int kSdCmdPin = 38;
constexpr int kSdD0Pin = 40;

// Plain status LED on GPIO 2 (unused by the firmware).
constexpr int kStatusLedPin = 2;

// WS2812 RGB LED on GPIO 48: the status light (status_led.h). setup() blanks
// it first, since an undriven WS2812 keeps a random power-up colour.
#define BIRDCAM_HAS_RGB_LED 1
constexpr int kRgbLedPin = 48;

#elif defined(BIRDCAM_BOARD_AI_THINKER) || !defined(CONFIG_IDF_TARGET_ESP32S3)

// AI Thinker ESP32-CAM (the original fleet boards).
#define BIRDCAM_BOARD_NAME "ai_thinker_esp32cam"
#define BIRDCAM_CAM_HAS_PWDN 1
constexpr int kCamPwdnPin = 32;
constexpr int kCamResetPin = -1;
constexpr int kCamXclkPin = 0;
constexpr int kCamSiodPin = 26;
constexpr int kCamSiocPin = 27;
constexpr int kCamY9Pin = 35;
constexpr int kCamY8Pin = 34;
constexpr int kCamY7Pin = 39;
constexpr int kCamY6Pin = 36;
constexpr int kCamY5Pin = 21;
constexpr int kCamY4Pin = 19;
constexpr int kCamY3Pin = 18;
constexpr int kCamY2Pin = 5;
constexpr int kCamVsyncPin = 25;
constexpr int kCamHrefPin = 23;
constexpr int kCamPclkPin = 22;

// SD_MMC uses the fixed SDMMC slot pins on the classic ESP32.
constexpr bool kSdNeedsPins = false;
constexpr int kSdClkPin = 14;
constexpr int kSdCmdPin = 15;
constexpr int kSdD0Pin = 2;

constexpr int kStatusLedPin = 33;  // red LED, active low (unused by firmware today)
#define BIRDCAM_HAS_RGB_LED 0

#else
#error "ESP32-S3 build without a BIRDCAM_BOARD_* flag: add a pin map to board_pins.h"
#endif

// Cameras without a PWDN line can't be hard-powered-down between uses. Guard
// PWDN code with #if BIRDCAM_CAM_HAS_PWDN so -1 never reaches pinMode().
static_assert((kCamPwdnPin >= 0) == (BIRDCAM_CAM_HAS_PWDN == 1), "PWDN pin and flag disagree");
