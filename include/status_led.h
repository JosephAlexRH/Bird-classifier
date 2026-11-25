#pragma once

#include <Arduino.h>

// Colour status light on the board's RGB LED (Freenove ESP32-S3: WS2812 on
// GPIO 48). Boards without one compile these to no-ops.
//
// Boot flashes, in order: white (booted), orange x2 (last restart was a crash,
// watchdog or brownout), purple (no SD card; running on SPIFFS config).
// Then, highest priority first:
//   red fast blink      camera failed to start (reboot within 60 s)
//   cyan solid          streaming
//   white flicker       a frame was just captured ("on" mode only)
//   blue slow blink     connecting to Wi-Fi, or waiting for the first heartbeat
//   yellow slow blink   on Wi-Fi but the hub is not answering heartbeats
//   green slow fade     all good, camera asleep (idle power-down)
//   green short blink   all good
//
// status_led in config.json: "off", "boot" (default: lit for the first
// 5 minutes after power-up, then dark) or "on" (always lit).

enum class StatusLedMode : uint8_t { Off = 0, Boot = 1, On = 2 };

const char *statusLedModeName(StatusLedMode mode);
bool parseStatusLedMode(const char *name, StatusLedMode &mode);

// Start the LED task. Call after loadConfig() and initializeDeviceHealth().
void startStatusLed();
// Camera initialization failed; shows red until the device reboots.
void statusLedNoteCameraFailed();
// A frame was captured (shown only in "on" mode).
void statusLedNoteCapture();
