#pragma once

#include <Arduino.h>

// Remote visibility for a camera that has no serial cable attached.
//
// - logEvent() writes a line to Serial *and* to a small ring buffer kept in RTC
//   memory, which survives crashes, watchdog resets and software reboots (it is
//   only cleared by a real power-on). The last ~32 events before a crash can
//   therefore be read back after the reboot.
// - After a crash the ESP-IDF panic handler saves a core dump to the flash
//   "coredump" partition. On the next boot its summary (task, PC, exception,
//   backtrace) is read, kept in RTC memory and reported.
// - A background task POSTs all of this to http://<hub>:<server_port>/api/heartbeat, where
//   <hub> is server_host looked up by name, or server_ip (see hub_locator.h),
//   every 15 s. The server can answer {"reboot": true} to restart the camera,
//   which works even when the camera's own web server is wedged.
// - GET /health on the camera returns the same JSON; POST /reboot restarts it.

// Call first thing in setup(), right after Serial.begin().
void initializeDeviceHealth();

// Call once the camera and web server are up (or have failed).
void startDeviceHealthMonitor();

void logEvent(const char *format, ...) __attribute__((format(printf, 1, 2)));

String deviceHealthJson();

[[noreturn]] void rebootDevice(const char *reason);
