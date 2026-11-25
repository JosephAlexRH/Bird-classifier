#include "status_led.h"

#include <WiFi.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "app_config.h"
#include "board_pins.h"
#include "camera_runtime.h"
#include "camera_service.h"
#include "device_health.h"

const char *statusLedModeName(StatusLedMode mode) {
  switch (mode) {
    case StatusLedMode::Off: return "off";
    case StatusLedMode::On: return "on";
    case StatusLedMode::Boot:
    default: return "boot";
  }
}

bool parseStatusLedMode(const char *name, StatusLedMode &mode) {
  if (name == nullptr) return false;
  if (strcasecmp(name, "off") == 0) mode = StatusLedMode::Off;
  else if (strcasecmp(name, "boot") == 0) mode = StatusLedMode::Boot;
  else if (strcasecmp(name, "on") == 0) mode = StatusLedMode::On;
  else return false;
  return true;
}

#if BIRDCAM_HAS_RGB_LED

namespace {

struct Rgb {
  uint8_t r, g, b;
  bool operator==(const Rgb &other) const { return r == other.r && g == other.g && b == other.b; }
  bool operator!=(const Rgb &other) const { return !(*this == other); }
};

// About 10 % brightness: visible up close, doesn't light up the enclosure.
constexpr Rgb kOff{0, 0, 0};
constexpr Rgb kWhite{24, 24, 24};
constexpr Rgb kOrange{26, 9, 0};
constexpr Rgb kPurple{18, 0, 26};
constexpr Rgb kBlue{0, 0, 26};
constexpr Rgb kYellow{22, 16, 0};
constexpr Rgb kGreen{0, 24, 0};
constexpr Rgb kCyan{0, 20, 20};
constexpr Rgb kRed{26, 0, 0};

constexpr uint32_t kBootWindowMs = 5UL * 60UL * 1000UL;
constexpr uint32_t kTickMs = 40;
constexpr uint32_t kCaptureFlickerMs = 60;

TaskHandle_t ledTask = nullptr;
volatile bool cameraFailed = false;
volatile uint32_t lastCaptureMs = 0;
volatile bool captureSeen = false;
Rgb shown{1, 1, 1};  // forces the first write

void show(const Rgb &colour) {
  if (colour == shown) return;
  neopixelWrite(kRgbLedPin, colour.r, colour.g, colour.b);
  shown = colour;
}

void flash(const Rgb &colour, uint32_t onMs, uint32_t offMs) {
  show(colour);
  vTaskDelay(pdMS_TO_TICKS(onMs));
  show(kOff);
  vTaskDelay(pdMS_TO_TICKS(offMs));
}

// on for onMs at the start of every periodMs.
bool blinkPhase(uint32_t now, uint32_t periodMs, uint32_t onMs) { return now % periodMs < onMs; }

Rgb scaled(const Rgb &colour, uint32_t numerator, uint32_t denominator) {
  return {static_cast<uint8_t>(colour.r * numerator / denominator),
          static_cast<uint8_t>(colour.g * numerator / denominator),
          static_cast<uint8_t>(colour.b * numerator / denominator)};
}

bool litNow(uint32_t now) {
  switch (static_cast<StatusLedMode>(cfg.statusLedMode)) {
    case StatusLedMode::Off: return false;
    case StatusLedMode::On: return true;
    case StatusLedMode::Boot:
    default: return now < kBootWindowMs;
  }
}

Rgb stateColour(uint32_t now) {
  if (cameraFailed) return blinkPhase(now, 200, 100) ? kRed : kOff;
  if (isCameraStreamActive()) return kCyan;
  if (static_cast<StatusLedMode>(cfg.statusLedMode) == StatusLedMode::On && captureSeen &&
      now - lastCaptureMs < kCaptureFlickerMs) {
    return kWhite;
  }
  if (WiFi.status() != WL_CONNECTED) return blinkPhase(now, 1500, 250) ? kBlue : kOff;
  switch (getHubLinkState()) {
    case HubLinkState::Unknown: return blinkPhase(now, 1500, 250) ? kBlue : kOff;
    case HubLinkState::Unreachable: return blinkPhase(now, 1500, 250) ? kYellow : kOff;
    case HubLinkState::Reachable: break;
  }
  if (!isCameraPowered()) {
    // Triangle-wave fade over 3 s.
    const uint32_t phase = now % 3000;
    const uint32_t level = phase < 1500 ? phase : 3000 - phase;
    return scaled(kGreen, level, 1500);
  }
  return blinkPhase(now, 3000, 80) ? kGreen : kOff;
}

void statusLedTask(void *) {
  if (litNow(millis())) {
    flash(kWhite, 250, 250);
    if (didLastResetFail()) {
      flash(kOrange, 150, 150);
      flash(kOrange, 150, 300);
    }
    if (!sdReady) flash(kPurple, 300, 300);
  }

  for (;;) {
    const uint32_t now = millis();
    show(litNow(now) ? stateColour(now) : kOff);
    vTaskDelay(pdMS_TO_TICKS(kTickMs));
  }
}

}  // namespace

void startStatusLed() {
  if (ledTask != nullptr) return;
  neopixelWrite(kRgbLedPin, 0, 0, 0);
  shown = kOff;
  // Runs even in "off" mode so a later change from the API takes effect; while
  // dark it only wakes every 40 ms and writes nothing.
  xTaskCreatePinnedToCore(statusLedTask, "status_led", 2048, nullptr, 1, &ledTask, 0);
}

void statusLedNoteCameraFailed() { cameraFailed = true; }

void statusLedNoteCapture() {
  lastCaptureMs = millis();
  captureSeen = true;
}

#else  // no RGB LED on this board

void startStatusLed() {}
void statusLedNoteCameraFailed() {}
void statusLedNoteCapture() {}

#endif
