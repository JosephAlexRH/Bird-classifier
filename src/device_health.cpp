#include "device_health.h"

#include <ArduinoJson.h>
#include <HTTPClient.h>
#include <WiFi.h>
#include <esp_arduino_version.h>
#include <esp_core_dump.h>
#include <esp_ota_ops.h>
#include <esp_system.h>
#include <esp_timer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <stdarg.h>

#include "app_config.h"
#include "camera_runtime.h"
#include "camera_service.h"
#include "device_telemetry.h"
#include "hub_locator.h"
#include "motion_detector.h"

namespace {

constexpr uint32_t kHealthMagic = 0x484c5432;  // "HLT2"; change if PersistentHealth changes
constexpr uint32_t kLogEntries = 32;
constexpr size_t kLogTextBytes = 84;
constexpr size_t kBacktraceDepth = 16;
constexpr uint32_t kHeartbeatIntervalMs = 15000;
constexpr uint32_t kFirstHeartbeatDelayMs = 2000;
// Reboot if Wi-Fi has been down this long. Long enough to ride out a router
// restart, short enough that a camera stuck in a bad radio state recovers.
constexpr uint32_t kWifiDownRebootMs = 10UL * 60UL * 1000UL;
constexpr uint32_t kDisconnectLogIntervalMs = 60000;
constexpr uint32_t kHealthTaskStackBytes = 8192;  // HTTP + JSON + the hub name lookup

struct LogEntry {
  uint32_t sequence;
  uint32_t boot;
  uint32_t uptimeMs;
  char text[kLogTextBytes];
};

struct CrashRecord {
  uint32_t valid;
  uint32_t boot;        // boot number that crashed; 0 if it happened before the last power-on
  uint32_t pc;
  uint32_t cause;
  uint32_t vaddr;
  uint32_t depth;
  uint32_t corrupted;
  uint32_t backtrace[kBacktraceDepth];
  char task[16];
  char elfSha256[17];
};

// Lives in RTC slow memory: kept across panics, watchdog and software resets,
// lost on power-off. Validated with a magic number on every boot.
struct PersistentHealth {
  uint32_t magic;
  uint32_t bootCount;
  uint32_t panicCount;
  uint32_t watchdogCount;
  uint32_t brownoutCount;
  uint32_t nextSequence;
  uint32_t head;
  uint32_t count;
  LogEntry entries[kLogEntries];
  CrashRecord lastCrash;
};

RTC_NOINIT_ATTR PersistentHealth persistent;

portMUX_TYPE logMux = portMUX_INITIALIZER_UNLOCKED;
bool healthReady = false;
esp_reset_reason_t bootResetReason = ESP_RST_UNKNOWN;
char firmwareElfSha256[17] = "";

volatile uint32_t wifiDisconnectCount = 0;
volatile uint8_t lastWifiDisconnectReason = 0;
volatile uint32_t lastDisconnectLogMs = 0;
volatile uint32_t disconnectsSinceLastLog = 0;

uint32_t heartbeatsOk = 0;
uint32_t heartbeatsFailed = 0;
int lastHeartbeatHttpCode = 0;
bool lastHeartbeatOk = true;

String cameraModel = "unknown";
TaskHandle_t healthTaskHandle = nullptr;

const char *resetReasonName(esp_reset_reason_t reason) {
  switch (reason) {
    case ESP_RST_POWERON: return "power_on";
    case ESP_RST_EXT: return "external_pin";
    case ESP_RST_SW: return "software";
    case ESP_RST_PANIC: return "crash";
    case ESP_RST_INT_WDT: return "interrupt_watchdog";
    case ESP_RST_TASK_WDT: return "task_watchdog";
    case ESP_RST_WDT: return "other_watchdog";
    case ESP_RST_DEEPSLEEP: return "deep_sleep";
    case ESP_RST_BROWNOUT: return "brownout";
    case ESP_RST_SDIO: return "sdio";
    default: return "unknown";
  }
}

const char *exceptionCauseName(uint32_t cause) {
  switch (cause) {
    case 0: return "IllegalInstruction";
    case 2: return "InstructionFetchError";
    case 3: return "LoadStoreError";
    case 6: return "IntegerDivideByZero";
    case 9: return "LoadStoreAlignment";
    case 20: return "InstrFetchProhibited";
    case 28: return "LoadProhibited";
    case 29: return "StoreProhibited";
    default: return "other";
  }
}

bool persistentLooksValid() {
  return persistent.magic == kHealthMagic && persistent.head < kLogEntries &&
         persistent.count <= kLogEntries && persistent.lastCrash.depth <= kBacktraceDepth;
}

void resetPersistent() {
  memset(&persistent, 0, sizeof(persistent));
  persistent.magic = kHealthMagic;
}

void readCoreDump() {
  if (esp_core_dump_image_check() != ESP_OK) return;

  esp_core_dump_summary_t *summary =
      static_cast<esp_core_dump_summary_t *>(malloc(sizeof(esp_core_dump_summary_t)));
  if (summary == nullptr) return;

  if (esp_core_dump_get_summary(summary) == ESP_OK) {
    CrashRecord &crash = persistent.lastCrash;
    memset(&crash, 0, sizeof(crash));
    crash.valid = 1;
    // A dump found right after power-on is from before the power cycle.
    crash.boot = bootResetReason == ESP_RST_POWERON ? 0 : persistent.bootCount - 1;
    crash.pc = summary->exc_pc;
    crash.cause = summary->ex_info.exc_cause;
    crash.vaddr = summary->ex_info.exc_vaddr;
    crash.depth = min<uint32_t>(summary->exc_bt_info.depth, kBacktraceDepth);
    crash.corrupted = summary->exc_bt_info.corrupted ? 1 : 0;
    for (uint32_t i = 0; i < crash.depth; ++i) crash.backtrace[i] = summary->exc_bt_info.bt[i];
    strncpy(crash.task, summary->exc_task, sizeof(crash.task) - 1);
    memcpy(crash.elfSha256, summary->app_elf_sha256,
           min(sizeof(crash.elfSha256) - 1, sizeof(summary->app_elf_sha256)));
    crash.elfSha256[sizeof(crash.elfSha256) - 1] = '\0';
    logEvent("CRASH%s: task %s, %s at 0x%08lx", crash.boot == 0 ? " (before power-on)" : "",
             crash.task, exceptionCauseName(crash.cause), static_cast<unsigned long>(crash.pc));
  }
  free(summary);
  // Erase so the same dump is not reported again after the next reboot. The
  // summary stays in RTC memory and keeps being reported until a newer crash.
  esp_core_dump_image_erase();
}

void onWifiEvent(arduino_event_id_t event, arduino_event_info_t info) {
  if (event == ARDUINO_EVENT_WIFI_STA_GOT_IP) {
    logEvent("wifi: connected, ip %s, rssi %d dBm", WiFi.localIP().toString().c_str(),
             WiFi.RSSI());
    disconnectsSinceLastLog = 0;
    lastDisconnectLogMs = 0;
  } else if (event == ARDUINO_EVENT_WIFI_STA_DISCONNECTED) {
    ++wifiDisconnectCount;
    ++disconnectsSinceLastLog;
    lastWifiDisconnectReason = info.wifi_sta_disconnected.reason;
    // Auto-reconnect fires one of these per failed attempt; don't let them flood
    // the 32-line log. Log the first, then at most one summary per minute.
    const uint32_t now = millis();
    if (lastDisconnectLogMs == 0 || now - lastDisconnectLogMs >= kDisconnectLogIntervalMs) {
      logEvent("wifi: disconnected, reason %u %s (%lu since last note)",
               lastWifiDisconnectReason,
               WiFi.disconnectReasonName(static_cast<wifi_err_reason_t>(lastWifiDisconnectReason)),
               static_cast<unsigned long>(disconnectsSinceLastLog));
      lastDisconnectLogMs = now == 0 ? 1 : now;
      disconnectsSinceLastLog = 0;
    }
  }
}

void sendHeartbeat() {
  const String body = deviceHealthJson();
  // Find the hub by name first; a failed heartbeat triggers an earlier re-lookup.
  refreshHubAddress(!lastHeartbeatOk);
  char url[64];
  formatHubUrl(url, sizeof(url), "/api/heartbeat");

  HTTPClient http;
  http.setConnectTimeout(2000);
  http.setTimeout(3000);
  if (!http.begin(url)) {
    ++heartbeatsFailed;
    lastHeartbeatHttpCode = -1;
    return;
  }
  http.addHeader("Content-Type", "application/json");
  const int code = http.POST(body);
  lastHeartbeatHttpCode = code;

  bool rebootRequested = false;
  if (code == 200) {
    ++heartbeatsOk;
    if (!lastHeartbeatOk) logEvent("heartbeat: server reachable again");
    lastHeartbeatOk = true;
    JsonDocument response;
    if (!deserializeJson(response, http.getString())) {
      rebootRequested = response["reboot"] | false;
      // No internet at a field site: take the time from the hub instead.
      applyHubTime(response["server_time_unix_ms"].as<long long>());
    }
  } else {
    ++heartbeatsFailed;
    if (lastHeartbeatOk) {
      logEvent("heartbeat: %s failed (%d %s)", url, code,
               code < 0 ? http.errorToString(code).c_str() : "http");
    }
    lastHeartbeatOk = false;
  }
  http.end();

  if (rebootRequested) rebootDevice("requested by server");
}

void healthTask(void *parameter) {
  (void)parameter;
  uint32_t lastHeartbeatMs = 0;
  uint32_t wifiDownSinceMs = 0;
  const uint32_t startedMs = millis();

  for (;;) {
    vTaskDelay(pdMS_TO_TICKS(1000));
    const uint32_t now = millis();
    const bool wifiUp = WiFi.status() == WL_CONNECTED;

    if (wifiUp) {
      wifiDownSinceMs = 0;
    } else if (wifiDownSinceMs == 0) {
      wifiDownSinceMs = now == 0 ? 1 : now;
    } else if (now - wifiDownSinceMs >= kWifiDownRebootMs) {
      rebootDevice("wifi down for 10 minutes");
    }

    const bool due = lastHeartbeatMs == 0 ? now - startedMs >= kFirstHeartbeatDelayMs
                                          : now - lastHeartbeatMs >= kHeartbeatIntervalMs;
    if (wifiUp && due) {
      lastHeartbeatMs = now == 0 ? 1 : now;
      sendHeartbeat();
    }
  }
}

}  // namespace

void logEvent(const char *format, ...) {
  char text[160];
  va_list args;
  va_start(args, format);
  vsnprintf(text, sizeof(text), format, args);
  va_end(args);

  const uint32_t uptimeMs = millis();
  Serial.printf("[%lu.%03lu] %s\n", static_cast<unsigned long>(uptimeMs / 1000),
                static_cast<unsigned long>(uptimeMs % 1000), text);
  if (!healthReady) return;

  portENTER_CRITICAL(&logMux);
  LogEntry &entry = persistent.entries[persistent.head];
  entry.sequence = persistent.nextSequence++;
  entry.boot = persistent.bootCount;
  entry.uptimeMs = uptimeMs;
  strncpy(entry.text, text, kLogTextBytes - 1);
  entry.text[kLogTextBytes - 1] = '\0';
  persistent.head = (persistent.head + 1) % kLogEntries;
  if (persistent.count < kLogEntries) ++persistent.count;
  portEXIT_CRITICAL(&logMux);
}

void initializeDeviceHealth() {
  bootResetReason = esp_reset_reason();
  if (bootResetReason == ESP_RST_POWERON || !persistentLooksValid()) resetPersistent();

  ++persistent.bootCount;
  switch (bootResetReason) {
    case ESP_RST_PANIC: ++persistent.panicCount; break;
    case ESP_RST_INT_WDT:
    case ESP_RST_TASK_WDT:
    case ESP_RST_WDT: ++persistent.watchdogCount; break;
    case ESP_RST_BROWNOUT: ++persistent.brownoutCount; break;
    default: break;
  }
  esp_ota_get_app_elf_sha256(firmwareElfSha256, sizeof(firmwareElfSha256));
  healthReady = true;

  logEvent("boot #%lu, reset reason: %s, firmware %s",
           static_cast<unsigned long>(persistent.bootCount), resetReasonName(bootResetReason),
           firmwareElfSha256);
  readCoreDump();
  WiFi.onEvent(onWifiEvent);
}

void startDeviceHealthMonitor() {
  if (healthTaskHandle != nullptr) return;
  uint16_t pid = 0;
  uint8_t version = 0;
  if (getCameraSensorIdentity(pid, version)) cameraModel = cameraSensorModelName(pid, version);

  // Core 0 next to the Wi-Fi stack, low priority: it only does network I/O.
  if (xTaskCreatePinnedToCore(healthTask, "health", kHealthTaskStackBytes, nullptr, 1,
                              &healthTaskHandle, 0) != pdPASS) {
    healthTaskHandle = nullptr;
    logEvent("health: could not start heartbeat task");
  }
}

String deviceHealthJson() {
  JsonDocument doc;
  doc["camera_id"] = getCameraId();
  doc["boot_id"] = getBootId();
  doc["firmware_elf_sha256"] = firmwareElfSha256;
  doc["camera_model"] = cameraModel;
  doc["ip"] = WiFi.localIP().toString();
  doc["uptime_s"] = static_cast<uint32_t>(esp_timer_get_time() / 1000000);
  doc["time_source"] = deviceTimeSource();

  // Fixed facts about the board, for the hub's Specs view. All are cheap reads.
  JsonObject hardware = doc["hardware"].to<JsonObject>();
#ifdef ARDUINO_BOARD
  hardware["board"] = ARDUINO_BOARD;
#endif
  hardware["chip_model"] = ESP.getChipModel();
  hardware["chip_revision"] = ESP.getChipRevision();
  hardware["cpu_cores"] = ESP.getChipCores();
  hardware["cpu_mhz"] = ESP.getCpuFreqMHz();
  hardware["flash_bytes"] = ESP.getFlashChipSize();
  hardware["psram_bytes"] = ESP.getPsramSize();
  hardware["mac"] = WiFi.macAddress();
  hardware["sdk"] = ESP.getSdkVersion();
  char arduinoCore[16];
  snprintf(arduinoCore, sizeof(arduinoCore), "%d.%d.%d", ESP_ARDUINO_VERSION_MAJOR,
           ESP_ARDUINO_VERSION_MINOR, ESP_ARDUINO_VERSION_PATCH);
  hardware["arduino_core"] = arduinoCore;
  hardware["firmware_built"] = __DATE__ " " __TIME__;

  JsonObject boot = doc["boot"].to<JsonObject>();
  boot["number_since_power_on"] = persistent.bootCount;
  boot["reset_reason"] = resetReasonName(bootResetReason);
  boot["crashes_since_power_on"] = persistent.panicCount;
  boot["watchdog_resets_since_power_on"] = persistent.watchdogCount;
  boot["brownouts_since_power_on"] = persistent.brownoutCount;

  JsonObject wifi = doc["wifi"].to<JsonObject>();
  wifi["connected"] = WiFi.status() == WL_CONNECTED;
  wifi["rssi_dbm"] = WiFi.RSSI();
  wifi["disconnects_this_boot"] = wifiDisconnectCount;
  wifi["last_disconnect_reason"] = lastWifiDisconnectReason;
  wifi["last_disconnect_reason_name"] =
      lastWifiDisconnectReason == 0
          ? ""
          : WiFi.disconnectReasonName(static_cast<wifi_err_reason_t>(lastWifiDisconnectReason));
  wifi["heartbeats_ok"] = heartbeatsOk;
  wifi["heartbeats_failed"] = heartbeatsFailed;

  JsonObject hub = doc["hub"].to<JsonObject>();
  char hubAddress[16];
  getHubAddress(hubAddress, sizeof(hubAddress));
  hub["host"] = cfg.serverHost;
  hub["address"] = hubAddress;
  hub["found_by_name"] = isHubResolvedByName();

  const DeviceTelemetrySnapshot device = sampleDeviceTelemetry();
  JsonObject memory = doc["memory"].to<JsonObject>();
  memory["free_heap"] = device.freeHeapBytes;
  memory["min_free_heap"] = device.minimumFreeHeapBytes;
  memory["largest_heap_block"] = device.largestFreeHeapBlockBytes;
  memory["free_psram"] = device.psramFreeBytes;
  memory["min_free_psram"] = device.psramMinimumFreeBytes;
  doc["chip_temperature_c"] = device.chipTemperatureC;

  JsonObject camera = doc["camera"].to<JsonObject>();
  camera["mode"] = cameraOperatingModeName(getCameraOperatingMode());
  camera["state"] = cameraRuntimeStateName(getCameraRuntimeState());
  camera["motion_enabled"] = cfg.motionEnabled;
  camera["powered"] = isCameraPowered();
  camera["last_decision"] = getMotionLastDecision();
  camera["motion_events"] = getMotionEventCount();
  camera["capture_failures"] = getMotionCaptureFailureCount();
  camera["consecutive_capture_failures"] = getConsecutiveMotionCaptureFailures();
  camera["last_upload_ok"] = wasLastMotionUploadSuccessful();
  const uint32_t lastMotionMs = getLastMotionMs();
  if (lastMotionMs == 0) {
    camera["seconds_since_motion"] = nullptr;
  } else {
    camera["seconds_since_motion"] = (millis() - lastMotionMs) / 1000;
  }
  const String error = getCameraRuntimeError();
  if (error.length() > 0) {
    camera["last_error"] = error;
  } else {
    camera["last_error"] = nullptr;
  }

  if (persistent.lastCrash.valid == 1) {
    const CrashRecord &crash = persistent.lastCrash;
    JsonObject crashJson = doc["last_crash"].to<JsonObject>();
    char hex[11];
    crashJson["boot"] = crash.boot;
    crashJson["before_power_on"] = crash.boot == 0;
    char task[sizeof(crash.task) + 1] = {};
    memcpy(task, crash.task, sizeof(crash.task));
    crashJson["task"] = task;
    snprintf(hex, sizeof(hex), "0x%08lx", static_cast<unsigned long>(crash.pc));
    crashJson["pc"] = hex;
    crashJson["cause"] = crash.cause;
    crashJson["cause_name"] = exceptionCauseName(crash.cause);
    snprintf(hex, sizeof(hex), "0x%08lx", static_cast<unsigned long>(crash.vaddr));
    crashJson["fault_address"] = hex;
    crashJson["backtrace_corrupted"] = crash.corrupted == 1;
    char sha[sizeof(crash.elfSha256) + 1] = {};
    memcpy(sha, crash.elfSha256, sizeof(crash.elfSha256));
    crashJson["firmware_elf_sha256"] = sha;
    JsonArray backtrace = crashJson["backtrace"].to<JsonArray>();
    for (uint32_t i = 0; i < crash.depth && i < kBacktraceDepth; ++i) {
      snprintf(hex, sizeof(hex), "0x%08lx", static_cast<unsigned long>(crash.backtrace[i]));
      backtrace.add(hex);
    }
  } else {
    doc["last_crash"] = nullptr;
  }

  // Oldest first. Copy one entry at a time so the critical section stays short
  // and no 3 KB buffer lands on this task's stack.
  JsonArray events = doc["events"].to<JsonArray>();
  portENTER_CRITICAL(&logMux);
  const uint32_t count = persistent.count;
  const uint32_t first = (persistent.head + kLogEntries - count) % kLogEntries;
  portEXIT_CRITICAL(&logMux);
  for (uint32_t i = 0; i < count; ++i) {
    LogEntry entry;
    portENTER_CRITICAL(&logMux);
    entry = persistent.entries[(first + i) % kLogEntries];
    portEXIT_CRITICAL(&logMux);
    entry.text[kLogTextBytes - 1] = '\0';
    JsonObject item = events.add<JsonObject>();
    item["seq"] = entry.sequence;
    item["boot"] = entry.boot;
    item["t_ms"] = entry.uptimeMs;
    item["text"] = entry.text;
  }

  String json;
  serializeJson(doc, json);
  return json;
}

void rebootDevice(const char *reason) {
  logEvent("rebooting: %s", reason);
  Serial.flush();
  delay(200);
  ESP.restart();
  while (true) {
  }
}
