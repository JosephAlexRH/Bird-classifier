#include "device_telemetry.h"

#include <esp_random.h>
#include <esp_sntp.h>
#include <esp_timer.h>
#include <esp32-hal.h>
#include <sys/time.h>
#include <time.h>

#include "device_health.h"

namespace {

String bootId;
String cameraId;

constexpr int64_t kEarliestPlausibleUnixMs = 1700000000000LL;  // late 2023
constexpr int64_t kHubTimeToleranceMs = 2000;
volatile bool ntpSynchronized = false;
volatile bool hubTimeApplied = false;

void onSntpSynchronized(struct timeval *) { ntpSynchronized = true; }

String formatUuid(const uint8_t bytes[16]) {
  char value[37];
  snprintf(value, sizeof(value),
           "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
           bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
           bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]);
  return String(value);
}

String utcTimestamp(const timeval &now) {
  tm utc = {};
  gmtime_r(&now.tv_sec, &utc);
  char date[24];
  strftime(date, sizeof(date), "%Y-%m-%dT%H:%M:%S", &utc);
  char result[36];
  snprintf(result, sizeof(result), "%s.%06ldZ", date, static_cast<long>(now.tv_usec));
  return String(result);
}

}  // namespace

void initializeDeviceTelemetry() {
  const uint64_t efuseMac = ESP.getEfuseMac() & 0x0000ffffffffffffULL;
  char stableId[26];
  snprintf(stableId, sizeof(stableId), "esp32cam-%04x%08x",
           static_cast<unsigned int>((efuseMac >> 32) & 0xffff),
           static_cast<unsigned int>(efuseMac & 0xffffffff));
  cameraId = stableId;

  uint8_t randomBytes[16];
  esp_fill_random(randomBytes, sizeof(randomBytes));
  randomBytes[6] = static_cast<uint8_t>((randomBytes[6] & 0x0f) | 0x40);
  randomBytes[8] = static_cast<uint8_t>((randomBytes[8] & 0x3f) | 0x80);
  bootId = formatUuid(randomBytes);

  // SNTP runs asynchronously; metadata never blocks waiting for synchronization.
  // With no internet it never completes and the hub's time is used instead.
  sntp_set_time_sync_notification_cb(onSntpSynchronized);
  configTime(0, 0, "pool.ntp.org", "time.google.com", "time.nist.gov");
}

DeviceTelemetrySnapshot sampleDeviceTelemetry() {
  DeviceTelemetrySnapshot snapshot = {};
  snapshot.cameraId = cameraId;
  snapshot.bootId = bootId;

  snapshot.chipTemperatureC = temperatureRead();
  snapshot.freeHeapBytes = ESP.getFreeHeap();
  snapshot.minimumFreeHeapBytes = ESP.getMinFreeHeap();
  snapshot.largestFreeHeapBlockBytes = ESP.getMaxAllocHeap();
  snapshot.psramTotalBytes = ESP.getPsramSize();
  snapshot.psramFreeBytes = ESP.getFreePsram();
  snapshot.psramMinimumFreeBytes = ESP.getMinFreePsram();
  snapshot.psramLargestFreeBlockBytes = ESP.getMaxAllocPsram();

  timeval now = {};
  snapshot.sampledAtBootUs = esp_timer_get_time();
  gettimeofday(&now, nullptr);
  tm utc = {};
  gmtime_r(&now.tv_sec, &utc);
  snapshot.timeSynchronized = utc.tm_year > (2016 - 1900);
  if (snapshot.timeSynchronized) snapshot.deviceTimeUtc = utcTimestamp(now);
  return snapshot;
}

void applyHubTime(int64_t hubUnixMs) {
  if (ntpSynchronized || hubUnixMs < kEarliestPlausibleUnixMs) return;

  timeval now = {};
  gettimeofday(&now, nullptr);
  const int64_t localMs = static_cast<int64_t>(now.tv_sec) * 1000 + now.tv_usec / 1000;
  const int64_t differenceMs = hubUnixMs - localMs;
  if (hubTimeApplied && differenceMs > -kHubTimeToleranceMs && differenceMs < kHubTimeToleranceMs) {
    return;
  }

  timeval hub = {};
  hub.tv_sec = static_cast<time_t>(hubUnixMs / 1000);
  hub.tv_usec = static_cast<suseconds_t>((hubUnixMs % 1000) * 1000);
  settimeofday(&hub, nullptr);
  if (hubTimeApplied) {
    logEvent("time: corrected from hub by %ld s", static_cast<long>(differenceMs / 1000));
  } else {
    logEvent("time: set from hub (no internet time)");
  }
  hubTimeApplied = true;
}

const char *deviceTimeSource() {
  if (ntpSynchronized) return "ntp";
  return hubTimeApplied ? "hub" : "none";
}

const String &getBootId() { return bootId; }

const String &getCameraId() { return cameraId; }