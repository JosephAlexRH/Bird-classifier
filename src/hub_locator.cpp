#include "hub_locator.h"

#include <ESPmDNS.h>
#include <WiFi.h>
#include <freertos/FreeRTOS.h>
#include <strings.h>

#include "app_config.h"
#include "device_health.h"
#include "device_telemetry.h"

namespace {

constexpr uint32_t kMdnsQueryTimeoutMs = 2000;
// A resolved address is trusted for this long while heartbeats keep succeeding.
constexpr uint32_t kRefreshIntervalMs = 5UL * 60UL * 1000UL;
// Minimum gap between lookups when the hub is missing or not answering.
constexpr uint32_t kRetryIntervalMs = 30000;

portMUX_TYPE hubMux = portMUX_INITIALIZER_UNLOCKED;
char resolvedAddress[16] = "";  // empty until server_host resolves

// Health-task-only state.
char resolvedForHost[sizeof(cfg.serverHost)] = "";
bool lookupAttempted = false;
bool fallbackLogged = false;
bool mdnsStarted = false;
bool mdnsFailureLogged = false;
uint32_t lastLookupMs = 0;

void storeResolvedAddress(const char *address) {
  portENTER_CRITICAL(&hubMux);
  strlcpy(resolvedAddress, address, sizeof(resolvedAddress));
  portEXIT_CRITICAL(&hubMux);
}

void loadResolvedAddress(char *address, size_t addressBytes) {
  portENTER_CRITICAL(&hubMux);
  strlcpy(address, resolvedAddress, addressBytes);
  portEXIT_CRITICAL(&hubMux);
}

bool endsWithDotLocal(const char *host) {
  const size_t length = strlen(host);
  return length > 6 && strcasecmp(host + length - 6, ".local") == 0;
}

bool startMdns() {
  if (mdnsStarted) return true;
  // The camera ID is already a valid host label: esp32cam-<12 hex digits>.
  mdnsStarted = MDNS.begin(getCameraId().c_str());
  if (!mdnsStarted && !mdnsFailureLogged) {
    logEvent("hub: could not start mDNS; using server_ip");
    mdnsFailureLogged = true;
  }
  return mdnsStarted;
}

bool lookUp(const char *host, IPAddress &address) {
  if (address.fromString(host)) return true;  // already an IPv4 address
  if (endsWithDotLocal(host)) {
    if (!startMdns()) return false;
    String label(host);
    label.remove(label.length() - 6);
    address = MDNS.queryHost(label, kMdnsQueryTimeoutMs);
    return static_cast<uint32_t>(address) != 0;  // 0.0.0.0 means nobody answered
  }
  return WiFi.hostByName(host, address) == 1;
}

}  // namespace

void refreshHubAddress(bool lastContactFailed) {
  char host[sizeof(cfg.serverHost)];
  strlcpy(host, cfg.serverHost, sizeof(host));

  const bool hostChanged = strcmp(host, resolvedForHost) != 0;
  if (hostChanged) {
    // A different name: forget the old hub straight away.
    strlcpy(resolvedForHost, host, sizeof(resolvedForHost));
    storeResolvedAddress("");
    lookupAttempted = false;
    fallbackLogged = false;
  }
  if (host[0] == '\0') return;  // name lookup switched off; server_ip is used

  char current[16];
  loadResolvedAddress(current, sizeof(current));
  const uint32_t now = millis();
  if (lookupAttempted) {
    const uint32_t age = now - lastLookupMs;
    const bool healthy = current[0] != '\0' && !lastContactFailed;
    if (age < (healthy ? kRefreshIntervalMs : kRetryIntervalMs)) return;
  }
  lookupAttempted = true;
  lastLookupMs = now;

  IPAddress address;
  if (lookUp(host, address)) {
    const String text = address.toString();
    if (strcmp(current, text.c_str()) != 0) {
      logEvent("hub: %s is at %s", host, text.c_str());
      storeResolvedAddress(text.c_str());
    }
    fallbackLogged = false;
    return;
  }

  // Keep the last address that worked; the hub may only be restarting.
  if (current[0] == '\0' && !fallbackLogged) {
    logEvent("hub: %s not found; using server_ip %s", host, cfg.serverIp);
    fallbackLogged = true;
  }
}

void getHubAddress(char *address, size_t addressBytes) {
  loadResolvedAddress(address, addressBytes);
  if (address[0] == '\0') strlcpy(address, cfg.serverIp, addressBytes);
}

bool isHubResolvedByName() {
  char address[16];
  loadResolvedAddress(address, sizeof(address));
  return address[0] != '\0';
}

void formatHubUrl(char *url, size_t urlBytes, const char *path) {
  char address[16];
  getHubAddress(address, sizeof(address));
  snprintf(url, urlBytes, "http://%s:%d%s", address, cfg.serverPort, path);
}
