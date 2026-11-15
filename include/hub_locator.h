#pragma once

#include <Arduino.h>

// Finds the hub (the laptop or Raspberry Pi server) by name so no camera needs
// the hub's IP address typed into its config.
//
// - config.json "server_host" (default "birdhub.local") is looked up on the
//   local network with mDNS. A plain IPv4 address or an ordinary DNS name is
//   also accepted there.
// - The lookup only ever runs in the health task, just before a heartbeat.
//   Uploads never wait on it: they use the last address that resolved.
// - Until the name has resolved once (or if "server_host" is empty), the
//   classic "server_ip" setting is used, so existing cameras keep working.
// - As a side effect the camera itself answers to <camera_id>.local.

// Health task only. Blocks for up to ~2 s when a lookup is due. Pass
// lastContactFailed = true after a failed heartbeat to look the hub up again
// sooner (it may have been given a new address).
void refreshHubAddress(bool lastContactFailed);

// Any task, never blocks. Writes "http://<address>:<server_port><path>".
void formatHubUrl(char *url, size_t urlBytes, const char *path);

// Any task, never blocks. The address uploads and heartbeats currently go to.
void getHubAddress(char *address, size_t addressBytes);

// True when that address came from resolving "server_host" rather than "server_ip".
bool isHubResolvedByName();
