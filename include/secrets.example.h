#pragma once

// Optional compile-time fallback used when config.json does not provide Wi-Fi
// credentials. Copy this file to secrets.h and replace both placeholder values.
// Never commit secrets.h.
constexpr char WIFI_SSID[] = "YOUR_2_4_GHZ_WIFI_NAME";
constexpr char WIFI_PASSWORD[] = "YOUR_WIFI_PASSWORD";