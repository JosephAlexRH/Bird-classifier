#include "camera_service.h"

#include <Arduino.h>
#include <ArduinoJson.h>
#include <esp_arduino_version.h>
#include <esp_camera.h>
#include <esp_timer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <mbedtls/base64.h>
#include <mbedtls/sha256.h>
#include <memory>
#include <new>

#include "app_config.h"
#include "device_telemetry.h"

static_assert(ESP_ARDUINO_VERSION == ESP_ARDUINO_VERSION_VAL(2, 0, 17),
              "Camera metadata enum/provenance mapping requires Arduino-ESP32 2.0.17");

namespace {

constexpr int kPowerDownPin = 32;
constexpr int kResetPin = -1;
constexpr int kXclkPin = 0;
constexpr int kSiodPin = 26;
constexpr int kSiocPin = 27;
constexpr int kY9Pin = 35;
constexpr int kY8Pin = 34;
constexpr int kY7Pin = 39;
constexpr int kY6Pin = 36;
constexpr int kY5Pin = 21;
constexpr int kY4Pin = 19;
constexpr int kY3Pin = 18;
constexpr int kY2Pin = 5;
constexpr int kVsyncPin = 25;
constexpr int kHrefPin = 23;
constexpr int kPclkPin = 22;
constexpr uint8_t kMaxExposureConsistencyAttempts = 3;
constexpr int kOv2640FrameBufferCount = 1;
constexpr camera_grab_mode_t kOv2640GrabMode = CAMERA_GRAB_WHEN_EMPTY;
constexpr int kOv5640FrameBufferCount = 2;
constexpr camera_grab_mode_t kOv5640GrabMode = CAMERA_GRAB_LATEST;
constexpr int kProbeXclkHz = 20000000;
constexpr uint32_t kOv2640AecDriverLimitLines = 1200;
constexpr size_t kOv2640RegisterCountPerBank = 255;
constexpr size_t kOv2640RegisterValidityBytes = 32;
constexpr size_t kOv2640RegisterDumpBytes =
    4 + 2 * (kOv2640RegisterCountPerBank + kOv2640RegisterValidityBytes);
constexpr uint16_t kOv5640DumpAddresses[] = {
    0x300a, 0x300b, 0x3400, 0x3401, 0x3402, 0x3403, 0x3404, 0x3405, 0x3406,
    0x3500, 0x3501, 0x3502, 0x350a, 0x350b, 0x380e, 0x380f,
};
constexpr size_t kOv5640RegisterDumpEntryBytes = 4;
constexpr size_t kOv5640RegisterDumpBytes =
    4 + sizeof(kOv5640DumpAddresses) / sizeof(uint16_t) * kOv5640RegisterDumpEntryBytes;
constexpr size_t kMaximumRegisterDumpBytes =
    kOv2640RegisterDumpBytes > kOv5640RegisterDumpBytes ? kOv2640RegisterDumpBytes
                                                        : kOv5640RegisterDumpBytes;

uint64_t framesSinceBoot = 0;
uint32_t settingsRevision = 0;
int64_t lastSettingsChangeUs = 0;
bool settingsChangeTimeValid = false;
uint32_t persistentSettingsRevision = 1;
int initializedFramesize = static_cast<int>(FRAMESIZE_VGA);
int initializedXclkHz = kProbeXclkHz;
int initializedFrameBufferCount = kOv2640FrameBufferCount;
camera_grab_mode_t initializedGrabMode = kOv2640GrabMode;
bool registerDumpPending = true;
bool registerDumpCacheValid = false;
uint8_t registerDumpCache[kMaximumRegisterDumpBytes];
size_t registerDumpCacheBytes = 0;
String registerDumpSha256;
String registerDumpPayloadBase64;
StaticSemaphore_t cameraMutexStorage;
SemaphoreHandle_t cameraMutex = nullptr;

bool lockCamera() {
  if (cameraMutex == nullptr) {
    cameraMutex = xSemaphoreCreateRecursiveMutexStatic(&cameraMutexStorage);
  }
  return cameraMutex != nullptr &&
         xSemaphoreTakeRecursive(cameraMutex, portMAX_DELAY) == pdTRUE;
}

void unlockCamera() {
  if (cameraMutex != nullptr) xSemaphoreGiveRecursive(cameraMutex);
}

class CameraMutexGuard {
 public:
  CameraMutexGuard() : locked_(lockCamera()) {}
  ~CameraMutexGuard() {
    if (locked_) unlockCamera();
  }
  bool locked() const { return locked_; }

 private:
  bool locked_;
};

String bytesToHex(const uint8_t *bytes, size_t length) {
  static const char digits[] = "0123456789abcdef";
  String result;
  result.reserve(length * 2);
  for (size_t index = 0; index < length; ++index) {
    result += digits[bytes[index] >> 4];
    result += digits[bytes[index] & 0x0f];
  }
  return result;
}

const char *grabModeName(int value) {
  switch (static_cast<camera_grab_mode_t>(value)) {
    case CAMERA_GRAB_WHEN_EMPTY: return "CAMERA_GRAB_WHEN_EMPTY";
    case CAMERA_GRAB_LATEST: return "CAMERA_GRAB_LATEST";
  }
  return "UNKNOWN";
}

String encodeRegisterDump() {
  if (registerDumpCacheBytes == 0) return String();
  const size_t capacity = 4 * ((registerDumpCacheBytes + 2) / 3) + 1;
  String encoded;
  if (!encoded.reserve(capacity)) return String();
  std::unique_ptr<unsigned char[]> buffer(new (std::nothrow) unsigned char[capacity]);
  if (!buffer) return String();
  size_t written = 0;
  if (mbedtls_base64_encode(buffer.get(), capacity, &written, registerDumpCache,
                            registerDumpCacheBytes) != 0) {
    return String();
  }
  buffer[written] = '\0';
  encoded = reinterpret_cast<const char *>(buffer.get());
  return encoded;
}

bool refreshRegisterDump(sensor_t *sensor) {
  if (sensor == nullptr || sensor->get_reg == nullptr) return false;
  memset(registerDumpCache, 0, sizeof(registerDumpCache));
  size_t offset = 4;
  if (sensor->id.PID == OV2640_PID) {
    memcpy(registerDumpCache, "OVD1", 4);
    for (int bank = 0; bank <= 1; ++bank) {
      uint8_t *values = registerDumpCache + offset;
      uint8_t *validity = values + kOv2640RegisterCountPerBank;
      for (size_t address = 0; address < kOv2640RegisterCountPerBank; ++address) {
        const int virtualAddress = (bank << 8) | static_cast<int>(address);
        const int value = sensor->get_reg(sensor, virtualAddress, 0xff);
        if (value >= 0) {
          values[address] = static_cast<uint8_t>(value);
          validity[address / 8] |= static_cast<uint8_t>(1U << (address % 8));
        }
      }
      offset += kOv2640RegisterCountPerBank + kOv2640RegisterValidityBytes;
    }
    registerDumpCacheBytes = kOv2640RegisterDumpBytes;
  } else if (sensor->id.PID == OV5640_PID) {
    memcpy(registerDumpCache, "OV5D", 4);
    for (uint16_t address : kOv5640DumpAddresses) {
      const int value = sensor->get_reg(sensor, address, 0xff);
      registerDumpCache[offset++] = static_cast<uint8_t>(address >> 8);
      registerDumpCache[offset++] = static_cast<uint8_t>(address & 0xff);
      registerDumpCache[offset++] = value >= 0 ? 1 : 0;
      registerDumpCache[offset++] = value >= 0 ? static_cast<uint8_t>(value) : 0;
    }
    registerDumpCacheBytes = kOv5640RegisterDumpBytes;
  } else {
    registerDumpCacheBytes = 0;
    return false;
  }

  uint8_t digest[32];
  if (mbedtls_sha256_ret(registerDumpCache, registerDumpCacheBytes, digest, 0) != 0) return false;
  registerDumpSha256 = bytesToHex(digest, sizeof(digest));
  registerDumpPayloadBase64 = encodeRegisterDump();
  if (registerDumpPayloadBase64.length() == 0) return false;
  registerDumpCacheValid = true;
  return true;
}

const char *framesizeName(int value) {
  switch (static_cast<framesize_t>(value)) {
    case FRAMESIZE_96X96: return "96X96";
    case FRAMESIZE_QQVGA: return "QQVGA";
    case FRAMESIZE_QCIF: return "QCIF";
    case FRAMESIZE_HQVGA: return "HQVGA";
    case FRAMESIZE_240X240: return "240X240";
    case FRAMESIZE_QVGA: return "QVGA";
    case FRAMESIZE_CIF: return "CIF";
    case FRAMESIZE_HVGA: return "HVGA";
    case FRAMESIZE_VGA: return "VGA";
    case FRAMESIZE_SVGA: return "SVGA";
    case FRAMESIZE_XGA: return "XGA";
    case FRAMESIZE_HD: return "HD";
    case FRAMESIZE_SXGA: return "SXGA";
    case FRAMESIZE_UXGA: return "UXGA";
    case FRAMESIZE_FHD: return "FHD";
    case FRAMESIZE_P_HD: return "P_HD";
    case FRAMESIZE_P_3MP: return "P_3MP";
    case FRAMESIZE_QXGA: return "QXGA";
    case FRAMESIZE_QHD: return "QHD";
    case FRAMESIZE_WQXGA: return "WQXGA";
    case FRAMESIZE_P_FHD: return "P_FHD";
    case FRAMESIZE_QSXGA: return "QSXGA";
    default: return "UNKNOWN";
  }
}

const char *pixelFormatName(int value) {
  switch (static_cast<pixformat_t>(value)) {
    case PIXFORMAT_RGB565: return "RGB565";
    case PIXFORMAT_YUV422: return "YUV422";
    case PIXFORMAT_YUV420: return "YUV420";
    case PIXFORMAT_GRAYSCALE: return "GRAYSCALE";
    case PIXFORMAT_JPEG: return "JPEG";
    case PIXFORMAT_RGB888: return "RGB888";
    case PIXFORMAT_RAW: return "RAW";
    case PIXFORMAT_RGB444: return "RGB444";
    case PIXFORMAT_RGB555: return "RGB555";
    default: return "UNKNOWN";
  }
}

String hexValue(uint32_t value, uint8_t digits) {
  char buffer[12];
  snprintf(buffer, sizeof(buffer), "0x%0*lX", digits, static_cast<unsigned long>(value));
  return String(buffer);
}

void noteSettingsChange() {
  ++settingsRevision;
  lastSettingsChangeUs = esp_timer_get_time();
  settingsChangeTimeValid = true;
}

int frameBufferCountForSensor(uint16_t sensorKey) {
  return sensorKey == kOv5640SensorKey ? kOv5640FrameBufferCount : kOv2640FrameBufferCount;
}

camera_grab_mode_t grabModeForSensor(uint16_t sensorKey) {
  return sensorKey == kOv5640SensorKey ? kOv5640GrabMode : kOv2640GrabMode;
}

camera_config_t cameraConfigFor(framesize_t framesize, int jpegQuality, int xclkHz,
                                uint16_t sensorKey = 0) {
  camera_config_t config = {};
  config.pin_pwdn = kPowerDownPin;
  config.pin_reset = kResetPin;
  config.pin_xclk = kXclkPin;
  config.pin_sccb_sda = kSiodPin;
  config.pin_sccb_scl = kSiocPin;
  config.pin_d7 = kY9Pin;
  config.pin_d6 = kY8Pin;
  config.pin_d5 = kY7Pin;
  config.pin_d4 = kY6Pin;
  config.pin_d3 = kY5Pin;
  config.pin_d2 = kY4Pin;
  config.pin_d1 = kY3Pin;
  config.pin_d0 = kY2Pin;
  config.pin_vsync = kVsyncPin;
  config.pin_href = kHrefPin;
  config.pin_pclk = kPclkPin;
  config.xclk_freq_hz = xclkHz;
  config.ledc_timer = LEDC_TIMER_0;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = framesize;
  config.jpeg_quality = jpegQuality;
  config.fb_count = frameBufferCountForSensor(sensorKey);
  config.grab_mode = grabModeForSensor(sensorKey);
  return config;
}

bool reinitializeCameraForProfile(uint16_t expectedSensorKey,
                                  const CameraSensorProfile &profile) {
  CameraMutexGuard cameraLock;
  if (!cameraLock.locked()) return false;
  camera_config_t cameraConfig =
      cameraConfigFor(static_cast<framesize_t>(profile.framesize), profile.jpegQuality,
                      profile.xclkHz, expectedSensorKey);
  esp_camera_deinit();
  delay(50);
  const esp_err_t error = esp_camera_init(&cameraConfig);
  if (error != ESP_OK) {
    Serial.printf("Camera profile initialization failed: 0x%x\n", error);
    return false;
  }

  sensor_t *sensor = esp_camera_sensor_get();
  if (sensor == nullptr ||
      cameraSensorConfigKey(sensor->id.PID, sensor->id.VER) != expectedSensorKey) {
    Serial.println("Camera identity changed after applying the selected profile");
    esp_camera_deinit();
    return false;
  }

  initializedFramesize = profile.framesize;
  initializedXclkHz = profile.xclkHz;
  initializedFrameBufferCount = cameraConfig.fb_count;
  initializedGrabMode = cameraConfig.grab_mode;
  noteSettingsChange();
  return true;
}

bool setJpegMode(sensor_t *sensor, framesize_t framesize, int quality, bool &changed) {
  changed = false;
  if (sensor->pixformat != PIXFORMAT_JPEG) {
    if (sensor->set_pixformat(sensor, PIXFORMAT_JPEG) != 0) return false;
    changed = true;
  }
  if (sensor->status.framesize != framesize) {
    if (sensor->set_framesize(sensor, framesize) != 0) {
      if (changed) noteSettingsChange();
      return false;
    }
    changed = true;
  }
  if (sensor->status.quality != quality) {
    if (sensor->set_quality(sensor, quality) != 0) {
      if (changed) noteSettingsChange();
      return false;
    }
    changed = true;
  }
  if (changed) noteSettingsChange();
  return true;
}

bool setConfiguredJpegMode(sensor_t *sensor, bool &changed) {
  return setJpegMode(sensor, static_cast<framesize_t>(cfg.framesize), cfg.jpegQuality, changed);
}

CameraSettingsSnapshot copyDriverCache(const sensor_t *sensor) {
  const camera_status_t &status = sensor->status;
  return {
      static_cast<int>(status.framesize), static_cast<int>(sensor->pixformat), status.quality,
      status.brightness, status.contrast, status.saturation, status.sharpness, status.denoise,
      status.special_effect, status.wb_mode, status.awb, status.awb_gain, status.aec, status.aec2,
      status.ae_level, status.aec_value, status.agc, status.agc_gain, status.gainceiling, status.bpc,
      status.wpc, status.raw_gma, status.lenc, status.hmirror, status.vflip, status.dcw,
      status.colorbar, status.scale, status.binning};
}

bool readByteRegister(sensor_t *sensor, int address, int mask, uint8_t &value) {
  const int result = sensor->get_reg(sensor, address, mask);
  if (result < 0) return false;
  value = static_cast<uint8_t>(result);
  return true;
}

bool readOv2640Exposure(sensor_t *sensor, uint8_t &high, uint8_t &middle, uint8_t &low) {
  return readByteRegister(sensor, 0x145, 0x3f, high) &&
         readByteRegister(sensor, 0x110, 0xff, middle) &&
         readByteRegister(sensor, 0x104, 0x03, low);
}

Ov2640RegisterSnapshot readObservedOv2640(sensor_t *sensor) {
  Ov2640RegisterSnapshot result = {};
  result.supported = sensor != nullptr && sensor->id.PID == OV2640_PID;
  if (!result.supported || sensor->get_reg == nullptr) return result;

  for (uint8_t attempt = 1; attempt <= kMaxExposureConsistencyAttempts; ++attempt) {
    uint8_t firstHigh = 0;
    uint8_t firstMiddle = 0;
    uint8_t firstLow = 0;
    uint8_t secondHigh = 0;
    uint8_t secondMiddle = 0;
    uint8_t secondLow = 0;
    result.exposureReadAttempts = attempt;
    if (!readOv2640Exposure(sensor, firstHigh, firstMiddle, firstLow) ||
        !readOv2640Exposure(sensor, secondHigh, secondMiddle, secondLow)) {
      break;
    }
    result.exposureValid = true;
    result.aecHigh = secondHigh;
    result.aecMiddle = secondMiddle;
    result.aecLow = secondLow;
    if (firstHigh == secondHigh && firstMiddle == secondMiddle && firstLow == secondLow) {
      result.exposureConsistent = true;
      break;
    }
  }
  if (result.exposureValid) {
    result.exposureLines = (static_cast<uint32_t>(result.aecHigh) << 10) |
                           (static_cast<uint32_t>(result.aecMiddle) << 2) | result.aecLow;
  }
  result.aecDriverLimitLines = kOv2640AecDriverLimitLines;
  result.aeSaturatedValid =
      result.exposureValid && result.exposureConsistent && sensor->status.aec != 0;
  result.aeSaturated =
      result.aeSaturatedValid && result.exposureLines >= kOv2640AecDriverLimitLines;

  result.gainValid = readByteRegister(sensor, 0x100, 0xff, result.gainRaw);
  if (result.gainValid) {
    uint8_t highNibble = result.gainRaw >> 4;
    uint8_t setBits = 0;
    while (highNibble != 0) {
      setBits += highNibble & 1;
      highNibble >>= 1;
    }
    result.nominalGainMultiplier =
        static_cast<float>(1U << setBits) * (1.0f + (result.gainRaw & 0x0f) / 16.0f);
  }
  result.exposureGainProductValid =
      result.exposureValid && result.exposureConsistent && result.gainValid;
  if (result.exposureGainProductValid) {
    result.exposureGainProduct = result.exposureLines * result.nominalGainMultiplier;
  }

  result.dspRawControlsValid =
      readByteRegister(sensor, 0x0cc, 0xff, result.dspControlCc) &&
      readByteRegister(sensor, 0x0cd, 0xff, result.dspControlCd) &&
      readByteRegister(sensor, 0x0ce, 0xff, result.dspControlCe);
  return result;
}

Ov5640RegisterSnapshot readExposureGainRegisters(sensor_t *sensor) {
  Ov5640RegisterSnapshot result = {};
  result.supported = sensor != nullptr && sensor->id.PID == OV5640_PID;
  if (!result.supported || sensor->get_reg == nullptr) return result;

  uint32_t exposure = 0;
  for (uint8_t attempt = 1; attempt <= kMaxExposureConsistencyAttempts; ++attempt) {
    const int first = sensor->get_reg(sensor, 0x3500, 0xffff0);
    const int second = sensor->get_reg(sensor, 0x3500, 0xffff0);
    result.exposureReadAttempts = attempt;
    if (first < 0 || second < 0) break;
    exposure = static_cast<uint32_t>(second);
    result.exposureValid = true;
    result.exposureConsistent = first == second;
    if (result.exposureConsistent) break;
  }
  if (result.exposureValid) {
    result.exposureRaw16ths = exposure;
    result.exposureLines = exposure >> 4;
  }

  const int frameLength = sensor->get_reg(sensor, 0x380e, 0xffff);
  result.frameLengthValid = frameLength >= 0;
  if (result.frameLengthValid) result.frameLengthLines = static_cast<uint16_t>(frameLength);

  const int gain = sensor->get_reg(sensor, 0x350a, 0x03ff);
  result.gainValid = gain >= 0;
  if (result.gainValid) {
    result.gainRaw = static_cast<uint16_t>(gain);
    result.nominalGainMultiplier = static_cast<float>(result.gainRaw) / 16.0f;
  }
  result.exposureGainProductValid =
      result.exposureValid && result.exposureConsistent && result.gainValid;
  if (result.exposureGainProductValid) {
    result.exposureGainProduct = static_cast<float>(result.exposureLines) *
                                 result.nominalGainMultiplier;
  }

  return result;
}

}  // namespace

uint16_t cameraSensorConfigKey(uint16_t pid, uint8_t version) {
  if (pid == OV2640_PID) return (pid << 8) | version;
  return pid;
}

const char *cameraSensorModelName(uint16_t pid, uint8_t version) {
  const uint16_t key = cameraSensorConfigKey(pid, version);
  if (key == kOv2640SensorKey) return "OV2640";
  if (key == kOv5640SensorKey) return "OV5640";
  return "unknown";
}

bool initializeCamera() {
  CameraMutexGuard cameraLock;
  if (!cameraLock.locked()) return false;
  camera_config_t cameraConfig = cameraConfigFor(FRAMESIZE_VGA, 10, kProbeXclkHz);

  esp_err_t error = esp_camera_init(&cameraConfig);
  if (error != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", error);
    return false;
  }

  sensor_t *sensor = esp_camera_sensor_get();
  if (sensor == nullptr) {
    Serial.println("Camera sensor detection failed");
    esp_camera_deinit();
    return false;
  }

  const uint16_t sensorKey = cameraSensorConfigKey(sensor->id.PID, sensor->id.VER);
  const CameraSensorProfile *profile = getCameraSensorProfile(sensorKey);
  if (profile == nullptr || !selectCameraSensorProfile(sensorKey)) {
    Serial.printf("Unsupported or unconfigured camera sensor: PID 0x%04x, VER 0x%02x, key 0x%04x\n",
                  static_cast<unsigned int>(sensor->id.PID), sensor->id.VER,
                  static_cast<unsigned int>(sensorKey));
    esp_camera_deinit();
    return false;
  }

  if (cfg.framesize < static_cast<int>(FRAMESIZE_QQVGA) ||
      cfg.framesize > cameraSensorMaximumFramesize(sensorKey)) {
    Serial.printf("Configured frame size %s is unsupported by %s\n",
                  configFramesizeName(cfg.framesize), cameraSensorModelName(sensor->id.PID, sensor->id.VER));
    esp_camera_deinit();
    return false;
  }

  if (!reinitializeCameraForProfile(sensorKey, *profile)) return false;
  sensor = esp_camera_sensor_get();
  if (sensor == nullptr) {
    Serial.println("Camera sensor unavailable after profile initialization");
    return false;
  }

  Serial.printf("Camera ready: %s (PID 0x%04x, VER 0x%02x, config %s, SCCB 0x%02x, XCLK %d Hz)\n",
                cameraSensorModelName(sensor->id.PID, sensor->id.VER),
                static_cast<unsigned int>(sensor->id.PID), sensor->id.VER,
                cameraSensorKeyName(sensorKey), sensor->slv_addr, sensor->xclk_freq_hz);

  return true;
}

bool applyCameraProfile() {
  CameraMutexGuard cameraLock;
  if (!cameraLock.locked()) return false;
  const uint16_t sensorKey = getActiveCameraSensorKey();
  const CameraSensorProfile *profile = getCameraSensorProfile(sensorKey);
  sensor_t *sensor = esp_camera_sensor_get();
  if (profile == nullptr || sensor == nullptr ||
      cameraSensorConfigKey(sensor->id.PID, sensor->id.VER) != sensorKey) {
    return false;
  }
  if (profile->framesize == initializedFramesize && profile->xclkHz == initializedXclkHz) {
    applyCameraSettings();
    return true;
  }
  if (profile->framesize == initializedFramesize) {
    applyCameraSettings();
    return true;
  }

  const CameraSensorProfile previousLiveProfile = {
      initializedFramesize,
      sensor->status.quality,
      initializedXclkHz,
  };
  CameraSensorProfile runtimeProfile = *profile;
  runtimeProfile.xclkHz = initializedXclkHz;
  if (reinitializeCameraForProfile(sensorKey, runtimeProfile)) return true;

  Serial.println("Restoring previous live camera profile after reinitialization failure");
  if (!reinitializeCameraForProfile(sensorKey, previousLiveProfile)) {
    Serial.println("Could not restore previous live camera profile");
  }
  markRegisterDumpPending();
  return false;
}

bool reinitializeActiveCamera() {
  const uint16_t sensorKey = getActiveCameraSensorKey();
  if (getCameraSensorProfile(sensorKey) == nullptr) return false;
  const CameraSensorProfile runtimeProfile = {
      initializedFramesize,
      cfg.jpegQuality,
      initializedXclkHz,
  };
  const bool recovered = reinitializeCameraForProfile(sensorKey, runtimeProfile);
  if (recovered) markRegisterDumpPending();
  return recovered;
}

bool isCameraRestartRequired() {
  const CameraSensorProfile *profile = getCameraSensorProfile(getActiveCameraSensorKey());
  return profile != nullptr && profile->xclkHz != initializedXclkHz;
}

void applyCameraSettings() {
  CameraMutexGuard cameraLock;
  if (!cameraLock.locked()) return;
  sensor_t *sensor = esp_camera_sensor_get();
  bool changed = false;
  if (sensor == nullptr || !setConfiguredJpegMode(sensor, changed)) {
    Serial.println("Could not apply configured JPEG camera settings");
  } else if (changed && !drainCameraBuffer()) {
    Serial.println("Could not drain stale frame after applying JPEG camera settings");
  }
}

bool drainCameraBuffer() {
  // With one framebuffer and CAMERA_GRAB_WHEN_EMPTY there can be exactly one
  // completed or in-flight frame to invalidate. Do not loop: fb_get blocks and
  // the sensor continuously produces a replacement after this frame is returned.
  delay(120);
  camera_fb_t *staleFrame = acquireCameraFrame();
  if (staleFrame == nullptr) return false;
  releaseCameraFrame(staleFrame);
  delay(30);
  return true;
}

bool discardQueuedCameraFrame() {
  // Same single-frame invalidation as drainCameraBuffer(), without the settle
  // delays: the sensor mode is unchanged, so the only problem is that the queued
  // frame was grabbed right after the previous fb_return and may be arbitrarily
  // old. Returning it makes the next fb_get block for a frame exposed from now.
  camera_fb_t *staleFrame = acquireCameraFrame();
  if (staleFrame == nullptr) return false;
  releaseCameraFrame(staleFrame);
  return true;
}

int getCameraFrameBufferCount() { return initializedFrameBufferCount; }

camera_grab_mode_t getCameraGrabMode() { return initializedGrabMode; }

bool cameraUsesLatestFrameGrab() { return initializedGrabMode == CAMERA_GRAB_LATEST; }

bool prepareJpegCapture(bool drainQueuedFrame) {
  CameraMutexGuard cameraLock;
  if (!cameraLock.locked()) return false;
  sensor_t *sensor = esp_camera_sensor_get();
  if (sensor == nullptr) return false;

  bool changed = false;
  if (!setConfiguredJpegMode(sensor, changed)) return false;
  if ((changed || drainQueuedFrame) && !drainCameraBuffer()) return false;
  return true;
}

bool configureMotionCapture(bool &changed) {
  CameraMutexGuard cameraLock;
  if (!cameraLock.locked()) {
    changed = false;
    return false;
  }
  sensor_t *sensor = esp_camera_sensor_get();
  if (sensor == nullptr) {
    changed = false;
    return false;
  }

  // The camera HAL was initialized for JPEG and cannot safely be changed to
  // raw grayscale by updating only the OV5640 sensor. Keep the HAL and sensor
  // in JPEG mode. QQVGA, QVGA, and VGA can all be decoded directly to the same
  // 80x60 analysis image. Keeping those configured capture sizes here avoids an
  // AEC/AGC restart immediately before the first JPEG in a motion burst.
  framesize_t motionFramesize = FRAMESIZE_QQVGA;
  const framesize_t configuredFramesize = static_cast<framesize_t>(cfg.framesize);
  if (configuredFramesize == FRAMESIZE_QQVGA || configuredFramesize == FRAMESIZE_QVGA ||
      configuredFramesize == FRAMESIZE_VGA) {
    motionFramesize = configuredFramesize;
  }
  return setJpegMode(sensor, motionFramesize, cfg.jpegQuality, changed);
}

camera_fb_t *acquireCameraFrame() {
  if (!lockCamera()) return nullptr;
  camera_fb_t *frameBuffer = esp_camera_fb_get();
  if (frameBuffer != nullptr) {
    ++framesSinceBoot;
  } else {
    unlockCamera();
  }
  return frameBuffer;
}

void releaseCameraFrame(camera_fb_t *frameBuffer) {
  if (frameBuffer == nullptr) return;
  esp_camera_fb_return(frameBuffer);
  unlockCamera();
}

bool getCameraSensorIdentity(uint16_t &pid, uint8_t &version) {
  CameraMutexGuard cameraLock;
  if (!cameraLock.locked()) return false;
  sensor_t *sensor = esp_camera_sensor_get();
  if (sensor == nullptr) return false;
  pid = sensor->id.PID;
  version = sensor->id.VER;
  return true;
}

bool captureJpeg(CapturedJpeg &capture, bool drainQueuedFrame) {
  CameraMutexGuard cameraLock;
  if (!cameraLock.locked()) return false;
  capture = {};
  if (!prepareJpegCapture(drainQueuedFrame)) return false;

  camera_fb_t *frameBuffer = acquireCameraFrame();
  if (!isCompleteJpeg(frameBuffer)) {
    if (frameBuffer != nullptr) releaseCameraFrame(frameBuffer);
    return false;
  }

  capture.frameBuffer = frameBuffer;
  // Register reads occur now, while this exact retained JPEG is still held.
  capture.metadata = sampleCameraMetadata(frameBuffer);
  return true;
}

bool isCompleteJpeg(const camera_fb_t *frameBuffer) {
  return frameBuffer != nullptr && frameBuffer->format == PIXFORMAT_JPEG &&
         frameBuffer->len >= 4 && frameBuffer->buf[0] == 0xff &&
         frameBuffer->buf[1] == 0xd8 && frameBuffer->buf[frameBuffer->len - 2] == 0xff &&
         frameBuffer->buf[frameBuffer->len - 1] == 0xd9;
}

CameraMetadataSnapshot sampleCameraMetadata(const camera_fb_t *frameBuffer) {
  CameraMutexGuard cameraLock;
  CameraMetadataSnapshot snapshot = {};
  if (!cameraLock.locked()) return snapshot;
  snapshot.requestedFramesize = cfg.framesize;
  snapshot.requestedJpegQuality = cfg.jpegQuality;
  snapshot.requestedMotionEnabled = cfg.motionEnabled;
  snapshot.requestedMotionOwner = cfg.motionOwner;
  snapshot.requestedMotionThreshold = cfg.motionThreshold;
  snapshot.requestedMotionSensitivity = cfg.motionSensitivity;
  snapshot.requestedMotionCheckMs = cfg.motionCheckMs;
  snapshot.requestedMotionRoiX = cfg.motionRoiX;
  snapshot.requestedMotionRoiY = cfg.motionRoiY;
  snapshot.requestedMotionRoiWidth = cfg.motionRoiWidth;
  snapshot.requestedMotionRoiHeight = cfg.motionRoiHeight;
  snapshot.requestedSaveBirdsToSd = cfg.saveBirdsToSd;
  snapshot.requestedBirdConfidenceThreshold = cfg.birdConfidenceThreshold;
  snapshot.settingsRevision = settingsRevision;
  snapshot.persistentSettingsRevision = persistentSettingsRevision;
  snapshot.frameBufferCount = initializedFrameBufferCount;
  snapshot.grabMode = initializedGrabMode;

  sensor_t *sensor = esp_camera_sensor_get();
  if (sensor != nullptr) {
    snapshot.sensorAvailable = true;
    snapshot.sensorPid = sensor->id.PID;
    snapshot.sensorVersion = sensor->id.VER;
    snapshot.manufacturerHigh = sensor->id.MIDH;
    snapshot.manufacturerLow = sensor->id.MIDL;
    snapshot.sensorAddress = sensor->slv_addr;
    snapshot.xclkHz = sensor->xclk_freq_hz;
    snapshot.driverCache = copyDriverCache(sensor);
    snapshot.registerReadBootUs = esp_timer_get_time();
    if (!registerDumpCacheValid) refreshRegisterDump(sensor);
    snapshot.ov2640Observed = readObservedOv2640(sensor);
    snapshot.ov5640Observed = readExposureGainRegisters(sensor);
    snapshot.registerDumpValid = registerDumpCacheValid;
    snapshot.registerDumpSha256 = registerDumpSha256;
    snapshot.includeRegisterDumpPayload = registerDumpPending && registerDumpCacheValid;
  }

  snapshot.device = sampleDeviceTelemetry();

  if (frameBuffer != nullptr) {
    snapshot.hasFrame = true;
    snapshot.frameWidth = frameBuffer->width;
    snapshot.frameHeight = frameBuffer->height;
    snapshot.frameBytes = frameBuffer->len;
    snapshot.framePixelFormat = frameBuffer->format;
    snapshot.frameTimestampUs = static_cast<int64_t>(frameBuffer->timestamp.tv_sec) * 1000000LL +
                                frameBuffer->timestamp.tv_usec;
    snapshot.frameNumberSinceBoot = framesSinceBoot;
    if (settingsChangeTimeValid && snapshot.frameTimestampUs >= lastSettingsChangeUs) {
      snapshot.settingsAgeValid = true;
      snapshot.msSinceSettingsChange =
          static_cast<uint64_t>((snapshot.frameTimestampUs - lastSettingsChangeUs) / 1000);
    }
  }
  return snapshot;
}

ExposureGainSample sampleExposureGain() {
  CameraMutexGuard cameraLock;
  ExposureGainSample sample = {};
  if (!cameraLock.locked()) return sample;
  sensor_t *sensor = esp_camera_sensor_get();
  if (sensor == nullptr) return sample;

  if (sensor->id.PID == OV2640_PID) {
    const Ov2640RegisterSnapshot observed = readObservedOv2640(sensor);
    sample.valid = observed.supported && observed.exposureGainProductValid &&
                   observed.exposureGainProduct > 0.0f && isfinite(observed.exposureGainProduct);
    sample.exposureLines = observed.exposureLines;
    sample.gainRaw = observed.gainRaw;
    sample.nominalGainMultiplier = observed.nominalGainMultiplier;
    sample.product = observed.exposureGainProduct;
  } else {
    const Ov5640RegisterSnapshot observed = readExposureGainRegisters(sensor);
    sample.valid = observed.supported && observed.exposureGainProductValid &&
                   observed.exposureGainProduct > 0.0f && isfinite(observed.exposureGainProduct);
    sample.exposureLines = observed.exposureLines;
    sample.gainRaw = observed.gainRaw;
    sample.nominalGainMultiplier = observed.nominalGainMultiplier;
    sample.product = observed.exposureGainProduct;
  }
  return sample;
}

void markRegisterDumpPending() {
  ++persistentSettingsRevision;
  registerDumpPending = true;
  registerDumpCacheValid = false;
  registerDumpCacheBytes = 0;
  registerDumpSha256 = "";
  registerDumpPayloadBase64 = "";
}

void acknowledgeRegisterDumpDelivery(const CameraMetadataSnapshot &snapshot) {
  if (snapshot.includeRegisterDumpPayload && snapshot.registerDumpValid &&
      snapshot.persistentSettingsRevision == persistentSettingsRevision &&
      snapshot.registerDumpSha256 == registerDumpSha256 && registerDumpPayloadBase64.length() > 0) {
    registerDumpPending = false;
  }
}

String buildCameraMetadata(const CameraMetadataSnapshot &snapshot, const char *source) {
  JsonDocument document;
  document["schema_version"] = 5;
  document["source"] = source;

  JsonObject device = document["device"].to<JsonObject>();
  device["camera_id"] = snapshot.device.cameraId;
  device["boot_id"] = snapshot.device.bootId;
  JsonObject clock = device["clock"].to<JsonObject>();
  clock["sync_status"] = snapshot.device.timeSynchronized ? "synchronized" : "unsynchronized";
  if (snapshot.device.timeSynchronized) {
    clock["device_time_utc"] = snapshot.device.deviceTimeUtc;
  } else {
    clock["device_time_utc"] = nullptr;
  }
  clock["device_time_boot_us"] = snapshot.device.sampledAtBootUs;
  clock["semantic"] = "wall_clock_sampled_at_device_time_boot_us_ntp_may_step";

  JsonObject resources = device["resources"].to<JsonObject>();
  resources["chip_temperature_c"] = snapshot.device.chipTemperatureC;
  resources["chip_temperature_qualifier"] =
      "uncalibrated_internal_die_temperature_not_ambient";
  JsonObject heap = resources["heap"].to<JsonObject>();
  heap["free_bytes"] = snapshot.device.freeHeapBytes;
  heap["minimum_free_bytes_since_boot"] = snapshot.device.minimumFreeHeapBytes;
  heap["largest_free_block_bytes"] = snapshot.device.largestFreeHeapBlockBytes;
  JsonObject psram = resources["psram"].to<JsonObject>();
  psram["present"] = snapshot.device.psramTotalBytes > 0;
  psram["total_bytes"] = snapshot.device.psramTotalBytes;
  psram["free_bytes"] = snapshot.device.psramFreeBytes;
  psram["minimum_free_bytes_since_boot"] = snapshot.device.psramMinimumFreeBytes;
  psram["largest_free_block_bytes"] = snapshot.device.psramLargestFreeBlockBytes;

  JsonObject config = device["config"].to<JsonObject>();
  config["storage"] = getConfigStorageName();
  config["file_hash_valid"] = isConfigFileHashValid();
  config["hash_scope"] = isConfigFileHashValid() ? "exact_active_config_file_bytes"
                                                  : "no_valid_active_config_file_hash";
  if (isConfigFileHashValid()) {
    config["file_sha256"] = getConfigFileSha256();
  } else {
    config["file_sha256"] = nullptr;
  }

  JsonObject camera = document["camera"].to<JsonObject>();
  JsonObject provenance = camera["driver"].to<JsonObject>();
  char arduinoVersion[16];
  snprintf(arduinoVersion, sizeof(arduinoVersion), "%d.%d.%d", ESP_ARDUINO_VERSION_MAJOR,
           ESP_ARDUINO_VERSION_MINOR, ESP_ARDUINO_VERSION_PATCH);
  provenance["arduino_esp32_version"] = arduinoVersion;
  provenance["arduino_esp32_version_value"] = ESP_ARDUINO_VERSION;
  JsonObject cameraVersion = provenance["esp32_camera"].to<JsonObject>();
  cameraVersion["distribution"] = "bundled_with_arduino_esp32";
  cameraVersion["independent_version"] = nullptr;
  cameraVersion["bundled_revision"] = "f0bb429";
  cameraVersion["bundled_revision_source"] = "Arduino-ESP32 2.0.17 tools/sdk/versions.txt";

  if (snapshot.sensorAvailable) {
    JsonObject identity = camera["sensor"].to<JsonObject>();
    const char *model = cameraSensorModelName(snapshot.sensorPid, snapshot.sensorVersion);
    identity["model"] = model;
    identity["manufacturer"] = strcmp(model, "unknown") == 0 ? "unknown" : "OmniVision";
    identity["pid"] = snapshot.sensorPid;
    identity["pid_hex"] = hexValue(snapshot.sensorPid, 4);
    const bool extendedIdentityAvailable = snapshot.sensorPid == OV2640_PID;
    identity["version_available"] = extendedIdentityAvailable;
    if (extendedIdentityAvailable) {
      identity["version"] = snapshot.sensorVersion;
      identity["version_hex"] = hexValue(snapshot.sensorVersion, 2);
    } else {
      identity["version"] = nullptr;
      identity["version_hex"] = nullptr;
    }
    identity["config_key"] = cameraSensorKeyName(
        cameraSensorConfigKey(snapshot.sensorPid, snapshot.sensorVersion));
    const uint16_t manufacturerId =
        (static_cast<uint16_t>(snapshot.manufacturerHigh) << 8) | snapshot.manufacturerLow;
    identity["manufacturer_id_available"] = extendedIdentityAvailable;
    if (extendedIdentityAvailable) {
      identity["manufacturer_id"] = manufacturerId;
      identity["manufacturer_id_hex"] = hexValue(manufacturerId, 4);
    } else {
      identity["manufacturer_id"] = nullptr;
      identity["manufacturer_id_hex"] = nullptr;
    }
    identity["identity_qualifier"] = snapshot.sensorPid == OV2640_PID
        ? "ov2640_config_key_combines_driver_pid_0x26_and_version_0x42"
        : "ov5640_driver_detection_populates_pid_only";
    identity["sccb_address"] = snapshot.sensorAddress;
    identity["sccb_address_hex"] = hexValue(snapshot.sensorAddress, 2);
    identity["xclk_hz"] = snapshot.xclkHz;
  }

  JsonObject requested = camera["requested"].to<JsonObject>();
  JsonObject requestedCapture = requested["capture"].to<JsonObject>();
  JsonObject requestedFrameSize = requestedCapture["frame_size"].to<JsonObject>();
  requestedFrameSize["name"] = framesizeName(snapshot.requestedFramesize);
  requestedFrameSize["value"] = snapshot.requestedFramesize;
  JsonObject requestedPixelFormat = requestedCapture["pixel_format"].to<JsonObject>();
  requestedPixelFormat["name"] = pixelFormatName(PIXFORMAT_JPEG);
  requestedPixelFormat["value"] = PIXFORMAT_JPEG;
  requestedCapture["jpeg_quality"] = snapshot.requestedJpegQuality;
  JsonObject requestedMotion = requested["motion"].to<JsonObject>();
  requestedMotion["owner"] = motionOwnerName(snapshot.requestedMotionOwner);
  requestedMotion["enabled"] = snapshot.requestedMotionEnabled;
  requestedMotion["threshold"] = snapshot.requestedMotionThreshold;
  requestedMotion["sensitivity"] = snapshot.requestedMotionSensitivity;
  requestedMotion["check_ms"] = snapshot.requestedMotionCheckMs;
  JsonObject requestedMotionRoi = requestedMotion["roi"].to<JsonObject>();
  requestedMotionRoi["x"] = snapshot.requestedMotionRoiX;
  requestedMotionRoi["y"] = snapshot.requestedMotionRoiY;
  requestedMotionRoi["width"] = snapshot.requestedMotionRoiWidth;
  requestedMotionRoi["height"] = snapshot.requestedMotionRoiHeight;
  JsonObject requestedPipeline = requested["pipeline"].to<JsonObject>();
  requestedPipeline["save_birds_to_sd"] = snapshot.requestedSaveBirdsToSd;
  requestedPipeline["bird_confidence_threshold"] = snapshot.requestedBirdConfidenceThreshold;

  JsonObject observed = camera["observed"].to<JsonObject>();
  JsonObject observedCapture = observed["capture"].to<JsonObject>();
  JsonObject observedFrameSize = observedCapture["frame_size"].to<JsonObject>();
  observedFrameSize["name"] = framesizeName(snapshot.driverCache.framesize);
  observedFrameSize["value"] = snapshot.driverCache.framesize;
  JsonObject observedPixelFormat = observedCapture["pixel_format"].to<JsonObject>();
  observedPixelFormat["name"] = pixelFormatName(snapshot.hasFrame ? snapshot.framePixelFormat
                                                                   : snapshot.driverCache.pixelFormat);
  observedPixelFormat["value"] = snapshot.hasFrame ? snapshot.framePixelFormat
                                                     : snapshot.driverCache.pixelFormat;
  observedCapture["jpeg_quality"] = snapshot.driverCache.jpegQuality;
  if (snapshot.hasFrame) {
    observedCapture["width"] = snapshot.frameWidth;
    observedCapture["height"] = snapshot.frameHeight;
    observedCapture["framebuffer_bytes"] = snapshot.frameBytes;
    observedCapture["frames_since_boot"] = snapshot.frameNumberSinceBoot;
  }

  JsonObject controls = observed["controls"].to<JsonObject>();
  controls["semantic"] = "sensor_driver_cached_state_not_live_register_observation";
  controls["brightness"] = snapshot.driverCache.brightness;
  controls["contrast"] = snapshot.driverCache.contrast;
  controls["saturation"] = snapshot.driverCache.saturation;
  controls["special_effect"] = snapshot.driverCache.specialEffect;
  controls["white_balance_mode"] = snapshot.driverCache.whiteBalanceMode;
  controls["auto_white_balance"] = snapshot.driverCache.autoWhiteBalance;
  controls["auto_white_balance_gain"] = snapshot.driverCache.autoWhiteBalanceGain;
  controls["auto_exposure"] = snapshot.driverCache.autoExposure;
  controls["auto_exposure_dsp"] = snapshot.driverCache.autoExposureDsp;
  controls["exposure_level"] = snapshot.driverCache.exposureLevel;
  controls["auto_gain"] = snapshot.driverCache.autoGain;
  controls["gain_ceiling"] = snapshot.driverCache.gainCeiling;
  controls["horizontal_mirror"] = snapshot.driverCache.horizontalMirror;
  controls["vertical_flip"] = snapshot.driverCache.verticalFlip;

  JsonObject frameBuffer = provenance["framebuffer"].to<JsonObject>();
  frameBuffer["fb_count"] = snapshot.frameBufferCount;
  frameBuffer["grab_mode"] = grabModeName(snapshot.grabMode);
  frameBuffer["grab_mode_value"] = snapshot.grabMode;
  frameBuffer["queue_age_possible"] =
      snapshot.frameBufferCount > 1 || snapshot.grabMode == CAMERA_GRAB_WHEN_EMPTY;
  frameBuffer["age_qualifier"] =
      "queued_frame_and_sensor_aec_agc_pipeline_can_make_frame_older_than_post_frame_registers";

  if (snapshot.ov2640Observed.supported) {
    const Ov2640RegisterSnapshot &sample = snapshot.ov2640Observed;
    JsonObject registers = observed["ov2640_registers"].to<JsonObject>();
    registers["observation_qualifier"] = snapshot.hasFrame
        ? "post_frame_non_atomic_with_buffer_and_sensor_pipeline_delay"
        : "profile_time_non_atomic_live_register_reads";
    registers["registers_are_exact_frame_state"] = false;
    JsonObject exposure = registers["aec"].to<JsonObject>();
    exposure["valid"] = sample.exposureValid;
    exposure["consistent"] = sample.exposureConsistent;
    exposure["read_attempts"] = sample.exposureReadAttempts;
    exposure["max_consistency_attempts"] = kMaxExposureConsistencyAttempts;
    exposure["sensor_bank_0x45_mask_0x3f"] = sample.aecHigh;
    exposure["sensor_bank_0x10_mask_0xff"] = sample.aecMiddle;
    exposure["sensor_bank_0x04_mask_0x03"] = sample.aecLow;
    if (sample.exposureValid) exposure["aec_lines"] = sample.exposureLines;
    exposure["aec_driver_limit_lines"] = sample.aecDriverLimitLines;
    exposure["aec_max_lines"] = sample.aecDriverLimitLines;
    exposure["aec_max_lines_source"] = "esp32_camera_f0bb429_set_aec_value_driver_limit";
    exposure["ae_saturated_valid"] = sample.aeSaturatedValid;
    if (sample.aeSaturatedValid) exposure["ae_saturated"] = sample.aeSaturated;
    else exposure["ae_saturated"] = nullptr;
    exposure["ae_saturated_qualifier"] =
        "derived_from_aec_enabled_consistent_lines_at_driver_limit_not_sensor_status_bit";
    JsonObject gain = registers["gain"].to<JsonObject>();
    gain["valid"] = sample.gainValid;
    gain["sensor_bank_0x00_mask_0xff"] = sample.gainRaw;
    gain["nominal_multiplier_formula"] = "2^popcount(high_nibble)*(1+low_nibble/16)";
    if (sample.gainValid) gain["nominal_multiplier"] = sample.nominalGainMultiplier;
    if (sample.exposureGainProductValid) registers["exposure_gain_product"] = sample.exposureGainProduct;
    else registers["exposure_gain_product"] = nullptr;
    JsonObject dsp = registers["dsp_raw_controls"].to<JsonObject>();
    dsp["label"] = "raw_controls_not_verified_live_awb_gains";
    dsp["verified_live_awb_gains"] = false;
    dsp["valid"] = sample.dspRawControlsValid;
    dsp["dsp_bank_0xcc"] = sample.dspControlCc;
    dsp["dsp_bank_0xcd"] = sample.dspControlCd;
    dsp["dsp_bank_0xce"] = sample.dspControlCe;
  } else if (snapshot.ov5640Observed.supported) {
    const Ov5640RegisterSnapshot &sample = snapshot.ov5640Observed;
    JsonObject registers = observed["ov5640_registers"].to<JsonObject>();
    registers["observation_qualifier"] = snapshot.hasFrame
        ? "post_frame_non_atomic_with_buffer_and_sensor_pipeline_delay"
        : "profile_time_non_atomic_live_register_reads";
    registers["registers_are_exact_frame_state"] = false;
    JsonObject exposure = registers["aec"].to<JsonObject>();
    exposure["valid"] = sample.exposureValid;
    exposure["consistent"] = sample.exposureConsistent;
    exposure["read_attempts"] = sample.exposureReadAttempts;
    exposure["max_consistency_attempts"] = kMaxExposureConsistencyAttempts;
    exposure["register_0x3500_mask_0xffff0_raw_1_16_lines"] =
        sample.exposureRaw16ths;
    if (sample.exposureValid) exposure["aec_lines"] = sample.exposureLines;
    exposure["frame_length_valid"] = sample.frameLengthValid;
    if (sample.frameLengthValid) {
      exposure["frame_length_lines_register_0x380e_mask_0xffff"] =
          sample.frameLengthLines;
    }
    exposure["qualifier"] =
        "ov5640_exposure_is_reported_in_1_16_line_units_then_floor_converted_to_lines";

    JsonObject gain = registers["gain"].to<JsonObject>();
    gain["valid"] = sample.gainValid;
    gain["register_0x350a_mask_0x03ff"] = sample.gainRaw;
    gain["nominal_multiplier_formula"] = "raw_10_bit_gain/16";
    if (sample.gainValid) {
      gain["nominal_multiplier"] = sample.nominalGainMultiplier;
    }
    if (sample.exposureGainProductValid) {
      registers["exposure_gain_product"] = sample.exposureGainProduct;
    } else {
      registers["exposure_gain_product"] = nullptr;
    }

  }

  JsonObject timing = camera["timing"].to<JsonObject>();
  timing["settings_revision"] = snapshot.settingsRevision;
  if (snapshot.settingsAgeValid) {
    timing["ms_since_settings_change"] = snapshot.msSinceSettingsChange;
  } else {
    timing["ms_since_settings_change"] = nullptr;
  }

  if (snapshot.hasFrame) {
    timing["frame_boot_us"] = snapshot.frameTimestampUs;
    timing["register_read_boot_us"] = snapshot.registerReadBootUs;
    timing["register_read_minus_frame_us"] = snapshot.registerReadBootUs - snapshot.frameTimestampUs;
    timing["frame_timestamp_semantic"] = "camera_driver_framebuffer_timestamp_not_exposure_start";
    timing["register_timestamp_semantic"] = "immediately_before_first_sccb_register_read";
  } else {
    timing["frame_boot_us"] = nullptr;
    timing["register_read_boot_us"] = snapshot.registerReadBootUs;
    timing["register_read_minus_frame_us"] = nullptr;
  }

  JsonObject dump = camera["register_dump"].to<JsonObject>();
  dump["valid"] = snapshot.registerDumpValid;
  if (snapshot.sensorPid == OV2640_PID) {
    dump["format"] = "ov2640_banked_register_dump_v1";
    dump["canonical_bytes"] = kOv2640RegisterDumpBytes;
    dump["address_range"] = "0x00-0xfe_per_bank";
    dump["binary_layout"] =
        "ascii_OVD1_then_dsp_255_values_then_dsp_32_byte_validity_bitmap_then_sensor_equivalent";
  } else {
    dump["format"] = "ov5640_sparse_register_dump_v1";
    dump["canonical_bytes"] = kOv5640RegisterDumpBytes;
    dump["addresses"] =
        "0x300a,0x300b,0x3400-0x3406,0x3500-0x3502,0x350a-0x350b,0x380e-0x380f";
    dump["binary_layout"] =
        "ascii_OV5D_then_repeated_big_endian_address_validity_byte_value_byte";
  }
  dump["revision"] = snapshot.persistentSettingsRevision;
  dump["payload_included"] = snapshot.includeRegisterDumpPayload;
  dump["qualifier"] = "best_effort_non_atomic_register_sweep";
  if (snapshot.registerDumpValid) {
    dump["sha256"] = snapshot.registerDumpSha256;
    if (snapshot.includeRegisterDumpPayload) {
      dump["encoding"] = "base64";
      dump["payload_base64"] = registerDumpPayloadBase64;
    }
  } else {
    dump["sha256"] = nullptr;
  }

  String json;
  serializeJson(document, json);
  return json;
}