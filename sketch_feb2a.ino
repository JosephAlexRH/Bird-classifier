#include <Arduino.h>

#include <WiFi.h>
#include "esp_camera.h"
#include "esp_timer.h"
#include "img_converters.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include "driver/rtc_io.h"

#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>

#include <SPIFFS.h>
#include <FS.h>

//WiFi credentials 
const char* ssid = "EE-68GCWZ";
const char* password = "7TQmcK6dVfLqLgM9";

const char* laptopIP = "192.168.1.129";
const int   laptopPort = 5000;

//burst configuration
volatile boolean captureBurst = false;
const int burstCount = 3;
const int burstDelayMs = 1500;   // 1.5 seconds

volatile boolean motionDetectionActive = false;
volatile boolean photoFromMotion = false;

//capture confirguration 
//#define CAPTURE_PIXEL_FORMAT PIXFORMAT_RGB565 //PIXFORMAT_JPEG FOR JPEG
//#define CAPTURE_FRAME_SIZE FRAMESIZE_QVGA  //FRAMESIZE_VGA

#define CAPTURE_PIXEL_FORMAT  PIXFORMAT_GRAYSCALE  
#define CAPTURE_FRAME_SIZE    FRAMESIZE_QQVGA       // 160×120 = 19,200 bytes

#define FRAME_W 160
#define FRAME_H 120
#define FRAME_PIXELS (FRAME_W * FRAME_H)

static uint8_t prevFrame[FRAME_PIXELS];
static uint8_t currFrame[FRAME_PIXELS];
static bool    hasPrevFrame = false;

//Downsample constants 
//scale factor: 2 = 80x60, 4 = 40x30
#define DOWNSAMPLE_FACTOR 2
#define DS_W (FRAME_W / DOWNSAMPLE_FACTOR)  
#define DS_H (FRAME_H / DOWNSAMPLE_FACTOR)  
#define DS_PIXELS (DS_W * DS_H)            

static uint8_t dsPrev[DS_PIXELS];
static uint8_t dsCurr[DS_PIXELS];

#define CONFIG_ASYNC_TCP_MAX_ACK_TIME 3000
#define CONFIG_ASYNC_TCP_PRIORITY 10
#define CONFIG_ASYNC_TCP_RUNNING_CORE 1
AsyncWebServer server(80);

volatile boolean takeNewPhoto = false;

#define FILE_PHOTO "/photo.jpg"

#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27

#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21 
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5

#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22


const char index_html[] PROGMEM = R"rawliteral(
<!DOCTYPE HTML><html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
  body { text-align:center; font-family: sans-serif; }
  #motionBtn { background-color: #888; color: white; padding: 6px 12px; border: none; cursor: pointer; }
  #motionBtn.active { background-color: #c0392b; }
  #photo { max-width: 90%; max-height: 80vh; width: auto; height: auto; }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
  </style>

</head>
<body>
  <div id="container">
    <h2>ESP32-CAM</h2>
    <p id="statusText" style="font-weight:bold; font-size:14px;">IDLE</p>
    <p>
      <button id="captureBtn" onclick="capturePhoto()">CAPTURE PHOTO</button>
      <button onclick="location.reload();">REFRESH PAGE</button>
      <button id="motionBtn" onclick="toggleMotion()">MOTION DETECTION: OFF</button>
      <button onclick="sendRequest('/resetcam', 2, function(){})">RESET CAMERA</button>
    </p>
    <div id="sensorReadings" style="margin-top:10px; font-size:14px;">
      <span>Chip Temp: <b id="chipTemp">--</b> C</span>
      &nbsp;&nbsp;
      <span>Heap: <b id="heapUsed">--</b>% used (<b id="freeHeap">--</b>KB free of <b id="totalHeap">--</b>KB)</span>
      &nbsp;&nbsp;
      <span>PSRAM: <b id="freePsram">--</b>KB free of <b id="totalPsram">--</b>KB</span>
      <span>
    </div>
  </div>
  <div><img src="/saved-photo" id="photo"></div>
</body>
<script>
  var motionActive = false;

  function setStatus(state, text) {
  var el = document.getElementById("statusText");
  el.textContent = text;
  el.style.color = state === "ready" ? "green" : state === "busy" ? "orange" : "red";
  }

  function setBusy(busy) {
    document.getElementById("captureBtn").disabled = busy;
    document.getElementById("motionBtn").disabled = busy;
    setStatus(busy ? "busy" : "ready", busy ? "BUSY..." : "IDLE");
  }

  // poll ESP32 status every second so the UI stays in sync
  function pollStatus() {
    var xhr = new XMLHttpRequest();
    xhr.open('GET', "/status", true);
    xhr.timeout = 2000;
    xhr.onload = function() {
      if (xhr.status === 200) {
        var busy = xhr.responseText.trim() === "busy";
        setBusy(busy);
        // if it just became ready and we were waiting, refresh the photo
        if (!busy && document.getElementById("captureBtn").dataset.waiting === "true") {
          document.getElementById("photo").src = "/saved-photo?t=" + new Date().getTime();
          document.getElementById("captureBtn").dataset.waiting = "false";
        }
      }
    }
    xhr.onerror = function() { setStatus("error", "NO CONNECTION"); }
    xhr.send();
  }
  setInterval(pollStatus, 1000);

  function pollSensors() {
    var xhr = new XMLHttpRequest();
    xhr.open('GET', "/sensors", true);
    xhr.timeout = 2000;
    xhr.onload = function() {
      if (xhr.status === 200) {
        var data = JSON.parse(xhr.responseText);
        document.getElementById("chipTemp").textContent = data.temp;
        document.getElementById("freeHeap").textContent   = data.freeHeap;
        document.getElementById("totalHeap").textContent  = data.totalHeap;
        document.getElementById("heapUsed").textContent   = data.heapUsed;
        document.getElementById("freePsram").textContent  = data.freePsram;
        document.getElementById("totalPsram").textContent = data.totalPsram;
      }
    }
  xhr.send();
  }
setInterval(pollSensors, 2000);  // update every 2 seconds

  function capturePhoto() {
    document.getElementById("captureBtn").dataset.waiting = "true";
    setBusy(true);
    sendRequest("/capture", 3, function() {
      // don't refresh photo immediately - wait for pollStatus to see ready
    });
  }

  function toggleMotion() {
    var btn = document.getElementById("motionBtn");
    btn.disabled = true;
    sendRequest("/motion", 3, function() {
      motionActive = !motionActive;
      btn.textContent = motionActive ? "MOTION DETECTION: ON" : "MOTION DETECTION: OFF";
      btn.classList.toggle("active", motionActive);
      btn.disabled = false;
    });
  }

  function sendRequest(url, retries, onSuccess) {
    var xhr = new XMLHttpRequest();
    xhr.open('GET', url, true);
    xhr.timeout = 5000;
    xhr.onload = function() {
      if (xhr.status === 200) {
        onSuccess();
      } else if (retries > 0) {
        setTimeout(function() { sendRequest(url, retries - 1, onSuccess); }, 1000);
      } else {
        setStatus("error", "FAILED");
        setBusy(false);
      }
    }
    xhr.onerror = function() {
      if (retries > 0) {
        setTimeout(function() { sendRequest(url, retries - 1, onSuccess); }, 1000);
      } else {
        setStatus("error", "CONNECTION ERROR");
        setBusy(false);
      }
    }
    xhr.ontimeout = function() {
      if (retries > 0) {
        setTimeout(function() { sendRequest(url, retries - 1, onSuccess); }, 1000);
      } else {
        setStatus("error", "TIMED OUT");
        setBusy(false);
      }
    }
    xhr.send();
  }
</script>
</html>)rawliteral";


#if ESP_CAMERA_SUPPORTED
static camera_config_t camera_config = {
  .pin_pwdn= PWDN_GPIO_NUM,
  .pin_reset = RESET_GPIO_NUM,
  .pin_xclk = XCLK_GPIO_NUM,
  .pin_sccb_sda = SIOD_GPIO_NUM,
  .pin_sccb_scl = SIOC_GPIO_NUM,
  
  .pin_d7 = Y9_GPIO_NUM,
  .pin_d6 = Y8_GPIO_NUM,
  .pin_d5 = Y7_GPIO_NUM,
  .pin_d4 = Y6_GPIO_NUM,
  .pin_d3 = Y5_GPIO_NUM,
  .pin_d2 = Y4_GPIO_NUM,
  .pin_d1 = Y3_GPIO_NUM,
  .pin_d0 = Y2_GPIO_NUM,

  .pin_vsync = VSYNC_GPIO_NUM,
  .pin_href = HREF_GPIO_NUM,
  .pin_pclk = PCLK_GPIO_NUM,

  .xclk_freq_hz = 20000000,
  .ledc_timer = LEDC_TIMER_0,
  .ledc_channel = LEDC_CHANNEL_0,

  .pixel_format =CAPTURE_PIXEL_FORMAT,  
  .frame_size = CAPTURE_FRAME_SIZE,
  .jpeg_quality = 12, 

  .fb_count =1,
  .fb_location =  CAMERA_FB_IN_PSRAM,
  .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
};
#endif
float getChipTemp() {
  return temperatureRead();
}

void setup(){

  Serial.begin(115200);
  
   // Turn-off the 'brownout detector'
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
  //WiFi

  WiFi.mode(WIFI_STA);
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);
  WiFi.begin(ssid, password); 
  while(WiFi.status() != WL_CONNECTED){
    Serial.print(".");
    delay(100);
  }

  if (!SPIFFS.begin(true)) {
  Serial.println("An Error has occurred while mounting SPIFFS");
  ESP.restart();
  }
  else {
  delay(500);
  Serial.println("SPIFFS mounted successfully");
  Serial.println("=== SPIFFS Info ===");
  Serial.printf("Total: %d KB\n", SPIFFS.totalBytes() / 1024);
  Serial.printf("Used:  %d KB\n", SPIFFS.usedBytes() / 1024);
  Serial.printf("Free:  %d KB\n", (SPIFFS.totalBytes() - SPIFFS.usedBytes()) / 1024);

  //delete oldphotos on boot to free space
  File root = SPIFFS.open("/");
  File file = root.openNextFile();
  while (file) {
  String name = file.name();
  if (name.startsWith("photo_")) {
    Serial.printf("Deleting old photo: %s\n", name.c_str());
    SPIFFS.remove("/" + name);
  }
  file = root.openNextFile();
  }
  root.close();
  Serial.println("Old photos cleared");
  }

  // Print ESP32 Local IP Address
  Serial.print("IP Address: http://");
  Serial.println(WiFi.localIP());

 
  #if ESP_CAMERA_SUPPORTED
    esp_err_t err = esp_camera_init(&camera_config);
    camera_fb_t *pic = esp_camera_fb_get();
      if (pic == nullptr){
      Serial.println("Camera capture failed (pic is null)");
      return;
      }
      

    Serial.println("=== Frame info ===");
    Serial.print("Width: ");
    Serial.println(pic->width);
    Serial.print("Height: ");
    Serial.println(pic->height);
    Serial.print("Length (bytes): ");
    Serial.println(pic->len);
    Serial.print("Format: ");
    Serial.println(pic->format);

    esp_camera_fb_return(pic);


    sensor_t* s = esp_camera_sensor_get();
    if (s != NULL) {
    s->set_exposure_ctrl(s, 1);      // enable auto exposure
    s->set_aec2(s, 1);               // enable auto exposure correction
    s->set_ae_level(s, 0);           // exposure level -2 to +2
    s->set_aec_value(s, 300);        // exposure value, lower = darker
    s->set_gain_ctrl(s, 1);          // enable auto gain
    s->set_agc_gain(s, 0);// starting gain 0-30
    s->set_gainceiling(s, (gainceiling_t)2);  // cap max gain - key setting for reducing noise
    s->set_bpc(s, 1); // bad pixel correction
    s->set_wpc(s, 1);// white pixel correction
    s->set_lenc(s, 1); // lens correction
}
  #endif
  
  // Route for root / web page
  server.on("/", HTTP_GET, [](AsyncWebServerRequest * request) {
    request->send(200, "text/html", index_html);
  });

  server.on("/capture", HTTP_GET, [](AsyncWebServerRequest * request) {
    photoFromMotion = false;
    takeNewPhoto = true;
    request->send(200, "text/plain", "Taking Photo");
  });
  server.on("/burst", HTTP_GET, [](AsyncWebServerRequest * request) {
  captureBurst = true;
  request->send(200, "text/plain", "Burst started");
  });

  server.on("/saved-photo", HTTP_GET, [](AsyncWebServerRequest * request) {
  request->send(SPIFFS, FILE_PHOTO, "image/jpg", false);
  });
  server.on("/photo1", HTTP_GET, [](AsyncWebServerRequest * request) {
  request->send(SPIFFS, "/photo1.jpg", "image/jpg", false);
  });

  server.on("/photo2", HTTP_GET, [](AsyncWebServerRequest * request) {
  request->send(SPIFFS, "/photo2.jpg", "image/jpg", false);
  });

  server.on("/photo3", HTTP_GET, [](AsyncWebServerRequest * request) {
  request->send(SPIFFS, "/photo3.jpg", "image/jpg", false);
  });
  server.on("/motion", HTTP_GET, [](AsyncWebServerRequest *request) {
  motionDetectionActive = !motionDetectionActive;
  request->send(200, "text/plain", motionDetectionActive ? "Motion ON" : "Motion OFF");
  });
  server.on("/resetcam", HTTP_GET, [](AsyncWebServerRequest *request) {
  sensor_t* s = esp_camera_sensor_get();
  if (s != NULL) {
    s->set_gainceiling(s, (gainceiling_t)2);
    s->set_aec_value(s, 300);
    s->set_agc_gain(s, 0);
    s->set_bpc(s, 1);
    s->set_wpc(s, 1);
  }
  request->send(200, "text/plain", "Camera reset");
});
  server.on("/status", HTTP_GET, [](AsyncWebServerRequest *request) {
  String state = "ready";
  if (takeNewPhoto || captureBurst) state = "busy";
  request->send(200, "text/plain", state);
  });
  server.on("/resetcam", HTTP_GET, [](AsyncWebServerRequest *request) {
  sensor_t* s = esp_camera_sensor_get();
  if (s != NULL) {
    s->set_gainceiling(s, (gainceiling_t)2);
    s->set_aec_value(s, 300);
    s->set_agc_gain(s, 0);
    s->set_bpc(s, 1);
    s->set_wpc(s, 1);
  }
  request->send(200, "text/plain", "Camera reset");
});

  server.on("/sensors", HTTP_GET, [](AsyncWebServerRequest *request) {
    float chipTemp = getChipTemp();

    uint32_t freeHeap     = ESP.getFreeHeap();
    uint32_t totalHeap    = ESP.getHeapSize();
    uint32_t freePsram    = ESP.getFreePsram();
    uint32_t totalPsram   = ESP.getPsramSize();
    uint8_t  heapPercent  = (uint8_t)(100.0 - ((float)freeHeap / totalHeap * 100.0));

    String json = "{";
    json += "\"temp\":"+ String(chipTemp, 1)+ ",";
    json += "\"freeHeap\":"+ String(freeHeap /1024)+ ",";
    json += "\"totalHeap\":"+ String(totalHeap/ 1024)+ ",";
    json += "\"heapUsed\":"+ String(heapPercent)+ ",";
    json += "\"freePsram\":" + String(freePsram/1024)+ ",";
    json += "\"totalPsram\":"+ String(totalPsram / 1024);
    json += "}";

    request->send(200, "application/json", json);
  });

  // Start server
  server.begin();
}

#define MOTION_THRESHOLD 10.0f  //lower = more sensitive

void loop() {
  //frame comparison
  if (motionDetectionActive) {
    camera_fb_t* fb = esp_camera_fb_get();
    if(fb){
      memcpy(currFrame, fb->buf, fb->len);
      esp_camera_fb_return(fb);

      downsample(currFrame, dsCurr);

      if (hasPrevFrame) {
        float delta = compareFrames();
        Serial.printf("Frame delta: %.2f\n", delta);

        if (delta > MOTION_THRESHOLD && !takeNewPhoto && !captureBurst) {
          Serial.println("Motion detected");
          photoFromMotion = true;
          takeNewPhoto = true;
        }
      }

    memcpy(dsPrev, dsCurr, DS_PIXELS);
    hasPrevFrame = true;
  }
  }else{
    hasPrevFrame=false; //reset so moteion detection doesnt reuse stale frame
  }

  //single capture 
  if (takeNewPhoto) {
    capturePhotoSaveSpiffs();
    takeNewPhoto = false;
  }

  //burst capture 
  if (captureBurst) {
    for (int i = 0; i < burstCount; i++) {
      capturePhotoSaveSpiffs();
      delay(burstDelayMs);
    }
    captureBurst = false;
  }

  delay(200);  //frame comparison rate 
}

// Check if photo capture was successful
bool checkPhoto( fs::FS &fs ) {
  File f_pic = fs.open( FILE_PHOTO );
  unsigned int pic_sz = f_pic.size();
  return ( pic_sz > 100 );
}

// Capture Photo and Save it to SPIFFS
void capturePhotoSaveSpiffs( void ) {
  camera_fb_t * fb = NULL;
  bool ok = false; 

  char filename[40];
  String source = photoFromMotion ? "motion" : "manual";

  //build file name 
  snprintf(filename, sizeof(filename), "/photo_%s_%lu.jpg", source.c_str(), millis());
    
  do {
    fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
      return;
    }
    //Photo file name
    Serial.printf("Picture file name: %s\n", FILE_PHOTO);
    File file = SPIFFS.open(filename, FILE_WRITE);

    //Insert the data in the photo file
    if (!file) {
      Serial.println("Failed to open file in writing mode");
      esp_camera_fb_return(fb);
      return;
    }
  uint8_t* jpg_buf = NULL;
  size_t   jpg_len = 0;

  bool converted = fmt2jpg(fb->buf, fb->len, fb->width, fb->height,PIXFORMAT_GRAYSCALE, 80, &jpg_buf, &jpg_len);
  if (converted) {
    file.write(jpg_buf, jpg_len);
    free(jpg_buf);
  } else {
    Serial.println("JPEG conversion failed");
    file.close();
    esp_camera_fb_return(fb);
    return;
  }
  file.close();
  esp_camera_fb_return(fb);

  File check = SPIFFS.open(filename);
  ok = check && check.size() > 100;
  check.close();
  File src = SPIFFS.open(filename, FILE_READ);
  File dst = SPIFFS.open(FILE_PHOTO, FILE_WRITE);
  if (src && dst) {
  uint8_t buf[512];
  while (src.available()) {
    int n = src.read(buf, sizeof(buf));
    dst.write(buf, n);
  }
}
src.close();
dst.close();
  } while ( !ok );
   sendPhoto(FILE_PHOTO);
} 
void downsample(uint8_t* src, uint8_t* dst) {
  for (int y = 0; y < DS_H; y++) {
    for (int x = 0; x < DS_W; x++) {
      // average a 2×2 block for better quality than nearest-neighbour
      int s = DOWNSAMPLE_FACTOR;
      uint32_t sum = 0;
      for (int dy = 0; dy < s; dy++)
        for (int dx = 0; dx < s; dx++)
          sum += src[(y*s + dy) * FRAME_W + (x*s + dx)];
      dst[y * DS_W + x] = sum / (s * s);
    }
  }
}
float compareFrames() {
  uint32_t diff = 0;
  for (int i = 0; i < DS_PIXELS; i++) {
    diff += abs((int)dsCurr[i] - (int)dsPrev[i]);
  }
  // returns average pixel difference, 0.0 = identical, 255.0 = total change
  return (float)diff / DS_PIXELS;
}
void sendPhoto(const char* filename) {
  File file = SPIFFS.open(filename, FILE_READ);
  if (!file) {
    Serial.println("Failed to open file for sending");
    return;
  }

  WiFiClient client;
  if (!client.connect(laptopIP, laptopPort)) {
    Serial.println("Connection to laptop failed");
    file.close();
    return;
  }

  size_t fileSize = file.size();

  // send HTTP POST headers
  client.printf("POST /receive HTTP/1.1\r\n");
  client.printf("Host: %s:%d\r\n", laptopIP, laptopPort);
  client.printf("Content-Type: image/jpeg\r\n");
  client.printf("Content-Length: %d\r\n", fileSize);
  client.printf("\r\n");

  // send file in chunks
  uint8_t buf[512];
  while (file.available()) {
    int bytesRead = file.read(buf, sizeof(buf));
    client.write(buf, bytesRead);
  }

  file.close();

  //server timeout timer 
  unsigned long timeout = millis();
  while (client.connected() && millis() - timeout < 3000) {
    if (client.available()) break;
    delay(10);
  }

  client.stop();
  Serial.println("Photo sent to laptop");
}
