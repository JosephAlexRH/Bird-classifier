# Freenove ESP32-S3 WROOM CAM:  field deployment


1. Seat the camera ribbon with the connector latch fully closed, contacts
   facing the board. Do this unpowered.
2. Prepare the SD card exactly as for the ESP32-CAM (FAT32, `config.json` at
   the root). Use the same `server_host` / `server_ip` as the rest of the fleet.
3. Plug a USB-C cable into the port wired to the **CH343 UART** chip (labelled
   UART/COM on most revisions), not the native-USB port. Windows shows a new
   COM port; note its number.
4. Build and flash:

   ```powershell
   pio run -e freenove_s3
   pio run -e freenove_s3 -t upload --upload-port COMx
   pio device monitor -e freenove_s3 --port COMx
   ```

   If upload can't connect, hold BOOT, tap RST, release BOOT, then retry.
   Once you know the COM number you can add `upload_port`/`monitor_port` to the
   `[env:freenove_s3]` block in `platformio.ini`.

The board's RGB LED flashes white at boot, then blinks blue while it joins
Wi-Fi and green once the hub answers; see
[Status light](camera-firmware.md#status-light) for every pattern. It stays lit
for 5 minutes after power-up by default.

**Pass:** the serial log shows the SD card mounted, the sensor detected (OV2640
or OV5640), Wi-Fi connected with an IP, and the camera finding `birdhub.local`.
`http://CAMERA_IP/status` answers and `http://CAMERA_IP/capture` returns a
sharp image.
