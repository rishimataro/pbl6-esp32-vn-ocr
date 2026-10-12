"""
ESP32-CAM WiFi dataset capture.

Firmware: CameraWebServer_copy_20261012160211 (with the /hires endpoint).

  Preview : http://<ip>:81/stream   MJPEG 640x480
  Capture : http://<ip>/hires       one JPEG 1600x1200

Usage:
  python esp32_wifi_capture.py --host 192.168.1.50
  python esp32_wifi_capture.py --host 192.168.1.50 --snap      # 1 capture, no UI

Keys (preview window):
  SPACE / C = capture 1600x1200
  Q / ESC   = quit
"""

import argparse
import csv
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests


# ============================================================
# CONFIG
# ============================================================

# IP printed on the ESP32 serial monitor (115200 baud) after WiFi connects
DEFAULT_HOST = "192.168.1.50"

STREAM_PORT = 81
DEFAULT_QUALITY = 10        # OV2640 JPEG quality 4-63, lower = better
CAPTURE_TIMEOUT = 20        # seconds

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "dataset" / "esp32"

METADATA_HEADER = [
    "id",
    "filename",
    "timestamp",
    "width",
    "height",
    "jpeg_bytes",
    "quality",
    "capture_ms",
    "esp_timestamp",
]

CONTENT_LENGTH_RE = re.compile(rb"Content-Length:\s*(\d+)", re.IGNORECASE)


# ============================================================
# MJPEG STREAM READER
# ============================================================

class MjpegStream:
    """Reads the ESP32 MJPEG stream in a background thread, keeps the latest frame."""

    def __init__(self, url):
        self.url = url
        self.running = False
        self.thread = None
        self.response = None

        self.lock = threading.Lock()
        self.frame = None
        self.frame_time = 0.0
        self.fps = 0.0
        self.status = "Connecting..."

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        # Closing the response unblocks iter_content() in the reader thread
        if self.response is not None:
            try:
                self.response.close()
            except Exception:
                pass
        if self.thread:
            self.thread.join(timeout=2)

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None, self.status, 0.0
            stale = time.time() - self.frame_time > 2.0
            status = "Stream paused" if stale else self.status
            return self.frame.copy(), status, self.fps

    def _run(self):
        while self.running:
            try:
                self._set_status("Connecting...")
                self.response = requests.get(self.url, stream=True, timeout=(5, 10))
                self.response.raise_for_status()
                self._set_status("Streaming")
                self._read_frames(self.response)
            except Exception as error:
                if self.running:
                    print(f"[STREAM] {error}")
                    self._set_status("Reconnecting...")
                    time.sleep(1.0)
            finally:
                if self.response is not None:
                    self.response.close()

    def _read_frames(self, response):
        buffer = bytearray()
        last_tick = time.time()
        frame_count = 0

        for chunk in response.iter_content(chunk_size=8192):
            if not self.running:
                return
            buffer += chunk

            # Each part: "\r\n--boundary\r\nContent-Type...\r\nContent-Length: N\r\n...\r\n\r\n<N bytes JPEG>"
            while True:
                header_end = buffer.find(b"\r\n\r\n")
                if header_end < 0:
                    break

                match = CONTENT_LENGTH_RE.search(buffer, 0, header_end)
                if not match:
                    del buffer[:header_end + 4]
                    continue

                length = int(match.group(1))
                start = header_end + 4
                if len(buffer) < start + length:
                    break

                jpeg = bytes(buffer[start:start + length])
                del buffer[:start + length]

                frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                frame_count += 1
                now = time.time()
                with self.lock:
                    self.frame = frame
                    self.frame_time = now
                    if now - last_tick >= 1.0:
                        self.fps = frame_count / (now - last_tick)
                        frame_count = 0
                        last_tick = now

    def _set_status(self, text):
        with self.lock:
            self.status = text


# ============================================================
# HIGH-RES CAPTURE
# ============================================================

def request_hires(session, host, quality):
    """GET /hires and return a dict with the raw JPEG bytes and info. Raises on failure."""

    url = f"http://{host}/hires"
    started = time.time()

    response = session.get(url, params={"quality": quality}, timeout=CAPTURE_TIMEOUT)
    response.raise_for_status()

    if not response.headers.get("Content-Type", "").startswith("image/jpeg"):
        raise RuntimeError(f"Unexpected response: {response.text[:100]!r}")

    jpeg = response.content
    elapsed_ms = int((time.time() - started) * 1000)

    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("JPEG decode failed")

    height, width = frame.shape[:2]

    # A JPEG bigger than the frame buffer gets cut off -> no EOI marker
    truncated = not jpeg.rstrip(b"\x00").endswith(b"\xff\xd9")

    return {
        "jpeg": jpeg,
        "frame": frame,
        "width": width,
        "height": height,
        "quality": quality,
        "capture_ms": elapsed_ms,
        "esp_timestamp": response.headers.get("X-Timestamp", ""),
        "truncated": truncated,
    }


# ============================================================
# DATASET (files + metadata)
# ============================================================

class Dataset:

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.output_dir / "metadata.csv"
        self._init_metadata()
        self.next_number = self._find_next_number()

    def _find_next_number(self):
        numbers = [int(f.stem) for f in self.output_dir.glob("*.jpg") if f.stem.isdigit()]
        return max(numbers, default=0) + 1

    def _init_metadata(self):
        rows = []
        if self.metadata_file.exists():
            with open(self.metadata_file, newline="", encoding="utf-8") as file:
                rows = list(csv.reader(file))

        if rows and rows[0] == METADATA_HEADER:
            return

        if len(rows) > 1:
            # Old file with data in another format: keep it, start a new one
            backup = self.metadata_file.with_name(f"metadata_old_{int(time.time())}.csv")
            self.metadata_file.rename(backup)
            print(f"[DATASET] Old metadata moved to {backup.name}")

        with open(self.metadata_file, "w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(METADATA_HEADER)

    def save(self, capture):
        image_id = f"{self.next_number:06d}"
        path = self.output_dir / f"{image_id}.jpg"

        # Save the original JPEG bytes (no cv2.imwrite -> no re-compression)
        path.write_bytes(capture["jpeg"])

        with open(self.metadata_file, "a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow([
                image_id,
                path.name,
                datetime.now().isoformat(timespec="milliseconds"),
                capture["width"],
                capture["height"],
                len(capture["jpeg"]),
                capture["quality"],
                capture["capture_ms"],
                capture["esp_timestamp"],
            ])

        self.next_number += 1

        print(
            f"[SAVED] {path.name}  {capture['width']}x{capture['height']}  "
            f"{len(capture['jpeg']) / 1024:.1f} KB  {capture['capture_ms']} ms"
        )
        if capture["truncated"]:
            print("        WARNING: JPEG may be truncated -> use a higher --quality number")
        if (capture["width"], capture["height"]) != (1600, 1200):
            print("        WARNING: unexpected resolution (firmware not updated?)")

        return path


# ============================================================
# UI
# ============================================================

def draw_interface(frame, status, fps, message, capturing, next_name):
    if frame is None:
        frame = np.zeros((480, 640, 3), np.uint8)
        cv2.putText(frame, status, (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    image = frame.copy()
    height, width = image.shape[:2]

    cv2.rectangle(image, (0, 0), (width, 36), (0, 0, 0), -1)
    cv2.putText(
        image,
        f"{width}x{height}  {fps:4.1f} fps  {status}",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
    )

    if capturing:
        bottom = "CAPTURING 1600x1200..."
        color = (0, 200, 255)
    elif message:
        bottom = message
        color = (0, 255, 0)
    else:
        bottom = f"SPACE = Capture   Q = Quit   Next: {next_name}"
        color = (255, 255, 255)

    cv2.rectangle(image, (0, height - 36), (width, height), (0, 0, 0), -1)
    cv2.putText(image, bottom, (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    return image


# ============================================================
# MAIN
# ============================================================

def run_snap(host, quality, dataset):
    with requests.Session() as session:
        capture = request_hires(session, host, quality)
    dataset.save(capture)


def run_ui(host, quality, dataset):
    stream = MjpegStream(f"http://{host}:{STREAM_PORT}/stream")
    stream.start()

    session = requests.Session()
    results = queue.Queue()
    capture_thread = None

    message = ""
    message_until = 0.0
    window = "ESP32-CAM Capture"

    def capture_worker():
        try:
            results.put(("ok", request_hires(session, host, quality)))
        except Exception as error:
            results.put(("error", error))

    try:
        while True:
            # Finished captures
            try:
                kind, payload = results.get_nowait()
                if kind == "ok":
                    path = dataset.save(payload)
                    message = f"SAVED {path.name} ({payload['width']}x{payload['height']})"
                else:
                    print(f"[CAPTURE ERROR] {payload}")
                    message = "CAPTURE FAILED (see console)"
                message_until = time.time() + 2.0
            except queue.Empty:
                pass

            if time.time() > message_until:
                message = ""

            capturing = capture_thread is not None and capture_thread.is_alive()
            frame, status, fps = stream.get_frame()

            cv2.imshow(
                window,
                draw_interface(frame, status, fps, message, capturing, f"{dataset.next_number:06d}.jpg"),
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (32, ord("c"), ord("C")) and not capturing:
                print("[CAPTURE] Requesting 1600x1200...")
                capture_thread = threading.Thread(target=capture_worker, daemon=True)
                capture_thread.start()

            elif key in (ord("q"), ord("Q"), 27):
                break

            # Window closed with the X button
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        pass

    finally:
        stream.stop()
        session.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="ESP32-CAM WiFi dataset capture")
    parser.add_argument("--host", default=DEFAULT_HOST, help="ESP32-CAM IP address")
    parser.add_argument("--quality", type=int, default=DEFAULT_QUALITY,
                        help="JPEG quality for capture, 4-63 (lower = better, default 10)")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR), help="output folder")
    parser.add_argument("--snap", action="store_true", help="capture one image and exit (no UI)")
    args = parser.parse_args()

    dataset = Dataset(args.out)

    print(f"ESP32-CAM : {args.host}")
    print(f"Preview   : http://{args.host}:{STREAM_PORT}/stream (640x480)")
    print(f"Capture   : http://{args.host}/hires (1600x1200, quality={args.quality})")
    print(f"Dataset   : {dataset.output_dir.resolve()}")
    print()

    if args.snap:
        run_snap(args.host, args.quality, dataset)
    else:
        run_ui(args.host, args.quality, dataset)


if __name__ == "__main__":
    main()
