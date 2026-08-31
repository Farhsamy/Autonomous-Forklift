"""
╔══════════════════════════════════════════════════════════════════════╗
║     AutoLift — Python Bridge  (main.py)  — Manual + Auto + Kinect   ║
║                                                                      ║
║  Runs on the PC.  Connects dashboard to Arduino Uno + ESP8266-01 AT firmware over WiFi.            ║
║  Streams webcam feed (with line-follower overlay) to dashboard.     ║
║                                                                      ║
║  Start:  python main.py                                              ║
║  Open:   forklift_dashboard.html  in your browser                   ║
╚══════════════════════════════════════════════════════════════════════╝

REST endpoints:
  GET  /status          → {connected, speed, steer, fork, limit_up, limit_dn,
                           mode, auto_running, emergency_stop}
  POST /forward         body: {"speed": 0-80}
  POST /backward        body: {"speed": 0-80}
  POST /stop
  POST /left
  POST /right
  POST /fork_up
  POST /fork_down
  POST /fork_stop
  POST /depth_blend     body: {"blend": 0.0-1.0}

  POST /auto/start      → Start autonomous line-follower mode
  POST /auto/stop       → Stop autonomous mode
  POST /emergency_stop  → Hard-stop everything (manual + auto)
  POST /emergency_clear → Clear emergency stop flag

  GET  /stream          → MJPEG stream  (Kinect depth/aruco panel)
  GET  /frame           → single JPEG   (Kinect, fallback polling)
  GET  /stream2         → MJPEG stream  (webcam / line-follower panel, autonomous mode)
  GET  /frame2          → single JPEG   (webcam, fallback polling)
"""

import sys
import time
import threading
import io
import serial

import cv2
import numpy as np
import requests
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# ── Try Kinect ────────────────────────────────────────────────────────
try:
    from pykinect2 import PyKinectRuntime, PyKinectV2
    KINECT_AVAILABLE = True
except ImportError:
    KINECT_AVAILABLE = False
    print("[WARN] pykinect2 not found — Kinect stream unavailable.")

# ── Try YOLO ─────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not found — person detection disabled.")

# ── Try ArUco / pallet detector (Kinect overlay) ──────────────────────
try:
    from aruco_pallet_follow import process_frame as aruco_process_frame
    ARUCO_AVAILABLE = True
except ImportError:
    ARUCO_AVAILABLE = False
    print("[WARN] aruco_pallet_follow not found — Kinect detection overlay disabled.")

# ═════════════════════════════════════════════════════════════════
# CONFIGURATION  ← edit these
# ═════════════════════════════════════════════════════════════════
import os

# ESP8266-01 IP when it is used as an AT Wi-Fi module connected directly to Arduino Uno.
# The Arduino sketch below sets a static IP by default. Keep this matching that sketch.
# You can also set it without editing: PowerShell → $env:AUTOLIFT_WIFI_IP="172.20.10.X"
WIFI_MODULE_IP        = os.getenv("AUTOLIFT_WIFI_IP", "172.20.10.5")   # ← CHANGE IF YOU CHANGE ARDUINO STATIC IP
WIFI_MODULE_PORT_HTTP = 80

# Direct Arduino USB serial is disabled in this ESP8266-direct version.
# Autonomous mode sends line-following commands through Wi-Fi to the ESP8266/Arduino server.
USE_DIRECT_ARDUINO_SERIAL = False

FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000

STREAM_W = 960
STREAM_H = 540
STREAM_JPEG_QUALITY = 75

# Serial (for line-follower → Arduino)
AUTO_SERIAL_PORT = "COM3"          # ← CHANGE to your Arduino port
AUTO_BAUD_RATE   = 9600

# ArUco / pallet detection overlay on Kinect feed (manual mode)
ARUCO_TARGET_LABEL = "A"           # marker label to track (A/B/MC)
ARUCO_OVERLAY_ENABLED = True       # draw marker + pallet overlay on Kinect stream

# ═════════════════════════════════════════════════════════════════
# SHARED STATE
# ═════════════════════════════════════════════════════════════════
state = {
    "connected":     False,
    "speed":         0,
    "steer":         0.0,
    "fork":          "ST",
    "limit_up":      0,
    "limit_dn":      0,
    "detections":    [],
    "mode":          "manual",   # "manual" | "auto"
    "auto_running":  False,
    "emergency_stop": False,
}
state_lock = threading.Lock()

# Kinect / webcam
kinect_lock  = threading.Lock()
latest_frame = None        # Kinect feed (depth/aruco overlay)
depth_blend  = 0.40

webcam_lock  = threading.Lock()
latest_webcam_frame = None  # Webcam / line-follower feed (autonomous mode)

# Auto mode
auto_stop_event = threading.Event()

# Direct serial link to the Arduino (used by autonomous mode AND by
# emergency stop, so e-stop works even if the autonomous thread isn't
# running and even if the ESP8266/WiFi link is down).
arduino_lock = threading.Lock()
arduino_serial = None

def arduino_connect():
    """Optional direct USB serial connection. Disabled in ESP8266-direct mode."""
    global arduino_serial
    if not USE_DIRECT_ARDUINO_SERIAL:
        return None
    with arduino_lock:
        if arduino_serial is not None and arduino_serial.is_open:
            return arduino_serial
        try:
            arduino_serial = serial.Serial(AUTO_SERIAL_PORT, AUTO_BAUD_RATE, timeout=1)
            time.sleep(2)
            print(f"[ARDUINO] Connected on {AUTO_SERIAL_PORT}")
        except Exception as e:
            print(f"[ARDUINO] Not available: {e}")
            arduino_serial = None
        return arduino_serial

def arduino_send(cmd: str):
    """Optional direct USB serial write. Disabled in ESP8266-direct mode."""
    if not USE_DIRECT_ARDUINO_SERIAL:
        return False
    with arduino_lock:
        if arduino_serial and arduino_serial.is_open:
            try:
                arduino_serial.write((cmd.strip() + "\n").encode())
                return True
            except Exception as e:
                print(f"[ARDUINO] write failed: {e}")
    return False

def hard_stop_all():
    """Send STOP through the active control channel."""
    send_cmd("STOP")
    arduino_send("STOP")

# ═════════════════════════════════════════════════════════════════
# ESP8266-01 COMMUNICATION
# ═════════════════════════════════════════════════════════════════
def _url(path):
    return f"http://{WIFI_MODULE_IP}:{WIFI_MODULE_PORT_HTTP}{path}"

def send_cmd(cmd: str):
    try:
        # Arduino + ESP8266 AT server reads commands from the URL query.
        # Example: http://172.20.10.5/cmd?c=F23
        r = requests.get(_url("/cmd"), params={"c": cmd}, timeout=1)
        if r.status_code != 200:
            raise Exception(f"HTTP {r.status_code}")
    except Exception as e:
        print(f"[WIFI] Send error ({cmd}): {e}")
        with state_lock:
            state["connected"] = False

def telemetry_poll():
    while True:
        try:
            r = requests.get(_url("/telemetry"), timeout=2)
            if r.status_code == 200:
                with state_lock:
                    state["connected"] = True
        except Exception:
            with state_lock:
                state["connected"] = False
        time.sleep(0.3)

def wifi_connect():
    try:
        r = requests.get(_url("/telemetry"), timeout=3)
        if r.status_code == 200:
            with state_lock:
                state["connected"] = True
            print(f"[WIFI] Connected to ESP8266-01 at {WIFI_MODULE_IP}")
        else:
            raise Exception(f"HTTP {r.status_code}")
    except Exception as e:
        print(f"[WIFI] Not connected: {e}")

# ═════════════════════════════════════════════════════════════════
# ── LINE FOLLOWER (AUTONOMOUS MODE) ──────────────────────────────
# ═════════════════════════════════════════════════════════════════

# PID settings (mirrored from line_follower.py)
Kp = 0.08
Ki = 0.0
Kd = 0.018
MAX_STEER_ANGLE   = 25.0
DEAD_BAND_PX      = 18
INTEGRAL_LIMIT    = 2500.0
FRAME_WIDTH       = 640
FRAME_HEIGHT      = 480
MAX_MASK_RATIO    = 0.12
MIN_MASK_RATIO    = 0.002
YOLO_SKIP_FRAMES  = 5
PERSON_CONFIDENCE = 0.55
PERSON_CENTER_ZONE = 220
PERSON_CLEAR_DELAY = 1.0
CLEAR_YOLO_FRAMES_REQUIRED = 3
SERIAL_SEND_INTERVAL = 0.05
AUTO_DRIVE_SPEED     = 50   # forward speed (0-80) sent to Arduino while line-following


def _reset_pid(pid_state):
    pid_state["integral"]  = 0.0
    pid_state["last_error"] = 0.0
    pid_state["last_time"]  = None


def _pid_to_angle(error_px, pid_state):
    if abs(error_px) <= DEAD_BAND_PX:
        error_px = 0.0
    now = time.time()
    if pid_state["last_time"] is None:
        dt = 0.02
        derivative = 0.0
    else:
        dt = max(now - pid_state["last_time"], 0.001)
        derivative = (error_px - pid_state["last_error"]) / dt
    pid_state["integral"] += error_px * dt
    pid_state["integral"]  = float(np.clip(pid_state["integral"], -INTEGRAL_LIMIT, INTEGRAL_LIMIT))
    angle = Kp * error_px + Ki * pid_state["integral"] + Kd * derivative
    angle = float(np.clip(angle, -MAX_STEER_ANGLE, MAX_STEER_ANGLE))
    if error_px == 0:
        angle = 0.0
    pid_state["last_error"] = error_px
    pid_state["last_time"]  = now
    return angle, error_px


def _remove_person_boxes(mask, person_boxes, roi_y):
    h, w = mask.shape[:2]
    for box in person_boxes:
        x1, y1, x2, y2 = box
        ry1 = y1 - roi_y
        ry2 = y2 - roi_y
        pad = 45
        x1 = max(0, x1 - pad);  x2 = min(w, x2 + pad)
        ry1 = max(0, ry1 - pad); ry2 = min(h, ry2 + pad)
        if ry2 > 0 and ry1 < h:
            cv2.rectangle(mask, (x1, ry1), (x2, ry2), 0, -1)
    return mask


def _create_black_line_mask(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, w = gray.shape[:2]
    total = h * w
    p_dark   = np.percentile(gray, 7)
    base_thr = int(np.clip(p_dark + 12, 40, 100))
    thresholds = [base_thr, base_thr-8, base_thr-16, base_thr-24, base_thr-32, base_thr+8]
    best_mask = None
    best_thr  = base_thr
    for thr in thresholds:
        thr = int(np.clip(thr, 35, 125))
        m_gray = cv2.inRange(gray, 0, thr)
        m_hsv  = cv2.inRange(hsv, np.array([0,0,0]), np.array([180,255,min(thr+10,125)]))
        mask   = cv2.bitwise_or(m_gray, m_hsv)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8))
        ratio  = cv2.countNonZero(mask) / total
        best_mask = mask
        best_thr  = thr
        if MIN_MASK_RATIO <= ratio <= MAX_MASK_RATIO:
            break
    return best_mask, best_thr


def _get_line_center(mask):
    h, w = mask.shape[:2]
    if cv2.countNonZero(mask) / (h * w) > 0.22:
        return None, None, None
    scan_levels = [0.78,0.70,0.62,0.54,0.46,0.38,0.30,0.22]
    best_cx = best_cy = None
    for level in scan_levels:
        y    = int(h * level)
        band = 14
        y1   = max(0, y - band); y2 = min(h, y + band)
        col_sum     = np.sum(mask[y1:y2, :] > 0, axis=0)
        active_cols = np.where(col_sum > 3)[0]
        if len(active_cols) < 8:
            continue
        groups = []
        start = prev = active_cols[0]
        for x in active_cols[1:]:
            if x - prev > 6:
                groups.append((start, prev))
                start = x
            prev = x
        groups.append((start, prev))
        valid = [(x2-x1, (x1+x2)//2, x1, x2) for x1,x2 in groups if 8 <= x2-x1 <= 180]
        if not valid:
            continue
        valid.sort(key=lambda g: g[0], reverse=True)
        _, cx, _, _ = valid[0]
        best_cx, best_cy = cx, y
        break
    if best_cx is None:
        return None, None, None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_cont = None
    if contours:
        min_d = 999999
        for c in contours:
            if cv2.contourArea(c) < 100:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            d = abs(int(M["m10"]/M["m00"]) - best_cx)
            if d < min_d:
                min_d = d; best_cont = c
    return best_cx, best_cy, best_cont


def _detect_persons_auto(frame, model):
    small  = cv2.resize(frame, (320, 320))
    results = model(small, verbose=False)
    sx = FRAME_WIDTH  / 320
    sy = FRAME_HEIGHT / 320
    boxes = []
    hazard = False
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            if int(box.cls[0]) == 0 and float(box.conf[0]) >= PERSON_CONFIDENCE:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                x1=int(x1*sx); y1=int(y1*sy); x2=int(x2*sx); y2=int(y2*sy)
                boxes.append((x1,y1,x2,y2))
                if abs((x1+x2)//2 - FRAME_WIDTH//2) < PERSON_CENTER_ZONE:
                    hazard = True
    return boxes, hazard


def autonomous_loop():
    """Runs in its own thread when autonomous mode is active."""
    global latest_webcam_frame

    print("[AUTO] Autonomous thread started")

    # Use the shared Arduino serial connection (also used by e-stop)
    arduino_connect()

    # Try to open a dedicated webcam first. If that fails (e.g. on a
    # Kinect-only rig where index 0 doesn't exist or is busy), fall back
    # to frames captured by the Kinect thread via `latest_frame`.
    cap = cv2.VideoCapture(0)
    use_webcam = cap.isOpened()
    if use_webcam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        print("[AUTO] Using dedicated webcam (index 0)")
    else:
        cap.release()
        cap = None
        if not KINECT_AVAILABLE:
            print("[AUTO] No webcam and Kinect unavailable — cannot start autonomous mode")
            with state_lock:
                state["auto_running"] = False
                state["mode"]         = "manual"
                state["detections"]   = [{"label": "NO CAMERA SOURCE"}]
            return
        print("[AUTO] No webcam found — using Kinect color feed instead")

    # Load YOLO
    yolo_model = None
    if YOLO_AVAILABLE:
        try:
            yolo_model = YOLO("yolov8n.pt")
            print("[AUTO] YOLO loaded")
        except Exception as e:
            print(f"[AUTO] YOLO load failed: {e}")

    pid_state = {"integral": 0.0, "last_error": 0.0, "last_time": None}

    frame_count     = 0
    e_stop          = False
    last_person_t   = 0.0
    clear_frames    = 0
    active_boxes    = []
    last_send_time  = 0.0
    no_frame_warns  = 0

    while not auto_stop_event.is_set():
        # Check global emergency stop
        with state_lock:
            global_estop = state["emergency_stop"]
        if global_estop:
            send_cmd("STOP")
            time.sleep(0.05)
            continue

        if use_webcam:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        else:
            # Pull the most recent Kinect color frame
            with kinect_lock:
                src = latest_frame
            if src is None:
                no_frame_warns += 1
                if no_frame_warns % 60 == 1:
                    print("[AUTO] Waiting for Kinect frame...")
                time.sleep(0.05)
                continue
            frame = cv2.resize(src, (FRAME_WIDTH, FRAME_HEIGHT))

        display = frame.copy()

        # YOLO person detection
        frame_count += 1
        if yolo_model and frame_count % YOLO_SKIP_FRAMES == 0:
            boxes, hazard = _detect_persons_auto(frame, yolo_model)
            active_boxes = boxes
            now = time.time()
            if hazard:
                e_stop = True; last_person_t = now; clear_frames = 0
                with state_lock:
                    state["detections"] = [{"label":"PERSON STOP"}]
            else:
                if e_stop and now - last_person_t >= PERSON_CLEAR_DELAY:
                    clear_frames += 1
                    if clear_frames >= CLEAR_YOLO_FRAMES_REQUIRED:
                        e_stop = False; clear_frames = 0
                        _reset_pid(pid_state)
                with state_lock:
                    state["detections"] = []

        # ROI + mask
        roi_y = int(FRAME_HEIGHT * 0.40)
        roi   = frame[roi_y:FRAME_HEIGHT, :]
        h, w  = roi.shape[:2]
        cx_center = w // 2

        mask, used_thr = _create_black_line_mask(roi)
        mask = _remove_person_boxes(mask, active_boxes, roi_y)
        best_cx, best_cy, line_cont = _get_line_center(mask)

        # Command
        cmd       = "NL"
        drive_cmd = "S0"
        angle     = 0.0
        error     = 0.0

        if e_stop:
            cmd       = "STOP"
            drive_cmd = "STOP"
            _reset_pid(pid_state)
        elif best_cx is not None:
            raw_err = float(best_cx - cx_center)
            angle, error = _pid_to_angle(raw_err, pid_state)
            cmd       = f"A:{angle:.1f}"
            drive_cmd = f"F{AUTO_DRIVE_SPEED}"
        else:
            cmd       = "NL"
            drive_cmd = "S0"
            _reset_pid(pid_state)

        # Update shared state
        with state_lock:
            state["steer"] = angle
            state["speed"] = AUTO_DRIVE_SPEED if cmd.startswith("A:") else 0

        # Draw overlay on display
        if line_cont is not None:
            cv2.drawContours(roi, [line_cont], -1, (0,255,0), 2)
        if best_cx is not None:
            cv2.circle(roi, (best_cx, best_cy), 8, (0,0,255), -1)
        cv2.line(roi, (cx_center,0), (cx_center,h), (255,0,0), 2)

        overlay_color = (0,0,255) if e_stop else (0,255,0)
        cv2.putText(display, f"CMD: {cmd}",             (10,40),  cv2.FONT_HERSHEY_SIMPLEX, 1,    overlay_color, 2)
        cv2.putText(display, f"ERR:{error:.0f} ANG:{angle:.1f}", (10,80),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, overlay_color, 2)
        cv2.putText(display, f"THR:{used_thr}",         (10,115), cv2.FONT_HERSHEY_SIMPLEX, 0.65, overlay_color, 2)
        if e_stop:
            cv2.putText(display, "PERSON DETECTED — STOPPED", (10,155), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)

        # Publish frame for dashboard webcam panel
        with webcam_lock:
            latest_webcam_frame = display.copy()

        # Serial output (direct to Arduino over shared connection)
        now_s = time.time()
        if now_s - last_send_time >= SERIAL_SEND_INTERVAL:
            if cmd == "STOP":
                send_cmd("STOP")
            else:
                send_cmd(drive_cmd)
                send_cmd(cmd)
            last_send_time = now_s

    # Cleanup
    if use_webcam and cap is not None:
        cap.release()
    send_cmd("STOP")
    with state_lock:
        state["auto_running"] = False
        state["mode"]         = "manual"
    with webcam_lock:
        latest_webcam_frame = None
    print("[AUTO] Autonomous thread stopped")


# ═════════════════════════════════════════════════════════════════
# KINECT CAPTURE THREAD
# ═════════════════════════════════════════════════════════════════
def kinect_capture_thread():
    global latest_frame, depth_blend
    if not KINECT_AVAILABLE:
        return
    print("[KINECT] Opening Kinect V2…")
    try:
        kinect = PyKinectRuntime.PyKinectRuntime(
            PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth)
        print("[KINECT] Kinect V2 ready.")
    except Exception as e:
        print(f"[KINECT] Failed: {e}"); return
    color_frame = depth_frame = None
    raw_depth = None
    while True:
        try:
            if kinect.has_new_color_frame():
                raw = kinect.get_last_color_frame()
                bgra = raw.reshape((1080,1920,4)).astype(np.uint8)
                color_frame = cv2.resize(cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR), (STREAM_W, STREAM_H))
            if kinect.has_new_depth_frame():
                raw = kinect.get_last_depth_frame()
                raw_depth = cv2.resize(raw.reshape((424,512)).astype(np.uint16), (STREAM_W, STREAM_H))
                depth_frame = cv2.applyColorMap(cv2.normalize(raw_depth,None,0,255,cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
            if color_frame is not None:
                with kinect_lock:
                    blend = depth_blend
                combined = cv2.addWeighted(color_frame,1-blend,depth_frame,blend,0) if depth_frame is not None and blend>0 else color_frame.copy()

                # ── ArUco marker + pallet detection overlay ──
                if ARUCO_AVAILABLE and ARUCO_OVERLAY_ENABLED:
                    try:
                        combined, detections, _angle = aruco_process_frame(
                            combined, raw_depth, target_label=ARUCO_TARGET_LABEL, draw=True)
                        with state_lock:
                            state["detections"] = detections
                    except Exception as e:
                        print(f"[ARUCO] {e}")

                with kinect_lock:
                    latest_frame = combined
        except Exception as e:
            print(f"[KINECT] {e}"); time.sleep(0.1)
        time.sleep(0.033)

# ═════════════════════════════════════════════════════════════════
# MJPEG GENERATOR
# ═════════════════════════════════════════════════════════════════
def gen_mjpeg():
    while True:
        with kinect_lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.05); continue
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY])
        if not ok: continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.033)

def gen_mjpeg_webcam():
    while True:
        with webcam_lock:
            frame = latest_webcam_frame
        if frame is None:
            time.sleep(0.05); continue
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY])
        if not ok: continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.033)

# ═════════════════════════════════════════════════════════════════
# FLASK APP
# ═════════════════════════════════════════════════════════════════
app = Flask(__name__)
CORS(app)

@app.route("/status")
def status():
    with state_lock:
        return jsonify(dict(state))

@app.route("/stream")
def stream():
    return Response(gen_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/stream2")
def stream2():
    """Webcam / line-follower feed (active during autonomous mode)."""
    return Response(gen_mjpeg_webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/frame")
def frame_endpoint():
    with kinect_lock:
        f = latest_frame
    if f is None:
        return Response(status=204)
    ok, buf = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY])
    return Response(buf.tobytes(), mimetype='image/jpeg') if ok else Response(status=204)

@app.route("/frame2")
def frame2_endpoint():
    with webcam_lock:
        f = latest_webcam_frame
    if f is None:
        return Response(status=204)
    ok, buf = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY])
    return Response(buf.tobytes(), mimetype='image/jpeg') if ok else Response(status=204)

@app.route("/depth_blend", methods=["POST"])
def set_depth_blend():
    global depth_blend
    v = float((request.get_json(silent=True) or {}).get("blend", 0.4))
    v = max(0.0, min(1.0, v))
    with kinect_lock:
        depth_blend = v
    return jsonify({"ok": True, "blend": v})

@app.route("/forward", methods=["POST"])
def forward():
    with state_lock:
        if state["emergency_stop"] or state["auto_running"]:
            return jsonify({"ok":False,"reason":"blocked"})
    speed = max(0, min(80, int((request.get_json(silent=True) or {}).get("speed", 50))))
    send_cmd(f"F{speed}")
    with state_lock: state["speed"] = speed
    return jsonify({"ok": True, "cmd": f"F{speed}"})

@app.route("/backward", methods=["POST"])
def backward():
    with state_lock:
        if state["emergency_stop"] or state["auto_running"]:
            return jsonify({"ok":False,"reason":"blocked"})
    speed = max(0, min(80, int((request.get_json(silent=True) or {}).get("speed", 50))))
    send_cmd(f"B{speed}")
    with state_lock: state["speed"] = -speed
    return jsonify({"ok": True, "cmd": f"B{speed}"})

@app.route("/stop", methods=["POST"])
def stop():
    send_cmd("STOP")
    with state_lock: state["speed"] = 0
    return jsonify({"ok": True})

@app.route("/left", methods=["POST"])
def left():
    with state_lock:
        if state["emergency_stop"] or state["auto_running"]: return jsonify({"ok":False})
    send_cmd("L")
    with state_lock: state["steer"] = -15.0
    return jsonify({"ok": True})

@app.route("/right", methods=["POST"])
def right():
    with state_lock:
        if state["emergency_stop"] or state["auto_running"]: return jsonify({"ok":False})
    send_cmd("R")
    with state_lock: state["steer"] = 15.0
    return jsonify({"ok": True})

@app.route("/fork_up", methods=["POST"])
def fork_up():
    send_cmd("FU")
    with state_lock: state["fork"] = "UP"
    return jsonify({"ok": True})

@app.route("/fork_down", methods=["POST"])
def fork_down():
    send_cmd("FD")
    with state_lock: state["fork"] = "DN"
    return jsonify({"ok": True})

@app.route("/fork_stop", methods=["POST"])
def fork_stop():
    send_cmd("FS")
    with state_lock: state["fork"] = "ST"
    return jsonify({"ok": True})

# ── Autonomous mode ────────────────────────────────────────────
@app.route("/auto/start", methods=["POST"])
def auto_start():
    with state_lock:
        if state["emergency_stop"]:
            return jsonify({"ok":False,"reason":"emergency_stop active — clear it first"})
        if state["auto_running"]:
            return jsonify({"ok":False,"reason":"already running"})
        state["mode"]        = "auto"
        state["auto_running"] = True

    arduino_connect()
    auto_stop_event.clear()
    t = threading.Thread(target=autonomous_loop, daemon=True)
    t.start()
    return jsonify({"ok": True, "mode": "auto"})

@app.route("/auto/stop", methods=["POST"])
def auto_stop():
    auto_stop_event.set()
    hard_stop_all()
    with state_lock:
        state["mode"]        = "manual"
        state["auto_running"] = False
        state["speed"]       = 0
    return jsonify({"ok": True, "mode": "manual"})

# ── Emergency stop ─────────────────────────────────────────────
@app.route("/emergency_stop", methods=["POST"])
def emergency_stop():
    # Stop auto thread if running
    auto_stop_event.set()
    # Send hardware stop on every channel (WiFi/ESP8266 + direct Arduino serial)
    hard_stop_all()
    with state_lock:
        state["emergency_stop"] = True
        state["auto_running"]   = False
        state["mode"]           = "manual"
        state["speed"]          = 0
        state["steer"]          = 0.0
    return jsonify({"ok": True})

@app.route("/emergency_clear", methods=["POST"])
def emergency_clear():
    with state_lock:
        state["emergency_stop"] = False
        # Make sure a stale auto_running flag can't block the next /auto/start
        state["auto_running"]   = False
        state["mode"]           = "manual"
    auto_stop_event.set()
    return jsonify({"ok": True})

# ═════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  AutoLift Bridge — Manual + Autonomous + Kinect V2")
    print("=" * 60)
    print(f"  ESP8266 target:     http://{WIFI_MODULE_IP}")
    print(f"  Dashboard bridge: http://127.0.0.1:{FLASK_PORT}")
    print(f"  Stream:           http://127.0.0.1:{FLASK_PORT}/stream")
    print("=" * 60)

    wifi_connect()
    arduino_connect()

    t_telem = threading.Thread(target=telemetry_poll, daemon=True)
    t_telem.start()

    if KINECT_AVAILABLE:
        t_kinect = threading.Thread(target=kinect_capture_thread, daemon=True)
        t_kinect.start()
    else:
        print("[WARN] Kinect not available — /stream serves webcam in auto mode only.")

    app.run(host=FLASK_HOST, port=FLASK_PORT, threaded=True)
