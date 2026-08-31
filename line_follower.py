import cv2
import numpy as np
import serial
import time
from ultralytics import YOLO

# ============================================================
# SERIAL SETTINGS
# ============================================================
SERIAL_PORT = "COM8"   # change to your Arduino port
BAUD_RATE = 9600
SERIAL_SEND_INTERVAL = 0.05

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    use_serial = True
    print("✅ Serial connected")
except Exception as e:
    print("⚠️ Serial not connected:", e)
    use_serial = False

last_send_time = 0.0

# ============================================================
# YOLO MODEL
# ============================================================
print("Loading YOLO...")
model = YOLO("yolov8n.pt")
print("✅ YOLO loaded")

# ============================================================
# CAMERA
# ============================================================
cap = cv2.VideoCapture(0)

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# ============================================================
# PID SETTINGS
# ============================================================
Kp = 0.08
Ki = 0.0
Kd = 0.018

MAX_STEER_ANGLE = 25.0
DEAD_BAND_PX = 18
INTEGRAL_LIMIT = 2500.0

integral = 0.0
last_error = 0.0
last_pid_time = None

# ============================================================
# LINE DETECTION SETTINGS
# ============================================================
MAX_MASK_RATIO = 0.12
MIN_MASK_RATIO = 0.002

# ============================================================
# PERSON / EMERGENCY STOP SETTINGS
# ============================================================
YOLO_SKIP_FRAMES = 5
PERSON_CONFIDENCE = 0.55
PERSON_CENTER_ZONE = 220

PERSON_CLEAR_DELAY = 1.0
CLEAR_YOLO_FRAMES_REQUIRED = 3

frame_count = 0
emergency_stop = False
last_person_seen_time = 0.0
clear_yolo_frames = 0
active_person_boxes = []


# ============================================================
# PID FUNCTIONS
# ============================================================
def reset_pid():
    global integral, last_error, last_pid_time

    integral = 0.0
    last_error = 0.0
    last_pid_time = None


def pid_to_angle(error_px):
    global integral, last_error, last_pid_time

    if abs(error_px) <= DEAD_BAND_PX:
        error_px = 0.0

    now = time.time()

    if last_pid_time is None:
        dt = 0.02
        derivative = 0.0
    else:
        dt = max(now - last_pid_time, 0.001)
        derivative = (error_px - last_error) / dt

    integral += error_px * dt
    integral = float(np.clip(integral, -INTEGRAL_LIMIT, INTEGRAL_LIMIT))

    angle = (Kp * error_px) + (Ki * integral) + (Kd * derivative)
    angle = float(np.clip(angle, -MAX_STEER_ANGLE, MAX_STEER_ANGLE))

    if error_px == 0:
        angle = 0.0

    last_error = error_px
    last_pid_time = now

    return angle, error_px


# ============================================================
# REMOVE PERSON FROM MASK
# ============================================================
def remove_person_boxes_from_mask(mask, person_boxes, roi_y):
    h, w = mask.shape[:2]

    for box in person_boxes:
        x1, y1, x2, y2 = box

        ry1 = y1 - roi_y
        ry2 = y2 - roi_y

        padding = 45

        x1 = max(0, x1 - padding)
        x2 = min(w, x2 + padding)

        ry1 = max(0, ry1 - padding)
        ry2 = min(h, ry2 + padding)

        if ry2 > 0 and ry1 < h:
            cv2.rectangle(mask, (x1, ry1), (x2, ry2), 0, -1)

    return mask


# ============================================================
# ADAPTIVE BLACK MASK
# ============================================================
def create_black_line_mask(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    h, w = gray.shape[:2]
    total_pixels = h * w

    p_dark = np.percentile(gray, 7)

    base_thr = int(np.clip(p_dark + 12, 40, 100))

    best_mask = None
    best_thr = base_thr

    thresholds = [
        base_thr,
        base_thr - 8,
        base_thr - 16,
        base_thr - 24,
        base_thr - 32,
        base_thr + 8
    ]

    for thr in thresholds:
        thr = int(np.clip(thr, 35, 125))

        mask_gray = cv2.inRange(gray, 0, thr)

        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, min(thr + 10, 125)])

        mask_hsv = cv2.inRange(hsv, lower_black, upper_black)

        mask = cv2.bitwise_or(mask_gray, mask_hsv)

        kernel_close = np.ones((5, 5), np.uint8)
        kernel_open = np.ones((3, 3), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

        white_pixels = cv2.countNonZero(mask)
        ratio = white_pixels / total_pixels

        best_mask = mask
        best_thr = thr

        if MIN_MASK_RATIO <= ratio <= MAX_MASK_RATIO:
            break

    return best_mask, best_thr


# ============================================================
# LINE DETECTION USING SCAN BANDS
# ============================================================
def get_line_center(mask):
    h, w = mask.shape[:2]

    white_ratio = cv2.countNonZero(mask) / (h * w)

    if white_ratio > 0.22:
        return None, None, None

    scan_levels = [0.78, 0.70, 0.62, 0.54, 0.46, 0.38, 0.30, 0.22]

    best_cx = None
    best_cy = None

    for level in scan_levels:
        y = int(h * level)
        band = 14

        y1 = max(0, y - band)
        y2 = min(h, y + band)

        band_img = mask[y1:y2, :]

        col_sum = np.sum(band_img > 0, axis=0)
        active_cols = np.where(col_sum > 3)[0]

        if len(active_cols) < 8:
            continue

        groups = []

        start = active_cols[0]
        prev = active_cols[0]

        for x in active_cols[1:]:
            if x - prev > 6:
                groups.append((start, prev))
                start = x
            prev = x

        groups.append((start, prev))

        valid_groups = []

        for x_start, x_end in groups:
            width = x_end - x_start

            if width < 8:
                continue

            if width > 180:
                continue

            cx = (x_start + x_end) // 2

            valid_groups.append((width, cx, x_start, x_end))

        if not valid_groups:
            continue

        valid_groups.sort(key=lambda item: item[0], reverse=True)

        width, cx, x_start, x_end = valid_groups[0]

        best_cx = cx
        best_cy = y
        break

    if best_cx is None:
        return None, None, None

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best_contour = None

    if contours:
        min_dist = 999999

        for c in contours:
            area = cv2.contourArea(c)

            if area < 100:
                continue

            M = cv2.moments(c)

            if M["m00"] == 0:
                continue

            cx_c = int(M["m10"] / M["m00"])
            dist = abs(cx_c - best_cx)

            if dist < min_dist:
                min_dist = dist
                best_contour = c

    return best_cx, best_cy, best_contour


# ============================================================
# YOLO PERSON DETECTION
# ============================================================
def detect_persons(frame, display):
    person_boxes = []
    hazard_person_now = False

    small_frame = cv2.resize(frame, (320, 320))

    results = model(small_frame, verbose=False)

    scale_x = FRAME_WIDTH / 320
    scale_y = FRAME_HEIGHT / 320

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == 0 and conf >= PERSON_CONFIDENCE:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

                person_boxes.append((x1, y1, x2, y2))

                person_center = (x1 + x2) // 2

                if abs(person_center - FRAME_WIDTH // 2) < PERSON_CENTER_ZONE:
                    hazard_person_now = True
                    color = (0, 0, 255)
                    label = "PERSON STOP"
                else:
                    color = (255, 0, 255)
                    label = "PERSON SIDE"

                cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)

                cv2.putText(
                    display,
                    label,
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

    return person_boxes, hazard_person_now


# ============================================================
# MAIN LOOP
# ============================================================
while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera frame not received")
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    display = frame.copy()

    # ========================================================
    # PERSON DETECTION
    # ========================================================
    frame_count += 1

    if frame_count % YOLO_SKIP_FRAMES == 0:
        person_boxes, hazard_person_now = detect_persons(frame, display)

        active_person_boxes = person_boxes

        now = time.time()

        if hazard_person_now:
            emergency_stop = True
            last_person_seen_time = now
            clear_yolo_frames = 0

        else:
            if emergency_stop:
                if now - last_person_seen_time >= PERSON_CLEAR_DELAY:
                    clear_yolo_frames += 1

                    if clear_yolo_frames >= CLEAR_YOLO_FRAMES_REQUIRED:
                        emergency_stop = False
                        clear_yolo_frames = 0
                        reset_pid()

    # ========================================================
    # ROI
    # ========================================================
    roi_y = int(FRAME_HEIGHT * 0.40)
    roi = frame[roi_y:FRAME_HEIGHT, :]

    h, w = roi.shape[:2]
    center_x = w // 2

    # ========================================================
    # MASK
    # ========================================================
    mask, used_threshold = create_black_line_mask(roi)

    mask = remove_person_boxes_from_mask(
        mask,
        active_person_boxes,
        roi_y
    )

    best_cx, best_cy, line_contour = get_line_center(mask)

    # ========================================================
    # COMMAND LOGIC
    # ========================================================
    cmd = "NL"
    angle = 0.0
    error_px = 0.0

    if emergency_stop:
        cmd = "STOP"
        reset_pid()

    else:
        if best_cx is not None:
            raw_error = float(best_cx - center_x)

            angle, error_px = pid_to_angle(raw_error)

            cmd = f"A:{angle:.1f}"

        else:
            cmd = "NL"
            reset_pid()

    # ========================================================
    # DRAW ROI
    # ========================================================
    if line_contour is not None:
        cv2.drawContours(roi, [line_contour], -1, (0, 255, 0), 2)

    if best_cx is not None:
        cv2.circle(roi, (best_cx, best_cy), 8, (0, 0, 255), -1)

    cv2.line(
        roi,
        (center_x, 0),
        (center_x, h),
        (255, 0, 0),
        2
    )

    cv2.line(
        roi,
        (0, int(h * 0.05)),
        (w, int(h * 0.05)),
        (0, 255, 255),
        2
    )

    cv2.line(
        roi,
        (0, int(h * 0.98)),
        (w, int(h * 0.98)),
        (0, 255, 255),
        2
    )

    # ========================================================
    # DISPLAY TEXT
    # ========================================================
    white_pixels = cv2.countNonZero(mask)
    mask_ratio = white_pixels / (h * w)

    cv2.putText(
        display,
        f"CMD: {cmd}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display,
        f"ERR: {error_px:.0f}  ANG: {angle:.1f}",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display,
        f"MASK: {white_pixels}  RATIO: {mask_ratio:.2f}  THR: {used_threshold}",
        (10, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    if emergency_stop:
        cv2.putText(
            display,
            "EMERGENCY STOP",
            (10, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

        cv2.putText(
            display,
            "Waiting until path is clear...",
            (10, 195),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2
        )

    # ========================================================
    # SEND SERIAL
    # ========================================================
    now_send = time.time()

    if use_serial and (now_send - last_send_time >= SERIAL_SEND_INTERVAL):
        ser.write((cmd + "\n").encode())
        last_send_time = now_send

    # ========================================================
    # SHOW WINDOWS
    # ========================================================
    cv2.imshow("Frame", display)
    cv2.imshow("ROI", roi)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1)

    if key == 27:
        break

# ============================================================
# CLEANUP
# ============================================================
cap.release()

if use_serial:
    ser.write(("STOP\n").encode())
    ser.close()

cv2.destroyAllWindows()