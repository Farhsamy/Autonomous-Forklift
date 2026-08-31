# camerakinect_aruco_pallet_follow.py (FIXED VERSION)

import os
import time
from collections import defaultdict, deque
from typing import Optional, Dict, Any, Tuple, List

import cv2
import numpy as np

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except Exception:
    serial = None
    SERIAL_AVAILABLE = False

from pykinect2 import PyKinectV2, PyKinectRuntime


FRAME_W = 640
FRAME_H = 360

ID_TO_LABEL = {1: "A", 2: "B", 3: "MC"}
DEFAULT_TARGET_LABEL = "A"

USE_SERIAL = True
SERIAL_PORT = "COM3"
SERIAL_BAUD = 9600
SERIAL_TIMEOUT = 0.05

STEER_PREFIX = "A:"
MAX_STEER_ANGLE = 25.0
STEER_KP = 0.085
CENTER_DEADBAND_PX = 30

SEARCH_WHEN_NO_TARGET = True
SEARCH_ANGLE = 18.0
STOP_COMMAND = "STOP\n"

ARUCO_DICT_NAME = "DICT_4X4_50"
MIN_MARKER_AREA = 250
ARUCO_CONFIRM_N = 2
TARGET_LOST_RESET_N = 8

WOOD_LOWER = np.array([5, 25, 35], dtype=np.uint8)
WOOD_UPPER = np.array([42, 235, 255], dtype=np.uint8)

WOOD2_LOWER = np.array([10, 15, 70], dtype=np.uint8)
WOOD2_UPPER = np.array([48, 170, 255], dtype=np.uint8)

MIN_CONTOUR_AREA = 180
PALLET_CONFIRM_N = 5
MISS_UNLOCK_N = 8

last_wood_mask = None
last_depth_mask = None
last_pallet_mask = None
last_roi_mask = None


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def normalize_label(x):
    return str(x).strip().upper()


class SerialController:
    def __init__(self, port, baud):
        self.port = port
        self.baud = baud
        self.ser = None
        self.last_cmd = None
        self.last_t = 0

    def connect(self):
        if not SERIAL_AVAILABLE:
            print("[SERIAL] pyserial missing")
            return
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=SERIAL_TIMEOUT)
            time.sleep(2)
            print("[SERIAL] Connected")
        except Exception as e:
            print("[SERIAL] FAIL:", e)
            self.ser = None

    def send(self, angle):
        angle = clamp(angle, -MAX_STEER_ANGLE, MAX_STEER_ANGLE)
        cmd = f"{STEER_PREFIX}{angle:.1f}\n"

        if self.ser:
            self.ser.write(cmd.encode())
        else:
            print("[CMD]", cmd.strip())

    def stop(self):
        self.send(0)

    def close(self):
        self.stop()
        if self.ser:
            self.ser.close()


class KinectReader:
    def __init__(self):
        self.kinect = PyKinectRuntime.PyKinectRuntime(
            PyKinectV2.FrameSourceTypes_Color |
            PyKinectV2.FrameSourceTypes_Depth
        )
        self.last_color = None
        self.last_depth = None

    def get(self):
        if self.kinect.has_new_color_frame():
            raw = self.kinect.get_last_color_frame()
            img = raw.reshape((1080, 1920, 4)).astype(np.uint8)
            img = img[:, :, :3]
            self.last_color = cv2.resize(img, (FRAME_W, FRAME_H))

        if self.kinect.has_new_depth_frame():
            raw = self.kinect.get_last_depth_frame()
            depth = raw.reshape((424, 512)).astype(np.uint16)
            self.last_depth = cv2.resize(depth, (FRAME_W, FRAME_H))

        return self.last_color, self.last_depth


def get_aruco_dictionary():
    return cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, ARUCO_DICT_NAME)
    )


def detect_markers(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dictionary = get_aruco_dictionary()

    params = cv2.aruco.DetectorParameters_create()

    corners, ids, _ = cv2.aruco.detectMarkers(
        gray, dictionary, parameters=params
    )

    out = []
    if ids is None:
        return out

    for c, i in zip(corners, ids.flatten()):
        pts = c.reshape(4, 2)
        x1, y1 = np.min(pts, axis=0)
        x2, y2 = np.max(pts, axis=0)

        out.append({
            "id": int(i),
            "label": ID_TO_LABEL.get(int(i)),
            "cx": int((x1 + x2) / 2),
            "cy": int((y1 + y2) / 2),
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
        })

    return out


def select_marker(markers, target):
    target = normalize_label(target)
    c = [m for m in markers if m["label"] == target]
    if not c:
        return None
    return max(c, key=lambda m: m["cx"])


def detect_pallet(frame, depth, marker):
    if frame is None or depth is None or marker is None:
        return None

    h, w = frame.shape[:2]
    x1, x2 = 0, w
    y1, y2 = marker["y2"], h

    roi = np.zeros((h, w), np.uint8)
    roi[y1:y2, x1:x2] = 255

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, WOOD_LOWER, WOOD_UPPER)
    mask2 = cv2.inRange(hsv, WOOD2_LOWER, WOOD2_UPPER)
    mask = cv2.bitwise_and(mask1 | mask2, roi)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_CONTOUR_AREA:
            continue

        x, y, w_, h_ = cv2.boundingRect(c)
        if y < marker["y2"]:
            continue

        if area > best_area:
            best_area = area
            best = {
                "cx": x + w_ // 2,
                "cy": y + h_ // 2
            }

    return best


def main():
    print("START")

    target = input("Target (A/B/MC): ") or DEFAULT_TARGET_LABEL

    cam = KinectReader()
    ctrl = SerialController(SERIAL_PORT, SERIAL_BAUD)
    ctrl.connect()

    cv2.namedWindow("view", cv2.WINDOW_NORMAL)

    try:
        while True:
            frame, depth = cam.get()

            if frame is None or depth is None:
                continue

            markers = detect_markers(frame)
            marker = select_marker(markers, target)

            pallet = detect_pallet(frame, depth, marker) if marker else None

            h, w = frame.shape[:2]
            cx = w // 2

            cmd = 0

            if pallet:
                err = pallet["cx"] - cx

                if abs(err) < CENTER_DEADBAND_PX:
                    cmd = 0
                else:
                    cmd = clamp(err * STEER_KP, -MAX_STEER_ANGLE, MAX_STEER_ANGLE)

                ctrl.send(cmd)
            else:
                cmd = SEARCH_ANGLE if SEARCH_WHEN_NO_TARGET else 0
                ctrl.send(cmd) if SEARCH_WHEN_NO_TARGET else ctrl.stop()

            cv2.imshow("view", frame)

            if cv2.waitKey(1) == 27:
                break

    finally:
        ctrl.close()
        cam.kinect.close()
        cv2.destroyAllWindows()
        print("DONE")


if __name__ == "__main__":
    main()