import cv2
import numpy as np
import platform
import os
import threading
import time

# === Konfiguration ===
BEEP_INTERVAL_SEC = 10
CENTER_TOLERANCE_PX = 20
DISTANCE_TOLERANCE_M = 0.05
TARGET_DISTANCE_M = 0.6
CALIB_OFFSET_M = 0.25
MARKER_LENGTH_M = 0.078
ANGLE_TOLERANCE_DEG = 5
ARROW_LENGTH = 80
ARROW_THICKNESS = 5


def play_beep_once():
    try:
        if platform.system() == 'Windows':
            import winsound
            winsound.Beep(1000, 300)
        elif platform.system() == 'Darwin':
            os.system('say "ausgerichtet" &')
        else:
            os.system('paplay /usr/share/sounds/freedesktop/stereo/message.oga &')
    except Exception:
        print("Akustisches Signal konnte nicht abgespielt werden.")


def is_centered(marker_center, frame_shape):
    h, w = frame_shape[:2]
    img_ctr = np.array([w//2, h//2])
    mc = np.array(marker_center)
    diff = mc - img_ctr
    dirs = []
    if abs(diff[1]) > CENTER_TOLERANCE_PX:
        dirs.append('oben' if diff[1] < 0 else 'unten')
    if abs(diff[0]) > CENTER_TOLERANCE_PX:
        dirs.append('links' if diff[0] < 0 else 'rechts')
    return len(dirs) == 0, dirs


def segment_green_bed(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 60, 60])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def get_main_contour(mask, min_area=5000):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = mask.shape[:2]
    margin = 20  # Pixel Abstand vom Rand

    filtered = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        if y <= margin or y + ch >= h - margin:
            continue  # verwerfen, wenn zu nah am Rand
        filtered.append(cnt)

    if not filtered:
        return None
    return max(filtered, key=cv2.contourArea)



def get_bed_rotation_angle(contour):
    rect = cv2.minAreaRect(contour)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    return angle, rect


def draw_bed_edge(frame, rect):
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    cv2.drawContours(frame, [box], 0, (0,255,0), 3)


def load_calibration():
    path = os.path.join("data", "calibration", "intrinsics", "camera_calibration_data.npz")
    if not os.path.exists(path):
        raise FileNotFoundError("Kalibrierdatei nicht gefunden: " + path)
    data = np.load(path)
    return data['camera_matrix'], data['dist_coeffs']


def main():
    cam_mtx, dist_coefs = load_calibration()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera-Error")
        return

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    objp = np.array([[-MARKER_LENGTH_M/2, MARKER_LENGTH_M/2, 0],
                     [MARKER_LENGTH_M/2, MARKER_LENGTH_M/2, 0],
                     [MARKER_LENGTH_M/2, -MARKER_LENGTH_M/2, 0],
                     [-MARKER_LENGTH_M/2, -MARKER_LENGTH_M/2, 0]], dtype=np.float32)

    last_beep = 0
    beep_done = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        h, w = frame.shape[:2]

        img_ctr = (w//2, h//2)
        cv2.circle(frame, img_ctr, 5, (0,255,255), -1)

        if ids is None or len(corners) == 0:
            cv2.putText(frame, "Kein Marker erkannt", (30,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            cv2.imshow('Ausrichtung - Live', frame)
            if cv2.waitKey(1)&0xFF == ord('q'): break
            continue

        pts = corners[0].reshape(4,2).astype(int)
        m_ctr = tuple(pts.mean(axis=0).astype(int))
        cv2.circle(frame, m_ctr, 5, (255,0,0), -1)

        centered, dirs = is_centered(m_ctr, frame.shape)
        if not centered:
            cv2.putText(frame, "Positioniere Kamera mittig zum Marker", (30,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            for d in dirs:
                if d=='oben':   cv2.arrowedLine(frame, (w//2,100), (w//2,100-ARROW_LENGTH),   (0,0,255), ARROW_THICKNESS)
                if d=='unten': cv2.arrowedLine(frame, (w//2,h-100), (w//2,h-100+ARROW_LENGTH), (0,0,255), ARROW_THICKNESS)
                if d=='links': cv2.arrowedLine(frame, (100,h//2), (100-ARROW_LENGTH,h//2), (0,0,255), ARROW_THICKNESS)
                if d=='rechts':cv2.arrowedLine(frame, (w-100,h//2), (w-100+ARROW_LENGTH,h//2), (0,0,255), ARROW_THICKNESS)
            cv2.imshow('Ausrichtung - Live', frame)
            if cv2.waitKey(1)&0xFF == ord('q'): break
            continue

        img_pts = corners[0].reshape(4,2).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(objp, img_pts, cam_mtx, dist_coefs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok:
            rvec, tvec = cv2.solvePnPRefineLM(objp, img_pts, cam_mtx, dist_coefs, rvec, tvec)

        z = float(tvec[2])
        if abs(z - TARGET_DISTANCE_M) > DISTANCE_TOLERANCE_M:
            text = 'Zu nah' if z < TARGET_DISTANCE_M else 'Zu weit'
            cv2.putText(frame, f"{text}: {z:.2f} m (Ziel: {TARGET_DISTANCE_M:.2f} m)", (30,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            
            cv2.imshow('Ausrichtung - Live', frame)
            if cv2.waitKey(1)&0xFF == ord('q'): break
            continue
        else:
            cv2.putText(frame, f"Abstand korrekt: {z:.2f} m", (30,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,0), 2)

        green_mask = segment_green_bed(frame)
        # Debug-Ansicht für Maske und Konturen
        cv2.imshow('Maske (gruen)', green_mask)
        for cnt in cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            cv2.drawContours(frame, [cnt], -1, (255,0,255), 1)
        main_contour = get_main_contour(green_mask)

        if main_contour is not None:
            angle, rect = get_bed_rotation_angle(main_contour)
            draw_bed_edge(frame, rect)
            if abs(angle) < ANGLE_TOLERANCE_DEG:
                cv2.putText(frame, "AUSGERICHTET", (30,80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,200,0), 2)
                if not beep_done and time.time() - last_beep > BEEP_INTERVAL_SEC:
                    threading.Thread(target=play_beep_once).start()
                    beep_done = True
                    last_beep = time.time()
            else:
                dir_txt = 'links drehen' if angle > 0 else 'rechts drehen'
                cv2.putText(frame, f"Bitte {dir_txt} ({angle:.1f} Grad)", (30,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        else:
            cv2.putText(frame, "Keine Bettkante erkannt!", (30,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.imshow('Ausrichtung - Live', frame)
        if cv2.waitKey(1)&0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
