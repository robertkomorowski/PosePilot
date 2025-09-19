# validation.py
# -*- coding: utf-8 -*-
"""
Validation (1 Run): läuft wie main, zeigt die normale UI/Sounds,
speichert aber genau ZWEI Ereignisse in runs/events.csv + Screenshots in runs/shots/:

  (1) centered        -> Screenshot + CSV (Z-Felder leer)
  (2) angle_ok (stabil AUSGERICHTET) -> Screenshot + CSV mit:
        - theta_main_horiz_deg   (aus Code)
        - theta_main_vert_deg    (aus Code, gegen Vertikale)
        - theta_from_screenshot_vert_deg (aus Bild gemessen)
        - theta_delta_deg        (Bild - Code)

Alle Z/Höhen-Felder bleiben LEER.

CSV-Felder (kompatibel zu capture_milestones.py):
  run_id, case, event, t_s, frame, screenshot,
  dx_px, dy_px, dist_center_px,
  z_m, z_gt_m, z_error_m,
  theta_main_horiz_deg, theta_main_vert_deg,
  theta_from_screenshot_vert_deg, theta_delta_deg,
  rect_fill, area_px, proc_ms, fps
"""

import os, csv, cv2, math, time, platform
from datetime import datetime
from collections import deque
import numpy as np

from config import *
import utils.bed_detection as beddet
from utils.calibration import load_calibration
from utils.marker_utils import is_centered
from utils.draw_feedback import (
    put_text, put_text_bg, text_scale, show_and_continue,
    draw_arrows, draw_rotation_arrow
)
from utils.window_fit import make_window_same_percent
from utils.audio import _play_dir_sound_async, SoundRateLimiter


# ------------------------ Pfade/Parameter ------------------------
CSV_PATH   = "runs/events.csv"
SHOTS_DIR  = "runs/shots"
WINDOW_NAME = "Ausrichtung - Live"

CAM_INDEX  = 0
FRAME_W    = 1920
FRAME_H    = 1080
FPS_REQ    = 30

CENTER_STABLE_FRAMES = 8     # wie main: vor Wechsel in Abstand
DIST_STABLE_FRAMES   = 8     # wie main: vor Kalibrierung/Align
SOUND_COOLDOWN_AFTER_CALIB_SEC = 2.5  # gegen Übersprechen

# ------------------------ Helfer ------------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return os.path.abspath(p)

def fold_angle_deg(a: float) -> float:
    while a > 90.0:  a -= 180.0
    while a < -90.0: a += 180.0
    return a

def angle_to_vertical_deg(angle_long_deg: float) -> float:
    # Winkel gegen die VERTIKALE (linke Bildkante)
    return fold_angle_deg(angle_long_deg - 90.0)

def measure_angle_from_screenshot(img_path: str) -> float | None:
    """
    Miss Winkel gegen VERTIKALE anhand der farbigen Overlay-Kontur (grün/rot) im Screenshot.
    Erwartet, dass in 'vis' die Bett-Box farbig gezeichnet wurde.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # grün + rot maskieren
    green = cv2.inRange(hsv, (35,150,150), (85,255,255))
    red1  = cv2.inRange(hsv, (0,150,150),  (10,255,255))
    red2  = cv2.inRange(hsv, (170,150,150),(179,255,255))
    mask  = cv2.bitwise_or(green, cv2.bitwise_or(red1, red2))

    edges = cv2.Canny(mask, 50, 150, apertureSize=3, L2gradient=True)
    H, W  = img.shape[:2]
    min_len = max(H, W) // 6
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                            minLineLength=min_len, maxLineGap=20)
    if lines is None:
        return None
    # längste Linie nehmen
    best = None; maxlen = 0.0
    for l in lines:
        x1,y1,x2,y2 = l[0]
        length = math.hypot(x2-x1, y2-y1)
        if length > maxlen:
            maxlen = length
            best = (x1,y1,x2,y2)
    if best is None:
        return None
    x1,y1,x2,y2 = best
    ang_horiz = math.degrees(math.atan2(y2 - y1, x2 - x1))
    ang_horiz = fold_angle_deg(ang_horiz)
    # gegen Vertikale:
    return fold_angle_deg(ang_horiz - 90.0)

# ------------------------ Hauptlauf ------------------------

def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    ensure_dir(os.path.dirname(CSV_PATH))
    ensure_dir(SHOTS_DIR)

    # CSV (append)
    fields = [
        "run_id","case","event","t_s","frame","screenshot",
        "dx_px","dy_px","dist_center_px",
        "z_m","z_gt_m","z_error_m",
        "theta_main_horiz_deg","theta_main_vert_deg",
        "theta_from_screenshot_vert_deg","theta_delta_deg",
        "rect_fill","area_px","proc_ms","fps"
    ]
    need_header = (not os.path.exists(CSV_PATH)) or (os.path.getsize(CSV_PATH)==0)
    f_csv = open(CSV_PATH, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_csv, fieldnames=fields)
    if need_header:
        writer.writeheader()

    # Kamera
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW if platform.system()=="Windows" else cv2.CAP_ANY)
    if not cap.isOpened():
        print("Kamera-Error: konnte nicht geöffnet werden.")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS,          FPS_REQ)

    # Kalibrierung
    try:
        cam_mtx, dist_coefs = load_calibration()
    except Exception as e:
        print(f"Kalibrier-Error: {e}")
        return 2

    # ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    objp = np.array([
        [-MARKER_LENGTH_M/2,  MARKER_LENGTH_M/2, 0],
        [ MARKER_LENGTH_M/2,  MARKER_LENGTH_M/2, 0],
        [ MARKER_LENGTH_M/2, -MARKER_LENGTH_M/2, 0],
        [-MARKER_LENGTH_M/2, -MARKER_LENGTH_M/2, 0]
    ], dtype=np.float32)

    # GUI
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    make_window_same_percent("Ausrichtung - Live")

    # LAB / Pipeline
    calibrated = False
    ema_lab = None
    thr = float(LAB_THR)
    K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

    # Gatekeeping-Zustände
    phase = "CENTER"  # CENTER -> DIST -> (CALIB) -> ALIGN
    center_ok_cnt = 0
    dist_ok_cnt   = 0
    warmup_cnt    = 0
    calib_buf, calib_polys = [], []

    # Sounds
    limiter_dirs = SoundRateLimiter(min_interval_sec=3.0)
    last_sound = 0.0
    sound_block_until = 0.0
    lab_sound_started = False
    pre_sound_played  = False
    hold_sound_played = False
    sound_done        = False

    # FPS
    t0 = time.perf_counter()
    last = t0
    fps_buf = deque(maxlen=120)

    # Logging-Flags (nur EINMAL je Event)
    logged_centered = False
    logged_angle_ok = False

    # Für AUSGERICHTET (stabil)
    align_start = None

    frame_idx = 0

    def log_event(name: str, vis_img, *, dx=None, dy=None,
                  angle_long=None, theta_vert=None, theta_shot=None,
                  rect_fill=None, area_px=None, proc_ms=None):
        shot_name = f"{run_id}_{name}.png"
        shot_path = os.path.join(SHOTS_DIR, shot_name)
        cv2.imwrite(shot_path, vis_img)

        # Falls kein theta_shot übergeben: vom soeben gespeicherten Bild messen
        if theta_shot is None:
            theta_shot = measure_angle_from_screenshot(shot_path)

        # Delta nur wenn beides vorhanden
        theta_delta_str = ""
        if (theta_shot is not None) and (theta_vert is not None):
            theta_delta_str = f"{(theta_shot - theta_vert):+.2f}"

        writer.writerow({
            "run_id": run_id,
            "case": "",
            "event": name,
            "t_s": f"{time.perf_counter() - t0:.3f}",
            "frame": frame_idx,
            "screenshot": os.path.relpath(shot_path).replace("\\","/"),
            "dx_px": f"{dx:.2f}" if dx is not None else "",
            "dy_px": f"{dy:.2f}" if dy is not None else "",
            "dist_center_px": f"{math.hypot(dx,dy):.2f}" if (dx is not None and dy is not None) else "",
            # Höhe/Abstand bewusst LEER lassen
            "z_m": "",
            "z_gt_m": "",
            "z_error_m": "",
            # Winkel
            "theta_main_horiz_deg": f"{angle_long:+.2f}" if angle_long is not None else "",
            "theta_main_vert_deg":  f"{theta_vert:+.2f}" if theta_vert  is not None else "",
            "theta_from_screenshot_vert_deg": f"{theta_shot:+.2f}" if theta_shot is not None else "",
            "theta_delta_deg": theta_delta_str,
            # Form
            "rect_fill": f"{rect_fill:.3f}" if rect_fill is not None else "",
            "area_px": f"{area_px:.0f}" if area_px is not None else "",
            "proc_ms": f"{proc_ms:.1f}" if proc_ms is not None else "",
            "fps": f"{np.mean(fps_buf):.1f}" if len(fps_buf)>0 else ""
        })
        f_csv.flush()

        return shot_path

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t_read = time.perf_counter()
            if ROTATE_90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            H, W = frame.shape[:2]
            scale = text_scale(H, base=0.5)

            # FPS
            dt = max(1e-6, t_read - last); last = t_read
            inst_fps = 1.0 / dt
            fps_buf.append(inst_fps)

            # Quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            seg_src = frame.copy()

            # Marker
            gray = cv2.cvtColor(seg_src, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            img_ctr = (W//2, H//2)
            cv2.circle(frame, img_ctr, 5, (0,255,255), -1)

            aruco_quad = None
            if ids is not None and len(corners) > 0:
                quads = [c.reshape(4,2).astype(np.float32) for c in corners]
                centers = [q.mean(axis=0) for q in quads]
                dists = [np.hypot(c[0]-img_ctr[0], c[1]-img_ctr[1]) for c in centers]
                best_i = int(np.argmin(dists))
                aruco_quad = quads[best_i]

            if aruco_quad is None:
                # Reset auf Start
                show_and_continue(frame, "Kein Marker erkannt")
                cv2.imshow(WINDOW_NAME, frame)
                phase = "CENTER"
                center_ok_cnt = 0
                dist_ok_cnt = 0
                warmup_cnt = 0
                calib_buf.clear(); calib_polys.clear()
                lab_sound_started = False
                pre_sound_played = False
                align_start = None
                hold_sound_played = False
                sound_done = False
                # wir laufen weiter – Logs bleiben, aber angle_ok kann dann nicht kommen
                frame_idx += 1
                continue

            # Zentrierung / Abstand
            pts = aruco_quad.astype(int)
            m_ctr = tuple(pts.mean(axis=0).astype(int))
            cv2.circle(frame, m_ctr, 5, (255,0,0), -1)

            centered, dirs = is_centered(m_ctr, frame.shape)

            # dx/dy für Logging
            dx = float(m_ctr[0] - img_ctr[0])
            dy = float(m_ctr[1] - img_ctr[1])

            # z via PnP (nur Gatekeeping, nicht loggen)
            z = None
            img_pts = aruco_quad.astype(np.float32)
            okp, rvec, tvec = cv2.solvePnP(objp, img_pts, cam_mtx, dist_coefs,
                                           flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if okp:
                rvec, tvec = cv2.solvePnPRefineLM(objp, img_pts, cam_mtx, dist_coefs, rvec, tvec)
                z = float(tvec[2]) + CALIB_OFFSET_M
            z_ok = (z is not None) and (abs(z - TARGET_DISTANCE_M) <= DISTANCE_TOLERANCE_M)

            # ---------------- PHASEN wie main ----------------

            # PHASE CENTER
            if phase == "CENTER":
                if not centered:
                    draw_arrows(frame, dirs, W, H, play_sounds=True, limiter=limiter_dirs)
                    put_text(frame, "Kamera zum Marker zentrieren", (30, int(40*scale)), scale, (0,0,255))
                    center_ok_cnt = 0
                else:
                    center_ok_cnt += 1
                    put_text(frame, "Stabil halten", (30, int(40*scale)), scale, (0,200,255))
                    if center_ok_cnt >= CENTER_STABLE_FRAMES:
                        phase = "DIST"
                        dist_ok_cnt = 0

                        # ---- EVENT (1) CENTERED LOGGEN (nur 1x) ----
                        if not logged_centered:
                            vis_c = frame.copy()
                            log_event(
                                "centered", vis_c,
                                dx=dx, dy=dy,
                                angle_long=None, theta_vert=None, theta_shot=None,
                                rect_fill=None, area_px=None, proc_ms=None
                            )
                            logged_centered = True

                cv2.imshow(WINDOW_NAME, frame)
                frame_idx += 1
                continue

            # PHASE DIST
            if phase == "DIST":
                if not centered:
                    phase = "CENTER"
                    center_ok_cnt = 0
                    cv2.imshow(WINDOW_NAME, frame)
                    frame_idx += 1
                    continue

                if z is None:
                    put_text(frame, "Abstand wird ermittelt …", (30, int(40*scale)), scale, (0,200,255))
                    dist_ok_cnt = 0
                elif not z_ok:
                    txt = f"{'Zu nah' if z < TARGET_DISTANCE_M else 'Zu weit'}: {z:.2f} m (Ziel: {TARGET_DISTANCE_M:.2f} m)"
                    put_text(frame, txt, (30, int(40*scale)), scale, (0,0,255))
                    dist_ok_cnt = 0
                else:
                    dist_ok_cnt += 1
                    put_text(frame, "Stabil halten", (30, int(40*scale)), scale, (0,200,255))
                    if dist_ok_cnt >= DIST_STABLE_FRAMES:
                        # Entweder direkt ALIGN (wenn schon kalibriert) oder CALIB
                        phase = "ALIGN" if calibrated else "CALIB"
                        warmup_cnt = 0
                        calib_buf.clear(); calib_polys.clear()
                        # Vor-Kalibrierungssound
                        if (time.time() - last_sound) > SOUND_INTERVAL_SEC and not pre_sound_played and not calibrated:
                            _play_dir_sound_async('warte_vor_kalibrierung')
                            last_sound = time.time()
                            pre_sound_played = True

                cv2.imshow(WINDOW_NAME, frame)
                frame_idx += 1
                continue

            # PHASE CALIB (nur falls noch nicht kalibriert)
            if phase == "CALIB":
                # Gates sichern
                if not centered:
                    phase = "CENTER"; center_ok_cnt = 0
                    cv2.imshow(WINDOW_NAME, frame); frame_idx += 1; continue
                if not z_ok:
                    phase = "DIST"; dist_ok_cnt = 0
                    cv2.imshow(WINDOW_NAME, frame); frame_idx += 1; continue

                if warmup_cnt < WARMUP_FRAMES:
                    warmup_cnt += 1
                    put_text(frame, "Kalibrierung läuft – stabil halten", (30, int(40*scale)), scale, (0,200,255))
                    cv2.imshow(WINDOW_NAME, frame)
                    frame_idx += 1
                    continue
                else:
                    if not lab_sound_started and (time.time() - last_sound) > SOUND_INTERVAL_SEC:
                        _play_dir_sound_async('kalibrieren_lab')
                        last_sound = time.time()
                        lab_sound_started = True

                    calib_buf.append(seg_src.copy())
                    calib_polys.append(aruco_quad.copy())
                    put_text(frame, f"Kalibrierung läuft – stabil halten", (30, int(40*scale)), scale, (0,200,255))

                    if len(calib_buf) >= CALIB_FRAMES:
                        center_lab = beddet.auto_calibrate_lab_center(
                            calib_buf, exclusion_polys_per_frame=calib_polys
                        )
                        ema_lab = center_lab.copy()
                        calibrated = True
                        calib_buf.clear(); calib_polys.clear()
                        put_text_bg(frame, "Kalibrierung abgeschlossen", (30, int(80*scale)),
                                    scale, (255,255,255), bg=(0,180,0))
                        sound_block_until = time.time() + SOUND_COOLDOWN_AFTER_CALIB_SEC
                        phase = "ALIGN"
                        align_start = None
                        hold_sound_played = False
                        sound_done = False

                cv2.imshow(WINDOW_NAME, frame)
                frame_idx += 1
                continue

            # PHASE ALIGN
            # Gates prüfen
            if not centered or not z_ok:
                phase = "CENTER"
                center_ok_cnt = 0
                dist_ok_cnt = 0
                warmup_cnt = 0
                lab_sound_started = False
                pre_sound_played = False
                align_start = None
                hold_sound_played = False
                sound_done = False
                put_text_bg(frame, "Zentrierung/Abstand verloren – bitte neu ausrichten",
                            (30, int(80*scale)), scale, (255,255,255), bg=(0,0,255))
                cv2.imshow(WINDOW_NAME, frame)
                frame_idx += 1
                continue

            # Segmentierung & Rechteck
            lab = cv2.cvtColor(seg_src, cv2.COLOR_BGR2Lab)

            excl_polys = [aruco_quad] if aruco_quad is not None else None
            if aruco_quad is not None:
                side = beddet._avg_marker_side_px(aruco_quad)
                pad_px = int(max(2.0, ARUCO_EXCL_PAD_FRAC * side))
            else:
                pad_px = 0

            center_now = beddet.robust_center_lab_with_exclusions(
                lab, outer_frac_w=CENTER_FRAC_W, outer_frac_h=CENTER_FRAC_H,
                exclusion_polys=excl_polys, pad_px=pad_px
            )
            if ema_lab is None:
                ema_lab = center_now.copy()
            else:
                ema_lab = (1.0 - float(LAB_EMA_ALPHA)) * ema_lab + float(LAB_EMA_ALPHA) * center_now

            t_proc0 = time.perf_counter()
            mask = beddet.segment_by_lab_distance(lab, ema_lab, thr, wL=0.25, min_chroma=10.0)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, K5, iterations=1)
            mask = beddet.trim_border_if_touching(mask, frac=0.01, px_min=4)
            cx1, cy1, cx2, cy2 = beddet._center_patch_rect(lab.shape, CENTER_FRAC_W, CENTER_FRAC_H)
            mask = beddet.keep_component_touching_center(mask, (cx1, cy1, cx2, cy2))
            mask = beddet.postprocess_mask(mask)

            min_draw_area = float(MIN_DRAW_AREA_FRAC) * (H * W)
            mask_for_rect = beddet.rect_guard_adaptive(
                mask, top_frac=0.04, side_frac=0.02, bottom_frac=0.02, min_px=6, inner_margin_frac=0.01
            )
            rect, area = beddet.min_area_rect_border_safe(
                mask_for_rect, inset_px=EDGE_INSET_PX, pad_px=40,
                use_hull_for_rect=False, extra_inset_if_touch=30
            )
            t_proc1 = time.perf_counter()
            proc_ms = (t_proc1 - t_proc0) * 1000.0

            vis = frame.copy()

            if rect is not None and area >= min_draw_area:
                angle_long = float(beddet.long_side_angle_deg(rect))
                err = beddet.alignment_error(angle_long, seg_src.shape, ALIGN_MODE)
                aligned = abs(err) <= ANGLE_TOLERANCE_DEG

                color = (0,200,0) if aligned else (0,0,255)
                box = cv2.boxPoints(rect).astype(np.int32)
                cv2.drawContours(vis, [box], 0, color, 3)

                nowt = time.time()
                if aligned:
                    if align_start is None:
                        align_start = time.perf_counter()
                        hold_sound_played = False
                        sound_done = False

                    held = (time.perf_counter() - align_start) >= float(STABLE_ALIGN_SEC)
                    if not held:
                        put_text(vis, "Bitte nicht bewegen – Stabilitätsprüfung läuft",
                                 (30, int(80*scale)), scale, (0,200,255))
                        if (not hold_sound_played) and (nowt >= sound_block_until) and ((nowt - last_sound) > SOUND_INTERVAL_SEC):
                            _play_dir_sound_async('warte_nach_ausrichtung')
                            hold_sound_played = True
                            last_sound = nowt
                    else:
                        put_text(vis, "AUSGERICHTET", (30, int(80*scale)), scale, (0,200,0))
                        if (not sound_done) and (nowt - last_sound) > SOUND_INTERVAL_SEC and (nowt >= sound_block_until):
                            _play_dir_sound_async('ausgerichtet')
                            sound_done = True
                            last_sound = nowt

                        # ---- EVENT (2) ANGLE_OK LOGGEN (nur 1x) ----
                        if not logged_angle_ok:
                            theta_vert = float(angle_to_vertical_deg(angle_long))
                            # rect_fill/area für Doku
                            rect_fill = beddet.rect_mask_fill_ratio(mask, rect)
                            # Screenshot speichern & aus Bild den Winkel messen
                            shot_path = log_event(
                                "angle_ok", vis,
                                dx=dx, dy=dy,
                                angle_long=angle_long,
                                theta_vert=theta_vert,
                                theta_shot=None,   # wird aus dem gespeicherten Bild gemessen
                                rect_fill=rect_fill,
                                area_px=float(area),
                                proc_ms=proc_ms
                            )
                            logged_angle_ok = True

                            # Wenn beide Events vorhanden -> Ende
                            if logged_centered and logged_angle_ok:
                                cv2.imshow(WINDOW_NAME, vis)
                                cv2.waitKey(300)  # kurzes visuelles Feedback
                                break
                else:
                    # Drehrichtung + Sound
                    align_start = None
                    hold_sound_played = False
                    sound_done = False
                    rot_dir = "rechts" if err > 0 else "links"
                    put_text_bg(vis, f"Bitte {rot_dir} drehen", (30, int(80*scale)), scale,
                                (255,255,255), bg=(0,0,255))
                    radius = max(60, min(W, H) // 4)
                    draw_rotation_arrow(
                        vis, (W // 2, H // 2),
                        radius=radius, clockwise=(err > 0),
                        color=(0, 0, 255), thickness=6, sweep_deg=200,
                        play_sound=(time.time() >= sound_block_until),
                        limiter=limiter_dirs
                    )
            else:
                put_text_bg(vis, "Kein Bett erkannt", (30, int(80*scale)), scale,
                            (255,255,255), bg=(0,0,255))

            # Anzeige
            cv2.imshow(WINDOW_NAME, vis)
            frame_idx += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        f_csv.close()

    print(f"[OK] Validation abgeschlossen. CSV: {CSV_PATH}  |  Shots: {SHOTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
