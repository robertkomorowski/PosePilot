# main.py
import cv2
import numpy as np
import time
import platform
from utils.audio import _play_dir_sound_async, SoundRateLimiter
from utils.marker_utils import is_centered
from utils.calibration import load_calibration
from utils.draw_feedback import (
    put_text, put_text_bg, text_scale, show_and_continue, draw_arrows, draw_rotation_arrow
)
from utils.window_fit import make_window_same_percent
from config import *

import utils.bed_detection as beddet


def main():
    # ---------------- Gatekeeping-Phasen ----------------
    PH_CENTER = "CENTER"   # Zentrieren
    PH_DIST   = "DIST"     # Abstand/Höhe
    PH_CALIB  = "CALIB"    # LAB-Kalibrierung (optional, wenn noch nicht kalibriert)
    PH_ALIGN  = "ALIGN"    # Z-Rotation/Parallelität

    phase = PH_CENTER
    

    # ---------------- Zustände & Flags ----------------
    align_start = None             # Eintritt in Stabilitäts-Haltephase (Ausrichtung)
    last_sound = 0.0               # globales Rate-Limit (Sek.) für Sprachhinweise
    hold_last_sound = 0.0          # Laufvariable für "warte_nach_ausrichtung"
    sound_done = False             # finaler 'ausgerichtet' nur einmal
    sound_block_until = 0.0        # blockiert neue Sounds bis zu dieser Zeit (nach Kalibrierung)
    center_last_sound = 0.0        # Laufvariable Zentrier-Voice-Prompts
    calib_last_sound = 0.0         # Laufvariable Kalibrierungs-Sounds (falls benötigt)

    calib_hint_done = False        # "Bitte warten – Kalibrierung läuft" nur 1× pro Session (optional)
    pre_sound_played = False       # 'warte_vor_kalibrierung' einmalig pro Kalibrier-Versuch (optional)
    lab_sound_started = False      # 'kalibrieren_lab' einmalig beim Start der Kalibrierframes

    # Zähler für Gatekeeping
    center_ok_cnt = 0
    dist_ok_cnt   = 0

    # Kalibrierstatus
    calibrated = False
    warmup_cnt = 0
    calib_buf = []
    calib_polys = []
    ema_lab = None
    thr = float(LAB_THR)

    # Kamera öffnen
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_ANY)

    if not cap.isOpened():
        print("Kamera-Error: konnte nicht geöffnet werden")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # ArUco-Setup
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    objp = np.array([
        [-MARKER_LENGTH_M/2,  MARKER_LENGTH_M/2, 0],
        [ MARKER_LENGTH_M/2,  MARKER_LENGTH_M/2, 0],
        [ MARKER_LENGTH_M/2, -MARKER_LENGTH_M/2, 0],
        [-MARKER_LENGTH_M/2, -MARKER_LENGTH_M/2, 0]
    ], dtype=np.float32)

    # Fenster
    cv2.namedWindow("Ausrichtung - Live", cv2.WINDOW_NORMAL)
    make_window_same_percent("Ausrichtung - Live")

    # Kamera-Calibration laden
    cam_mtx, dist_coefs = load_calibration()

    # --- Limiter: beide Phasen (Center/Align) nutzen dasselbe Intervall ---
    limiter_dirs_center = SoundRateLimiter(min_interval_sec=PHASE_SOUND_INTERVAL_SEC)
    limiter_dirs_align  = SoundRateLimiter(min_interval_sec=PHASE_SOUND_INTERVAL_SEC)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if ROTATE_90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        seg_src = frame.copy()
        h, w = frame.shape[:2]
        scale = text_scale(h, base=0.5)
        now = time.time()

        # Quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # --------- Marker erkennen ---------
        gray = cv2.cvtColor(seg_src, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        img_ctr = (w // 2, h // 2)
        cv2.circle(frame, img_ctr, 5, (0, 255, 255), -1)

        aruco_quad = None
        if ids is not None and len(corners) > 0:
            quads = [c.reshape(4, 2).astype(np.float32) for c in corners]
            centers = [q.mean(axis=0) for q in quads]
            dists = [np.hypot(c[0] - img_ctr[0], c[1] - img_ctr[1]) for c in centers]
            best_i = int(np.argmin(dists))
            aruco_quad = quads[best_i]

        if aruco_quad is None:
            # Kein Marker -> alles zurück auf Start
            show_and_continue(frame, "Kein Marker erkannt")
            cv2.imshow('Ausrichtung - Live', frame)

            phase = PH_CENTER
            center_ok_cnt = 0
            dist_ok_cnt = 0
            warmup_cnt = 0
            calib_buf.clear(); calib_polys.clear()
            lab_sound_started = False
            pre_sound_played = False
            align_start = None
            sound_done = False
            hold_last_sound = 0.0
            center_last_sound = 0.0
            # Kalibrierung bleibt erhalten; ohne Marker geht's nicht weiter
            continue

        # Visualisierung + Zentrier-Check
        pts = aruco_quad.astype(int)
        m_ctr = tuple(pts.mean(axis=0).astype(int))
        cv2.circle(frame, m_ctr, 5, (255, 0, 0), -1)

        centered, dirs = is_centered(m_ctr, frame.shape)

        # Pose / Abstand (z)
        img_pts = aruco_quad.astype(np.float32)
        ok_pnp, rvec, tvec = cv2.solvePnP(objp, img_pts, cam_mtx, dist_coefs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok_pnp:
            rvec, tvec = cv2.solvePnPRefineLM(objp, img_pts, cam_mtx, dist_coefs, rvec, tvec)
        z = (float(np.ravel(tvec)[2]) + CALIB_OFFSET_M) if ok_pnp else None
        z_ok = (z is not None) and (abs(z - TARGET_DISTANCE_M) <= DISTANCE_TOLERANCE_M)

        # ======== PHASE 1: ZENTRIEREN ========
        if phase == PH_CENTER:
            if not centered:

                can_play_center = ((now - center_last_sound) >= PHASE_SOUND_INTERVAL_SEC) and (now >= sound_block_until)
                draw_arrows(
                    frame, dirs, w, h,
                    play_sounds=can_play_center,
                    limiter=limiter_dirs_center
                )
                if can_play_center:
                    center_last_sound = now
                    last_sound = now  # globaler Abstand, verhindert Übersprechen mit anderen Prompts

                put_text(frame, "Kamera zum Marker zentrieren", (30, int(40 * scale)), scale, (0, 0, 255))
                center_ok_cnt = 0
            else:
                center_ok_cnt += 1
                put_text(frame, "Stabil halten", (30, int(40 * scale)), scale, (0, 200, 255))
                if center_ok_cnt >= CENTER_STABLE_FRAMES:
                    phase = PH_DIST
                    dist_ok_cnt = 0  # frisch für Abstand
            cv2.imshow('Ausrichtung - Live', frame)
            continue

        # ======== PHASE 2: ABSTAND/HÖHE ========
        if phase == PH_DIST:
            # Rücksprung falls Zentrierung verloren
            if not centered:
                phase = PH_CENTER
                center_ok_cnt = 0
                cv2.imshow('Ausrichtung - Live', frame)
                continue

            if z is None:
                put_text(frame, "Abstand wird ermittelt …", (30, int(40 * scale)), scale, (0, 200, 255))
                dist_ok_cnt = 0
            elif not z_ok:
                put_text(
                    frame,
                    f"{'Zu nah' if z < TARGET_DISTANCE_M else 'Zu weit'}: {z:.2f} m (Ziel: {TARGET_DISTANCE_M:.2f} m)",
                    (30, int(40 * scale)), scale, (0, 0, 255)
                )
                dist_ok_cnt = 0
            else:
                dist_ok_cnt += 1
                put_text(frame, "Stabil halten", (30, int(40 * scale)), scale, (0, 200, 255))
                if dist_ok_cnt >= DIST_STABLE_FRAMES:
                    # Wenn schon kalibriert, direkt zu ALIGN; sonst zur Kalibrierung
                    if calibrated:
                        phase = PH_ALIGN
                        align_start = None
                        sound_done = False
                        hold_last_sound = 0.0
                    else:
                        phase = PH_CALIB
                        warmup_cnt = 0
                        calib_buf.clear(); calib_polys.clear()
                        pre_sound_played = False
                        lab_sound_started = False
                        calib_last_sound = 0.0

            cv2.imshow('Ausrichtung - Live', frame)
            continue

        # ======== PHASE 3: LAB-KALIBRIERUNG ========
        if phase == PH_CALIB:
            # Gates müssen weiterhin gehalten werden, sonst Neustart
            if not centered:
                phase = PH_CENTER
                center_ok_cnt = 0
                warmup_cnt = 0
                calib_buf.clear(); calib_polys.clear()
                cv2.imshow('Ausrichtung - Live', frame)
                continue
            if not z_ok:
                phase = PH_DIST
                dist_ok_cnt = 0
                warmup_cnt = 0
                calib_buf.clear(); calib_polys.clear()
                cv2.imshow('Ausrichtung - Live', frame)
                continue

            # Start-Hinweis sofort beim Eintritt in PH_CALIB (ohne Übersprechen)
            if not lab_sound_started:
                _play_dir_sound_async('kalibrieren_lab')
                lab_sound_started = True
                last_sound = now
                sound_block_until = now + 1.0  # kurze Schutzpause gegen Übersprechen

            # Warm-up (ohne Zähleranzeige)
            if warmup_cnt < WARMUP_FRAMES:
                warmup_cnt += 1
                put_text(frame, "Kalibrierung läuft – stabil halten", (30, int(40 * scale)), scale, (0, 200, 255))
                cv2.imshow('Ausrichtung - Live', frame)
                continue
            else:
                # Frames sammeln (Markerfläche ausschließen)
                calib_buf.append(seg_src.copy())
                calib_polys.append(aruco_quad.copy())

                put_text(frame, "Kalibrierung läuft – stabil halten", (30, int(40 * scale)), scale, (0, 200, 255))

                if len(calib_buf) >= CALIB_FRAMES:
                    center_lab = beddet.auto_calibrate_lab_center(
                        calib_buf, exclusion_polys_per_frame=calib_polys
                    )
                    ema_lab = center_lab.copy()
                    calibrated = True
                    calib_buf.clear(); calib_polys.clear()
                    put_text_bg(frame, "Kalibrierung abgeschlossen", (30, int(80 * scale)),
                                scale, (255, 255, 255), bg=(0, 180, 0))
                    # Cooldown nach Kalibrierton -> verhindert Überlappung mit Drehhinweisen
                    sound_block_until = time.time()
                    # Weiter zur Ausrichtung
                    phase = PH_ALIGN
                    align_start = None
                    sound_done = False
                    hold_last_sound = 0.0

                cv2.imshow('Ausrichtung - Live', frame)
                continue

        # ======== PHASE 4: Z-ROTATION / AUSRICHTUNG ========
        # Vor dem Rechnen: Gates erneut prüfen – bei Verlust zurück an den Start
        if not centered or not z_ok:
            phase = PH_CENTER
            center_ok_cnt = 0
            dist_ok_cnt = 0
            warmup_cnt = 0
            lab_sound_started = False
            pre_sound_played = False
            align_start = None
            sound_done = False
            hold_last_sound = 0.0
            center_last_sound = 0.0
            put_text_bg(frame,
                        "Zentrierung/Abstand verloren – bitte neu ausrichten",
                        (30, int(80 * scale)), scale, (255, 255, 255), bg=(0, 0, 255))
            cv2.imshow('Ausrichtung - Live', frame)
            continue

        # Nur hier wird segmentiert/ausgerichtet.
        lab = cv2.cvtColor(seg_src, cv2.COLOR_BGR2Lab)

        # Zentrales Patch mit ArUco-Exclusion messen + EMA-Glättung
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
        ema_lab = (1.0 - float(LAB_EMA_ALPHA)) * ema_lab + float(LAB_EMA_ALPHA) * center_now

        # ΔE + Chroma + Morph + Guards
        mask = beddet.segment_by_lab_distance(lab, ema_lab, thr, wL=0.25, min_chroma=10.0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        mask = beddet.trim_border_if_touching(mask, frac=0.01, px_min=4)
        cx1, cy1, cx2, cy2 = beddet._center_patch_rect(lab.shape, CENTER_FRAC_W, CENTER_FRAC_H)
        mask = beddet.keep_component_touching_center(mask, (cx1, cy1, cx2, cy2))
        mask = beddet.postprocess_mask(mask)

        H, W = seg_src.shape[:2]
        min_draw_area = float(MIN_DRAW_AREA_FRAC) * (H * W)
        mask_for_rect = beddet.rect_guard_adaptive(
            mask, top_frac=0.04, side_frac=0.02, bottom_frac=0.02, min_px=6, inner_margin_frac=0.01
        )
        rect, area = beddet.min_area_rect_border_safe(
            mask_for_rect,
            inset_px=EDGE_INSET_PX,
            pad_px=40,
            use_hull_for_rect=False,
            extra_inset_if_touch=30
        )

        vis = frame.copy()

        if rect is not None and area >= min_draw_area:
            ang_long = beddet.long_side_angle_deg(rect)
            err = beddet.alignment_error(ang_long, seg_src.shape, ALIGN_MODE)
            aligned = abs(err) <= ANGLE_TOLERANCE_DEG

            color = (0, 200, 0) if aligned else (0, 0, 255)
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.drawContours(vis, [box], 0, color, 3)

            now = time.time()
            if aligned:
                if align_start is None:
                    align_start = now
                    sound_done = False

                stable = (now - align_start) >= STABLE_ALIGN_SEC
                if stable:
                    put_text(vis, "AUSGERICHTET", (30, int(80 * scale)), scale, (0, 200, 0))
                    if not sound_done and (now - last_sound) > SOUND_INTERVAL_SEC and now >= sound_block_until:
                        _play_dir_sound_async('ausgerichtet')
                        sound_done = True
                        last_sound = now
                else:
                    put_text(vis, "Bitte nicht bewegen – Stabilitätsprüfung läuft",
                             (30, int(80 * scale)), scale, (0, 200, 255))
                    if (now - hold_last_sound) > PHASE_SOUND_INTERVAL_SEC and now >= sound_block_until:
                        last_sound = now
                        _play_dir_sound_async('warte_nach_ausrichtung')
                        hold_last_sound = now
            else:
                align_start = None
                sound_done = False
                # Drehrichtungshinweis
                dir_txt = "Bitte rechts drehen" if err > 0 else "Bitte links drehen"
                put_text_bg(vis, dir_txt, (30, int(80 * scale)), scale,
                            (255, 255, 255), bg=(0, 0, 255))
                radius = max(60, min(W, H) // 4)
                draw_rotation_arrow(
                    vis, (W // 2, H // 2),
                    radius=radius, clockwise=(err > 0),
                    color=(0, 0, 255), thickness=6, sweep_deg=200,
                    play_sound=(time.time() >= sound_block_until),
                    limiter=limiter_dirs_align
                )

        else:
            # In ALIGN ohne gültige Maske -> Info (Gates sind oben schon geprüft)
            put_text_bg(vis, "Kein Bett erkannt", (30, int(80 * scale)), scale,
                        (255, 255, 255), bg=(0, 0, 255))

        # Anzeige
        cv2.imshow('Ausrichtung - Live', vis)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
