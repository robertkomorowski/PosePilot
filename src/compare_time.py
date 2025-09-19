# run_compare_like_main_onephoto.py
# -*- coding: utf-8 -*-
"""
Vergleich: guided (wie main) vs. intuitive (ohne Hinweise).
- 1:1-Ablauf wie main.py im guided-Modus (Phasen, Sounds, Gatekeeping).
- Pro Durchlauf wird GENAU EIN RAW-Screenshot gespeichert und NUR daraus ausgewertet.
- CSV enthält keine FPS/Systemwinkel.

Start:
  - Script fragt interaktiv: 1=guided, 2=intuitive.
  - Nach jedem Run: CSV+Bild -> Programm beendet (für nächsten Run neu starten).

Tasten:
  guided:   'q' beendet (sonst automatisch bei AUSGERICHTET)
  intuitive:'SPACE' markiert "ausgerichtet" -> Screenshot & Ende
            'q' beendet ohne Speicherung
"""

import os, csv, cv2, math, time, platform
from datetime import datetime
import numpy as np

from utils.audio import _play_dir_sound_async, SoundRateLimiter
from utils.marker_utils import is_centered
from utils.calibration import load_calibration
from utils.draw_feedback import (
    put_text, put_text_bg, text_scale, show_and_continue,
    draw_arrows, draw_rotation_arrow
)
from utils.window_fit import make_window_same_percent
from config import *
import utils.bed_detection as beddet


# ---------------- Pfade ----------------
CSV_PATH  = "runs/compare_onephoto.csv"
SHOTS_DIR = "runs/compare_onephoto_shots"
WINDOW    = "Ausrichtung - Live"


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return os.path.abspath(p)


# ---------------- Bildbasierte Auswertung (nur aus Screenshot) ----------------

def _fold_deg_90(a: float) -> float:
    while a > 90.0:   a -= 180.0
    while a <= -90.0: a += 180.0
    return a

def _angle_vertical_from_long(long_deg: float) -> float:
    # 0° = parallel zur linken Bildkante (Vertikale). Uhrzeigersinn positiv.
    return _fold_deg_90(long_deg - 90.0)

def analyze_screenshot(raw_path: str):
    """
    Auswertung rein aus dem RAW-Screenshot:
      - ArUco -> dx, dy, delta_c (Distanz zur Bildmitte)
      - Segmentierung + RotatedRect -> Winkel gegen Vertikale (theta_img_vert_deg)
      - Qualitätsmaße: rect_fill, area_px
    Rückgabe: dict (Floats oder None).
    """
    img = cv2.imread(raw_path)
    if img is None:
        return {"dx_px": None, "dy_px": None, "dist_center_px": None,
                "theta_img_vert_deg": None, "rect_fill": None, "area_px": None}

    H, W = img.shape[:2]
    img_ctr = (W//2, H//2)

    # --- ArUco für dx/dy/delta_c ---
    dx = dy = delta_c = None
    quad = None
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        detector   = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None and len(corners) > 0:
            quads   = [c.reshape(4,2).astype(np.float32) for c in corners]
            centers = [q.mean(axis=0) for q in quads]
            dists   = [np.hypot(c[0]-img_ctr[0], c[1]-img_ctr[1]) for c in centers]
            best_i  = int(np.argmin(dists))
            quad    = quads[best_i]
            m_ctr   = quad.mean(axis=0)
            dx = float(m_ctr[0] - img_ctr[0])
            dy = float(m_ctr[1] - img_ctr[1])
            delta_c = float(math.hypot(dx, dy))
    except Exception:
        pass

    # --- Winkel über Segmentierung (wie Live-Pfad, aber offline auf dem Einzelbild) ---
    theta_img_vert_deg = rect_fill = area_px = None
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

        # Farbreferenz aus dem Einzelbild bestimmen (Markerbereich optional ausschließen)
        ema_lab = None
        polys = [quad.copy()] if quad is not None else None
        try:
            center_lab = beddet.auto_calibrate_lab_center([img.copy()],
                                                          exclusion_polys_per_frame=polys)
            ema_lab = center_lab.copy()
        except Exception:
            ema_lab = lab[H//2, W//2].astype(np.float32)

        thr = float(LAB_THR)
        K5  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

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
        if rect is not None and area >= min_draw_area:
            angle_long = float(beddet.long_side_angle_deg(rect))
            theta_img_vert_deg = float(_angle_vertical_from_long(angle_long))
            rect_fill = float(beddet.rect_mask_fill_ratio(mask, rect))
            area_px   = float(area)
    except Exception:
        pass

    return {
        "dx_px": dx, "dy_px": dy, "dist_center_px": delta_c,
        "theta_img_vert_deg": theta_img_vert_deg,
        "rect_fill": rect_fill, "area_px": area_px
    }


# ---------------- CSV ----------------

def csv_open_with_header():
    fields = [
        "run_id","mode","platform","t_align_s",
        "screenshot_raw",
        "dx_px","dy_px","dist_center_px",
        "theta_img_vert_deg","rect_fill","area_px"
    ]
    need_header = (not os.path.exists(CSV_PATH)) or (os.path.getsize(CSV_PATH)==0)
    f = open(CSV_PATH, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fields)
    if need_header:
        writer.writeheader(); f.flush()
    return f, writer


# ---------------- Hauptprogramm ----------------

def main():
    # Ordner anlegen
    ensure_dir(os.path.dirname(CSV_PATH))
    ensure_dir(SHOTS_DIR)
    f_csv, writer = csv_open_with_header()

    # Modus interaktiv wählen
    sel = input("Modus wählen (1=guided mit Hinweisen, 2=intuitive ohne Hinweise) [1/2]: ").strip()
    mode = "guided" if sel != "2" else "intuitive"
    plat = "Windows" if platform.system()=="Windows" else "macOS"

    # Kamera öffnen
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if platform.system()=="Windows" else cv2.CAP_ANY)
    if not cap.isOpened():
        print("Kamera-Error: konnte nicht geöffnet werden.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS,          30)

    # Kalibrierung (für guided Gatekeeping)
    cam_mtx, dist_coefs = load_calibration()

    # ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector   = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    # Marker-Objektpunkte (guided Gatekeeping)
    objp = np.array([
        [-MARKER_LENGTH_M/2,  MARKER_LENGTH_M/2, 0],
        [ MARKER_LENGTH_M/2,  MARKER_LENGTH_M/2, 0],
        [ MARKER_LENGTH_M/2, -MARKER_LENGTH_M/2, 0],
        [-MARKER_LENGTH_M/2, -MARKER_LENGTH_M/2, 0]
    ], dtype=np.float32)

    # Fenster
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    make_window_same_percent(WINDOW)

    # Run-IDs/Zeit
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{mode}"
    t_start = time.perf_counter()

    # Guided: Phasen (1:1 zu main.py)
    PH_CENTER, PH_DIST, PH_CALIB, PH_ALIGN = "CENTER","DIST","CALIB","ALIGN"
    phase = PH_CENTER

    CENTER_STABLE_FRAMES = 8
    DIST_STABLE_FRAMES   = 8

    # Guided Zustände/Flags (1:1)
    calibrated = False
    warmup_cnt = 0
    calib_buf, calib_polys = [], []
    ema_lab = None
    thr = float(LAB_THR)

    align_start = None
    last_sound = 0.0
    hold_last_sound = 0.0
    sound_done = False
    sound_block_until = 0.0
    pre_sound_played = False
    lab_sound_started = False
    center_ok_cnt = 0
    dist_ok_cnt   = 0

    limiter_dirs = SoundRateLimiter(min_interval_sec=3.0)

    print(f"[RUN] Start {run_id}  |  Mode={mode}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if ROTATE_90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        seg_src = frame.copy()   # RAW (ohne Overlays) -> wird gespeichert/ausgewertet
        H, W = frame.shape[:2]
        scale = text_scale(H, base=0.5)
        now = time.time()

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

        # ---------------- INTUITIVE (ohne Hinweise) ----------------
        if mode == "intuitive":
            vis = frame.copy()
            put_text(vis,
                     "Intuitiv: SPACE drücken, wenn Sie glauben, dass es ausgerichtet ist.",
                     (30, int(60*scale)), scale, (0,200,255))

            if key == 32:  # SPACE -> Screenshot & Ende
                t_end   = time.perf_counter()
                t_align = t_end - t_start

                # GENAU EIN RAW-Foto speichern
                shot_raw = os.path.join(SHOTS_DIR, f"{run_id}_raw.png")
                cv2.imwrite(shot_raw, seg_src)
                shot_raw = shot_raw.replace("\\","/")

                # Nur Bildanalyse
                res = analyze_screenshot(shot_raw) or {}

                # CSV
                writer.writerow({
                    "run_id": run_id, "mode": mode, "platform": plat,
                    "t_align_s": f"{t_align:.3f}",
                    "screenshot_raw": shot_raw,
                    "dx_px": f"{res.get('dx_px',''):.2f}" if res.get('dx_px') is not None else "",
                    "dy_px": f"{res.get('dy_px',''):.2f}" if res.get('dy_px') is not None else "",
                    "dist_center_px": f"{res.get('dist_center_px',''):.2f}" if res.get('dist_center_px') is not None else "",
                    "theta_img_vert_deg": f"{res.get('theta_img_vert_deg',''):+.2f}" if res.get('theta_img_vert_deg') is not None else "",
                    "rect_fill": f"{res.get('rect_fill',''):.3f}" if res.get('rect_fill') is not None else "",
                    "area_px":   f"{res.get('area_px',''):.0f}" if res.get('area_px') is not None else ""
                })
                f_csv.flush()

                # Ende nach EINEM Durchlauf
                cap.release(); cv2.destroyAllWindows(); f_csv.close()
                print(f"[OK] Intuitiver Run abgeschlossen. CSV: {CSV_PATH}  |  Shot: {shot_raw}")
                return

            cv2.imshow(WINDOW, vis)
            continue

        # ---------------- GUIDED (1:1 zu main.py) ----------------

        # Marker
        gray = cv2.cvtColor(seg_src, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        img_ctr = (W // 2, H // 2)
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
            cv2.imshow(WINDOW, frame)

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
            continue

        # Visualisierung + Zentrier-Check
        pts = aruco_quad.astype(int)
        m_ctr = tuple(pts.mean(axis=0).astype(int))
        cv2.circle(frame, m_ctr, 5, (255, 0, 0), -1)

        centered, dirs = is_centered(m_ctr, frame.shape)

        # Pose / Abstand (z) – nur Gatekeeping
        img_pts = aruco_quad.astype(np.float32)
        ok_pnp, rvec, tvec = cv2.solvePnP(objp, img_pts, cam_mtx, dist_coefs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok_pnp:
            rvec, tvec = cv2.solvePnPRefineLM(objp, img_pts, cam_mtx, dist_coefs, rvec, tvec)
        z = (float(np.ravel(tvec)[2]) + CALIB_OFFSET_M) if ok_pnp else None
        z_ok = (z is not None) and (abs(z - TARGET_DISTANCE_M) <= DISTANCE_TOLERANCE_M)

        # ======== PHASE 1: ZENTRIEREN ========
        if phase == PH_CENTER:
            if not centered:
                draw_arrows(
                    frame, dirs, W, H,
                    play_sounds=(now >= sound_block_until),
                    limiter=limiter_dirs
                )
                put_text(frame, "Kamera zum Marker zentrieren", (30, int(40 * scale)), scale, (0, 0, 255))
                center_ok_cnt = 0
            else:
                center_ok_cnt += 1
                put_text(frame, "Stabil halten", (30, int(40 * scale)), scale, (0, 200, 255))
                if center_ok_cnt >= CENTER_STABLE_FRAMES:
                    phase = PH_DIST
                    dist_ok_cnt = 0
            cv2.imshow(WINDOW, frame)
            continue

        # ======== PHASE 2: ABSTAND/HÖHE ========
        if phase == PH_DIST:
            if not centered:
                phase = PH_CENTER
                center_ok_cnt = 0
                cv2.imshow(WINDOW, frame)
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
                    if calibrated:
                        phase = PH_ALIGN
                        align_start = None
                        sound_done = False
                        hold_last_sound = 0.0
                    else:
                        phase = PH_CALIB
                        warmup_cnt = 0
                        calib_buf.clear(); calib_polys.clear()
                        lab_sound_started = False
                        if (now - last_sound) > SOUND_INTERVAL_SEC and not pre_sound_played:
                            _play_dir_sound_async('warte_vor_kalibrierung')
                            last_sound = now
                            pre_sound_played = True

            cv2.imshow(WINDOW, frame)
            continue

        # ======== PHASE 3: LAB-KALIBRIERUNG ========
        if phase == PH_CALIB:
            if not centered:
                phase = PH_CENTER
                center_ok_cnt = 0
                warmup_cnt = 0
                calib_buf.clear(); calib_polys.clear()
                cv2.imshow(WINDOW, frame)
                continue
            if not z_ok:
                phase = PH_DIST
                dist_ok_cnt = 0
                warmup_cnt = 0
                calib_buf.clear(); calib_polys.clear()
                cv2.imshow(WINDOW, frame)
                continue

            if warmup_cnt < WARMUP_FRAMES:
                warmup_cnt += 1
                put_text(frame, "Kalibrierung läuft – stabil halten", (30, int(40 * scale)), scale, (0, 200, 255))
                cv2.imshow(WINDOW, frame)
                continue
            else:
                if not lab_sound_started and (now - last_sound) > SOUND_INTERVAL_SEC:
                    _play_dir_sound_async('kalibrieren_lab')
                    last_sound = now
                    lab_sound_started = True

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
                    sound_block_until = time.time()
                    phase = PH_ALIGN
                    align_start = None
                    sound_done = False
                    hold_last_sound = 0.0

            cv2.imshow(WINDOW, frame)
            continue

        # ======== PHASE 4: Z-ROTATION / AUSRICHTUNG ========
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
            put_text_bg(frame,
                        "Zentrierung/Abstand verloren – bitte neu ausrichten",
                        (30, int(80 * scale)), scale, (255, 255, 255), bg=(0, 0, 255))
            cv2.imshow(WINDOW, frame)
            continue

        # Nur hier wird segmentiert/ausgerichtet (Live-Entscheidung)
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

        K5  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
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
                    sound_done  = False

                stable = (now - align_start) >= STABLE_ALIGN_SEC
                if stable:
                    put_text(vis, "AUSGERICHTET", (30, int(80 * scale)), scale, (0, 200, 0))
                    if not sound_done and (now - last_sound) > SOUND_INTERVAL_SEC and now >= sound_block_until:
                        _play_dir_sound_async('ausgerichtet')
                        sound_done = True
                        last_sound = now

                    # ---- HIER Screenshot & Ende (GENAU 1 Foto) ----
                    t_end   = time.perf_counter()
                    t_align = t_end - t_start

                    shot_raw = os.path.join(SHOTS_DIR, f"{run_id}_raw.png")
                    cv2.imwrite(shot_raw, seg_src)
                    shot_raw = shot_raw.replace("\\","/")

                    res = analyze_screenshot(shot_raw) or {}

                    writer.writerow({
                        "run_id": run_id, "mode": mode, "platform": plat,
                        "t_align_s": f"{t_align:.3f}",
                        "screenshot_raw": shot_raw,
                        "dx_px": f"{res.get('dx_px',''):.2f}" if res.get('dx_px') is not None else "",
                        "dy_px": f"{res.get('dy_px',''):.2f}" if res.get('dy_px') is not None else "",
                        "dist_center_px": f"{res.get('dist_center_px',''):.2f}" if res.get('dist_center_px') is not None else "",
                        "theta_img_vert_deg": f"{res.get('theta_img_vert_deg',''):+.2f}" if res.get('theta_img_vert_deg') is not None else "",
                        "rect_fill": f"{res.get('rect_fill',''):.3f}" if res.get('rect_fill') is not None else "",
                        "area_px":   f"{res.get('area_px',''):.0f}" if res.get('area_px') is not None else ""
                    })
                    f_csv.flush()

                    cap.release(); cv2.destroyAllWindows(); f_csv.close()
                    print(f"[OK] Guided Run abgeschlossen. CSV: {CSV_PATH}  |  Shot: {shot_raw}")
                    return
                else:
                    put_text(vis, "Bitte nicht bewegen – Stabilitätsprüfung läuft",
                             (30, int(80 * scale)), scale, (0, 200, 255))
                    if (now - hold_last_sound) > SOUND_INTERVAL_SEC and now >= sound_block_until:
                        _play_dir_sound_async('warte_nach_ausrichtung')
                        hold_last_sound = now
            else:
                align_start = None
                sound_done  = False
                dir_txt = "Bitte rechts drehen" if err > 0 else "Bitte links drehen"
                put_text_bg(vis, dir_txt, (30, int(80 * scale)), scale,
                            (255, 255, 255), bg=(0, 0, 255))
                radius = max(60, min(W, H) // 4)
                draw_rotation_arrow(
                    vis, (W // 2, H // 2),
                    radius=radius, clockwise=(err > 0),
                    color=(0, 0, 255), thickness=6, sweep_deg=200,
                    play_sound=(time.time() >= sound_block_until),
                    limiter=limiter_dirs
                )
        else:
            put_text_bg(vis, "Kein Bett erkannt", (30, int(80 * scale)), scale,
                        (255, 255, 255), bg=(0, 0, 255))

        cv2.imshow(WINDOW, vis)

    # Graceful exit
    cap.release()
    cv2.destroyAllWindows()
    f_csv.close()


if __name__ == "__main__":
    main()
