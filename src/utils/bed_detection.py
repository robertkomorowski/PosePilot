# -*- coding: utf-8 -*-
import cv2
import numpy as np
from config import *   # ggf. auf "from config import *" anpassen


# ==============================
# Hilfsfunktionen
# ==============================
def inner_roi(frame, margin_frac=MARGIN_FRAC):
    h, w = frame.shape[:2]
    mx, my = int(w * margin_frac), int(h * margin_frac)
    x1, y1 = mx, my
    x2, y2 = w - mx, h - my
    return x1, y1, x2, y2

def _center_patch_rect(shape, frac_w=CENTER_FRAC_W, frac_h=CENTER_FRAC_H):
    H, W = shape[:2]
    ww = max(4, int(W * frac_w))
    hh = max(4, int(H * frac_h))
    cx, cy = W // 2, H // 2
    x1 = max(0, cx - ww // 2); x2 = min(W, cx + ww // 2)
    y1 = max(0, cy - hh // 2); y2 = min(H, cy + hh // 2)
    return x1, y1, x2, y2

def _center_exclusion_rect(shape, frac_w=0.035, frac_h=0.035):
    """Kleines Innen-Rechteck (z.B. Marker-Zentrum) ausschließen – fixe ~3.5%."""
    H, W = shape[:2]
    ww = max(2, int(W * frac_w))
    hh = max(2, int(H * frac_h))
    cx, cy = W // 2, H // 2
    x1 = max(0, cx - ww // 2); x2 = min(W, cx + ww // 2)
    y1 = max(0, cy - hh // 2); y2 = min(H, cy + hh // 2)
    return x1, y1, x2, y2

def _avg_marker_side_px(poly4x2: np.ndarray) -> float:
    p = poly4x2.astype(np.float32)
    d01 = np.linalg.norm(p[1] - p[0])
    d12 = np.linalg.norm(p[2] - p[1])
    d23 = np.linalg.norm(p[3] - p[2])
    d30 = np.linalg.norm(p[0] - p[3])
    return float((d01 + d12 + d23 + d30) / 4.0)

def _poly_mask(shape, poly4x2, pad_px=0):
    H, W = shape[:2]
    m = np.zeros((H, W), np.uint8)
    poly = poly4x2.astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(m, [poly], 255)
    if pad_px and pad_px > 0:
        k = max(1, int(pad_px))
        k = k if k % 2 == 1 else k + 1
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.dilate(m, ker, iterations=1)
    return m

def robust_center_lab_with_exclusions(lab_img,
                                      outer_frac_w=CENTER_FRAC_W, outer_frac_h=CENTER_FRAC_H,
                                      exclusion_polys=None, pad_px=0):
    """Farbmittelwert aus zentralem Rechteck, aber mit Ausschluss-Polygonen (z. B. ArUco)."""
    H, W = lab_img.shape[:2]
    ox1, oy1, ox2, oy2 = _center_patch_rect((H, W), outer_frac_w, outer_frac_h)

    sel_mask = np.zeros((H, W), np.uint8)
    sel_mask[oy1:oy2, ox1:ox2] = 255

    if exclusion_polys:
        for poly in exclusion_polys:
            excl = _poly_mask((H, W), poly, pad_px=pad_px)
            sel_mask[excl > 0] = 0

    pts = lab_img[sel_mask.astype(bool)].reshape(-1, 3).astype(np.float32)
    if pts.size == 0:
        return robust_center_lab(lab_img, (ox1, oy1, ox2, oy2))

    med = np.median(pts, axis=0)
    d = np.linalg.norm(pts - med, axis=1)
    keep = d < np.percentile(d, 80)
    kept = pts[keep]
    return kept.mean(axis=0) if kept.size else pts.mean(axis=0)

def robust_center_lab_ring(lab_img,
                           outer_frac_w=CENTER_FRAC_W, outer_frac_h=CENTER_FRAC_H,
                           excl_frac_w=0.035, excl_frac_h=0.035):
    """„Donut“-Sampling: Außenrechteck MINUS kleines Innenrechteck."""
    H, W = lab_img.shape[:2]
    ox1, oy1, ox2, oy2 = _center_patch_rect((H, W), outer_frac_w, outer_frac_h)
    ix1, iy1, ix2, iy2 = _center_exclusion_rect((H, W), excl_frac_w, excl_frac_h)
    sel_mask = np.zeros((H, W), np.uint8)
    sel_mask[oy1:oy2, ox1:ox2] = 255
    sel_mask[iy1:iy2, ix1:ix2] = 0

    pts = lab_img[sel_mask.astype(bool)].reshape(-1, 3).astype(np.float32)
    if pts.size == 0:
        return robust_center_lab(lab_img, (ox1, oy1, ox2, oy2))

    med = np.median(pts, axis=0)
    d = np.linalg.norm(pts - med, axis=1)
    keep = d < np.percentile(d, 80)
    kept = pts[keep]
    return kept.mean(axis=0) if kept.size else pts.mean(axis=0)

def robust_center_lab(lab_img, patch_rect):
    x0, y0, x1, y1 = patch_rect
    patch = lab_img[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    if patch.size == 0:
        return np.array([50., 0., 0.], np.float32)
    med = np.median(patch, axis=0)
    d = np.linalg.norm(patch - med, axis=1)
    keep = d < np.percentile(d, 80)
    return patch[keep].mean(axis=0)

# --------- robuste ΔE + Chroma-Gate ----------
def segment_by_lab_distance(lab_img, center_lab, thr, wL=0.25, wA=1.0, wB=1.0, min_chroma=10.0):
    """ΔE mit reduzierter L-Gewichtung (robuster gg. Schatten) + Chroma-Gate."""
    L = lab_img[...,0].astype(np.float32)
    A = lab_img[...,1].astype(np.float32)
    B = lab_img[...,2].astype(np.float32)
    dL = L - float(center_lab[0])
    dA = A - float(center_lab[1])
    dB = B - float(center_lab[2])

    delta  = np.sqrt(wL*dL*dL + wA*dA*dA + wB*dB*dB)
    chroma = np.sqrt(A*A + B*B)  # C*ab
    mask = (delta < float(thr)) & (chroma >= float(min_chroma))
    return (mask.astype(np.uint8)) * 255


# ---------- Masken-Postprocessing ----------
def _kernel(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(k), int(k)))

def keep_largest_component(mask, min_area=MIN_LCC_AREA):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    lcc = np.zeros_like(mask); lcc[labels == idx] = 255
    if int(areas.max()) < min_area:
        return mask
    return lcc

def keep_component_touching_center(mask, center_rect):
    x0,y0,x1,y1 = center_rect
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num <= 1:
        return mask
    seed = labels[y0:y1, x0:x1]
    seed = seed[mask[y0:y1, x0:x1] > 0]
    out = np.zeros_like(mask)
    if seed.size:
        vals, cnts = np.unique(seed, return_counts=True)
        keep_idx = int(vals[np.argmax(cnts)])
        out[labels == keep_idx] = 255
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep_idx = 1 + int(np.argmax(areas)) if areas.size else 0
        out[labels == keep_idx] = 255
    return out

def fill_holes_in_mask(mask: np.ndarray) -> np.ndarray:
    """
    Füllt Löcher robust – auch wenn die Vordergrundmaske den Bildrand berührt.
    Macht ein 1px Null-Padding und floodfillt von außerhalb.
    """
    m = (mask > 0).astype(np.uint8) * 255
    h, w = m.shape[:2]

    # 1px Nullrand drumherum, damit (0,0) sicher Background ist
    pad = 1
    mp = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    flood = mp.copy()
    flood_mask = np.zeros((h + 2*pad + 2, w + 2*pad + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)

    # wieder ent-padden
    flood = flood[pad:-pad, pad:-pad]

    # „Löcher“ = Bereiche, die NICHT vom Außenhintergrund erreicht wurden
    holes = cv2.bitwise_not(flood)
    holes = cv2.bitwise_and(holes, cv2.bitwise_not(m))  # nur echte Löcher innerhalb der Maske

    return cv2.bitwise_or(m, holes)


def convex_hull_mask(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask
    cnt = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [hull], -1, 255, thickness=cv2.FILLED)
    return out

def postprocess_mask(mask):
    mask_pp = mask.copy()
    if USE_MEDIAN:
        mask_pp = cv2.medianBlur(mask_pp, MEDIAN_KSIZE)
    if USE_MORPH_CLOSE:
        mask_pp = cv2.morphologyEx(mask_pp, cv2.MORPH_CLOSE, _kernel(CLOSE_KSIZE), iterations=CLOSE_ITERS)
    if USE_MORPH_OPEN:
        mask_pp = cv2.morphologyEx(mask_pp, cv2.MORPH_OPEN, _kernel(OPEN_KSIZE), iterations=OPEN_ITERS)
    if KEEP_LARGEST_COMP:
        mask_pp = keep_largest_component(mask_pp, MIN_LCC_AREA)
    if FILL_HOLES:
        mask_pp = fill_holes_in_mask(mask_pp)
    if USE_CONVEX_HULL:
        mask_pp = convex_hull_mask(mask_pp)
    return mask_pp

def trim_border_if_touching(mask, frac=0.01, px_min=4):
    """Wenn die Maske den Bildrand berührt, einen dünnen Rand (≈1%) wegschneiden."""
    H, W = mask.shape[:2]
    if mask[0,:].any() or mask[-1,:].any() or mask[:,0].any() or mask[:,-1].any():
        b = max(px_min, int(frac * min(H, W)))
        mask[:b, :]  = 0
        mask[-b:, :] = 0
        mask[:, :b]  = 0
        mask[:, -b:] = 0
    return mask

# --------- Randkontakt-Erkennung + adaptive Guards ----------
def borders_touch(mask: np.ndarray):
    """(top, bottom, left, right) – je True, wenn Maske diesen Rand berührt."""
    return (bool(mask[0, :].any()),
            bool(mask[-1, :].any()),
            bool(mask[:, 0].any()),
            bool(mask[:, -1].any()))

def rect_guard_adaptive(mask: np.ndarray,
                        top_frac=0.02, side_frac=0.02, bottom_frac=0.02, min_px=6,
                        inner_margin_frac=0.01):
    """
    Stärkerer Guard nur an den tatsächlich berührten Rändern.
    """
    m = mask.copy()
    H, W = m.shape[:2]
    t, b, l, r = borders_touch(m)

    if t:
        bpx = max(min_px, int(top_frac * min(H, W)))
        m[:bpx, :] = 0
    if b:
        bpx = max(min_px, int(bottom_frac * min(H, W)))
        m[-bpx:, :] = 0
    if l:
        bpx = max(min_px, int(side_frac * min(H, W)))
        m[:, :bpx] = 0
    if r:
        bpx = max(min_px, int(side_frac * min(H, W)))
        m[:, -bpx:] = 0

    if inner_margin_frac and inner_margin_frac > 0:
        x1 = int(W * inner_margin_frac)
        y1 = int(H * inner_margin_frac)
        x2 = W - x1
        y2 = H - y1
        gate = np.zeros_like(m)
        gate[y1:y2, x1:x2] = 255
        m = cv2.bitwise_and(m, gate)

    return m

# ---------- Rechteck & Ausrichtungsfehler ----------
def min_area_rect_from_mask(mask, inset_px=EDGE_INSET_PX, use_hull_for_rect=False):
    mask_use = mask
    if inset_px and inset_px > 0:
        k = max(3, 2 * (inset_px // 2) + 1)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        shrunk = cv2.erode(mask, ker, iterations=1)
        if cv2.countNonZero(shrunk) > 500:
            mask_use = shrunk

    cnts, _ = cv2.findContours(mask_use, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0
    cnt = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))

    if use_hull_for_rect:
        geom = cv2.convexHull(cnt)
    else:
        geom = cnt  # genauer – bläht das Rechteck nicht künstlich auf

    rect = cv2.minAreaRect(geom)  # ((cx,cy),(w,h),angle in (-90,0])
    return rect, area

def long_side_angle_deg(rect):
    (_, _), (w, h), ang = rect
    long_ang = ang + 90.0 if w < h else ang
    if long_ang > 90:  long_ang -= 180
    if long_ang < -90: long_ang += 180
    return long_ang

def alignment_error(angle_long_deg, frame_shape, mode=ALIGN_MODE):
    H, W = frame_shape[:2]
    axis = ("horizontal" if mode == "horizontal"
            else "vertical" if mode == "vertical"
            else ("horizontal" if W >= H else "vertical"))
    if axis == "horizontal":
        err = angle_long_deg
    else:
        e1 = angle_long_deg - 90.0
        e2 = angle_long_deg + 90.0
        err = e1 if abs(e1) < abs(e2) else e2
        if err > 90:  err -= 180
        if err < -90: err += 180
    return err


# ==============================
# Lab-Kalibrierung
# ==============================
def auto_calibrate_lab_center(frames, exclusion_polys_per_frame=None, pad_frac=0.35):
    """
    frames: Liste von BGR-Frames
    exclusion_polys_per_frame: Liste gleicher Länge; pro Frame None oder Polygon(e).
    """
    labs = []
    for i, bgr in enumerate(frames):
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)

        polys = None
        pad_px = 0
        if exclusion_polys_per_frame is not None:
            pf = exclusion_polys_per_frame[i]
            if pf is not None:
                if isinstance(pf, (list, tuple)):
                    polys = [np.array(p, dtype=np.float32) for p in pf]
                else:
                    polys = [np.array(pf, dtype=np.float32)]
                if len(polys) > 0:
                    side = _avg_marker_side_px(polys[0])
                    pad_px = int(max(2.0, pad_frac * side))

        labs.append(
            robust_center_lab_with_exclusions(
                lab,
                outer_frac_w=CENTER_FRAC_W, outer_frac_h=CENTER_FRAC_H,
                exclusion_polys=polys, pad_px=pad_px
            )
        )

    labs = np.array(labs, dtype=np.float32)
    med = np.median(labs, axis=0)
    d = np.linalg.norm(labs - med, axis=1)
    keep = d < np.percentile(d, 80)
    center = labs[keep].mean(axis=0)
    return center.astype(np.float32)

def rect_mask_fill_ratio(mask_bin: np.ndarray, rect) -> float:
    """Anteil des Rechteckbereichs, der auch in der Maske gefüllt ist (0..1)."""
    if rect is None:
        return 0.0
    canvas = np.zeros_like(mask_bin)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillPoly(canvas, [box], 255)
    inter = cv2.bitwise_and(canvas, mask_bin)
    den = max(1, cv2.countNonZero(canvas))
    return cv2.countNonZero(inter) / den

# --------- Rand-sichere Rechteckschätzung ----------
def touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())

def _largest_contour(mask: np.ndarray):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0
    cnt = max(cnts, key=cv2.contourArea)
    return cnt, float(cv2.contourArea(cnt))

def min_area_rect_border_safe(mask: np.ndarray,
                              inset_px: int = EDGE_INSET_PX,
                              pad_px: int = 60,
                              use_hull_for_rect: bool = False,
                              extra_inset_if_touch: int = 40):
    """
    Robust, wenn die Maske den Bildrand berührt (Erosion + REFLECT-Padding).
    """
    cnt_orig, area_orig = _largest_contour(mask)
    if cnt_orig is None:
        return None, 0.0

    inset = int(inset_px)
    if touches_border(mask):
        inset = max(inset, int(extra_inset_if_touch))

    mask_use = mask
    if inset > 0:
        k = max(3, 2 * (inset // 2) + 1)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        shrunk = cv2.erode(mask, ker, iterations=1)
        if cv2.countNonZero(shrunk) > 0:
            mask_use = shrunk

    if not touches_border(mask):
        cnt, _ = _largest_contour(mask_use)
        if cnt is None:
            return None, 0.0
        geom = cv2.convexHull(cnt) if use_hull_for_rect else cnt
        rect = cv2.minAreaRect(geom)
        return rect, area_orig

    p = int(max(8, pad_px))
    padded = cv2.copyMakeBorder(mask_use, p, p, p, p, borderType=cv2.BORDER_REFLECT)

    cnt_pad, _ = _largest_contour(padded)
    if cnt_pad is None:
        return None, 0.0

    geom_pad = cv2.convexHull(cnt_pad) if use_hull_for_rect else cnt_pad
    rect_pad = cv2.minAreaRect(geom_pad)

    (cx, cy), (rw, rh), ang = rect_pad
    rect = ((cx - p, cy - p), (rw, rh), ang)
    return rect, area_orig


# ==============================
# Einfacher Test-Runner
# ==============================
def _make_side_by_side(vis_bgr: np.ndarray, mask_gray: np.ndarray) -> np.ndarray:
    """Erzeugt eine Side-by-Side-Ansicht (links: vis, rechts: Maske BGR) mit Labels & Trennlinie."""
    if mask_gray is None or mask_gray.size == 0:
        mask_gray = np.zeros(vis_bgr.shape[:2], np.uint8)
    mask_bgr = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)

    pair = np.hstack([vis_bgr, mask_bgr])

    h, w = vis_bgr.shape[:2]
    # Trennlinie zeichnen
    cv2.line(pair, (w, 0), (w, h), (128, 128, 128), 1)
    # Labels
    cv2.putText(pair, "Bild", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(pair, "Maske", (w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

    return pair


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera konnte nicht geöffnet werden.")
        return

    calibrated = False
    warmup_cnt = 0
    calib_buf = []
    ema_lab = None
    thr = float(LAB_THR)

    cv2.namedWindow("Bett (Bild + Maske)", cv2.WINDOW_NORMAL)

    # einmalige Ressourcen
    K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if ROTATE_90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        vis = frame.copy()
        current_mask = None  # zwecks Side-by-Side auch in Kalibrierphase

        # ---------- Kalibrierung ----------
        if not calibrated:
            if warmup_cnt < WARMUP_FRAMES:
                warmup_cnt += 1
            else:
                calib_buf.append(frame.copy())
                if len(calib_buf) >= CALIB_FRAMES:
                    center_lab = auto_calibrate_lab_center(calib_buf)  # ohne ArUco
                    ema_lab = center_lab.copy()
                    calibrated = True
                    calib_buf.clear()
                    print(f"[LAB] center={center_lab.round(2).tolist()} thr={thr}")
            cv2.putText(vis, "Kalibriere Farbe (Lab)...", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2, cv2.LINE_AA)

            pair = _make_side_by_side(vis, np.zeros(vis.shape[:2], np.uint8))
            cv2.imshow("Bett (Bild + Maske)", pair)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # ---------- Segmentierung (pro Frame) ----------
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)

        # Donut-Sampling + EMA
        center_now = robust_center_lab_ring(
            lab,
            outer_frac_w=CENTER_FRAC_W, outer_frac_h=CENTER_FRAC_H,
            excl_frac_w=0.035, excl_frac_h=0.035
        )
        ema_lab = (1.0 - float(LAB_EMA_ALPHA)) * ema_lab + float(LAB_EMA_ALPHA) * center_now

        # ΔE mit weniger L-Einfluss + Chroma-Gate
        mask = segment_by_lab_distance(lab, ema_lab, thr, wL=0.25, min_chroma=10.0)

        # kleine Öffnung vor dem Seeden (dünne Brücken kappen)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, K5, iterations=1)

        # Mini-Trim, falls an irgendeinem Rand angedockt
        mask = trim_border_if_touching(mask, frac=0.008, px_min=3)

        # Seedauswahl: nur die Komponente, die das Center-Rechteck berührt
        cx1, cy1, cx2, cy2 = _center_patch_rect(lab.shape, CENTER_FRAC_W, CENTER_FRAC_H)
        mask = keep_component_touching_center(mask, (cx1, cy1, cx2, cy2))

        # Jetzt erst "schönmachen"
        mask = postprocess_mask(mask)
        current_mask = mask

        # ---------- Rechteck/Feedback ----------
        H, W = frame.shape[:2]
        min_draw_area = float(MIN_DRAW_AREA_FRAC) * (H * W)

        # ADAPTIVER Guard: oben strenger, wenn oben berührt wird
        mask_for_rect = rect_guard_adaptive(
            mask, top_frac=0.04, side_frac=0.02, bottom_frac=0.02, min_px=6, inner_margin_frac=0.01
        )

        # 1) Rechteck robust auf der Guard-Maske
        rect, _ = min_area_rect_border_safe(
            mask_for_rect,
            inset_px=EDGE_INSET_PX,
            pad_px=80,                 # etwas höher, hilft bei oben berührt
            use_hull_for_rect=False,
            extra_inset_if_touch=60
        )

        # 2) Fläche auf der ORIGINAL-Maske messen (Gate)
        cnt, area = _largest_contour(mask)
        area = float(area) if cnt is not None else 0.0

        # Sanity-Check: Rechteck muss die Maske „gut füllen“
        fill_ratio = rect_mask_fill_ratio(mask, rect)
        if rect is None or area < min_draw_area or fill_ratio < 0.60:
            # Fallback: noch aggressiver
            rect, _ = min_area_rect_border_safe(
                mask_for_rect,
                inset_px=EDGE_INSET_PX + 20,
                pad_px=100,
                use_hull_for_rect=False,
                extra_inset_if_touch=80
            )
            fill_ratio = rect_mask_fill_ratio(mask, rect)

        if rect is not None and area >= min_draw_area:
            ang_long = long_side_angle_deg(rect)
            err = alignment_error(ang_long, frame.shape, ALIGN_MODE)
            aligned = abs(err) <= float(ANGLE_TOLERANCE_DEG)

            color = (0, 200, 0) if aligned else (0, 0, 255)
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.drawContours(vis, [box], 0, color, 3)

            if aligned:
                cv2.putText(vis, "AUSGERICHTET", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,0), 2, cv2.LINE_AA)
            else:
                msg = "Bitte rechts drehen" if err > 0 else "Bitte links drehen"
                cv2.putText(vis, msg, (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2, cv2.LINE_AA)
        else:
            cv2.putText(vis, "Rand nicht gefunden – LAB_THR/Morphologie anpassen.",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)

        # ---------- Side-by-Side anzeigen ----------
        pair = _make_side_by_side(vis, current_mask)
        cv2.imshow("Bett (Bild + Maske)", pair)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
