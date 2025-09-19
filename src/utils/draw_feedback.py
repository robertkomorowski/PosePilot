# utils/draw_feedback.py
import os
import cv2
import numpy as np

from config import REF_H, TEXT_THICK, ARROW_COLOR, ARROW_THICKNESS, ARROW_LENGTH
from utils.audio import _play_dir_sound_async, SoundRateLimiter

TEXT_LINE = cv2.LINE_AA

# =========================
#  TrueType Text Rendering
# =========================
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except Exception:
    HAS_PIL = False

# Optional: OpenCV FreeType (falls verfügbar)
HAS_FT = False
_ft2 = None
try:
    if hasattr(cv2, "freetype"):
        _ft2 = cv2.freetype.createFreeType2()
        HAS_FT = True
except Exception:
    HAS_FT = False
    _ft2 = None


def _find_font_path():
    """
    Versucht, eine TTF-Schrift zu finden, die Umlaute sicher kann.
    Reihenfolge: Projektschrift -> Systemschriften (Linux/macOS/Windows)
    """
    candidates = [
        # Projektlokal
        "data/fonts/DejaVuSans.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        # macOS
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        # Windows
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


_TTF_PATH = _find_font_path()


def _bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _rgb_to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _ensure_font(font_size_px: int):
    """
    Liefert (modus, resource, font_metrics)
      modus == "PIL"  -> resource: PIL.ImageFont.FreeTypeFont, metrics: (ascent, descent)
      modus == "FT2"  -> resource: cv2.freetype.FreeType2,     metrics: (approx_ascent, approx_descent)
      modus == "CV2"  -> resource: None,                        metrics: None   (Fallback ohne Umlaute)
    """
    # Priorität: Pillow TTF -> OpenCV FreeType -> Hershey
    if HAS_PIL and _TTF_PATH:
        try:
            fnt = ImageFont.truetype(_TTF_PATH, font_size_px)
            try:
                ascent, descent = fnt.getmetrics()
            except Exception:
                ascent, descent = int(font_size_px * 0.8), int(font_size_px * 0.2)
            return "PIL", fnt, (ascent, descent)
        except Exception:
            pass

    if HAS_FT and _TTF_PATH:
        try:
            _ft2.loadFontData(_TTF_PATH, 0)
            # grobe Metrik-Schätzung
            ascent = int(font_size_px * 0.8)
            descent = int(font_size_px * 0.2)
            return "FT2", _ft2, (ascent, descent)
        except Exception:
            pass

    return "CV2", None, None  # Fallback (keine Umlaute)


def _measure_text_ttf(text: str, font_size_px: int):
    """
    Ermittelt Textbreite/ -höhe für Hintergrundbox.
    Nutzt TTF wenn möglich, sonst CV2-Schätzung.
    Rückgabe: (width, height, ascent, descent)
    """
    mode, res, metrics = _ensure_font(font_size_px)
    if mode == "PIL":
        # Bounding Box relativ zu (0,0)
        try:
            bbox = res.getbbox(text)  # (x0, y0, x1, y1)
            w = bbox[2] - bbox[0]
            # Höhe aus Metriken, nicht aus BBox (stabiler)
            ascent, descent = metrics
            h = ascent + descent
            return int(w), int(h), int(ascent), int(descent)
        except Exception:
            pass

    if mode == "FT2":
        # Näherung, reicht für Hintergrund:
        ascent, descent = metrics
        w = int(0.6 * font_size_px * max(1, len(text)))
        h = ascent + descent
        return w, h, ascent, descent

    # Fallback CV2 (keine Umlaute, aber Boxgröße passt):
    scale = font_size_px / 30.0
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, TEXT_THICK)
    ascent = th
    descent = bl
    return tw, th + bl, ascent, descent


def _draw_text_ttf(img_bgr, text: str, org, font_px: int, color_bgr, thickness=2):
    """
    Zeichnet Text (ohne Hintergrund). Nutzt TTF (PIL/FT2), sonst Hershey-Fallback.
    org = (x, y) ist die linke UNTERE Ecke (Baseline) – wie bei cv2.putText.
    """
    mode, res, metrics = _ensure_font(font_px)
    x, y = int(org[0]), int(org[1])

    if mode == "PIL":
        img_rgb = _bgr_to_rgb(img_bgr)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        ascent, descent = metrics
        # Pillow erwartet top-left; wir haben baseline -> top = y - ascent
        top_left = (x, y - ascent)
        color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
        draw.text(top_left, text, font=res, fill=color_rgb)
        return _rgb_to_bgr(np.array(pil_img))

    if mode == "FT2":
        # OpenCV FreeType benutzt Baseline-Org, wie cv2.putText
        res.putText(
            img_bgr, text, (x, y),
            fontHeight=font_px, color=color_bgr,
            thickness=thickness, line_type=cv2.LINE_AA,
            bottomLeftOrigin=False
        )
        return img_bgr

    # Fallback: Hershey (keine Umlaute) – sollte selten nötig sein
    cv2.putText(img_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_px / 30.0, color_bgr, thickness, TEXT_LINE)
    return img_bgr


# =========================
#  Öffentliche Helfer/API
# =========================
def text_scale(h, base=0.5):
    """Font-Scale dynamisch zur Bildhöhe (identisch zur alten Signatur)."""
    return max(0.8, base * (h / REF_H))


def put_text(img, text, org, scale, color, thickness=TEXT_THICK):
    """
    Text ohne Hintergrund (mit Umlauten, via TTF).
    org: linke untere Ecke (Baseline) wie cv2.putText.
    scale: wie bisher genutzt -> wir mappen auf Pixelhöhe.
    """
    # Mapping: bei scale ~1.0 etwa 30px Schrift
    font_px = max(12, int(48 * float(scale)))
    img[:] = _draw_text_ttf(img, text, org, font_px, color, thickness=thickness)


def put_text_bg(img, text, org, scale, color, bg=(0, 0, 0), pad=6, thickness=TEXT_THICK):
    """
    Text mit Hintergrundkasten (abgerundete Ecken) – Umlaute voll unterstützt.
    org: linke untere Ecke (Baseline).
    """
    font_px = max(12, int(48 * float(scale)))
    x, y = int(org[0]), int(org[1])

    tw, th, ascent, descent = _measure_text_ttf(text, font_px)
    x0 = x - pad
    y0 = y - ascent - pad          # top
    x1 = x + tw + pad
    y1 = y + descent + pad         # bottom

    # Abgerundetes Rechteck zeichnen
    radius = 6
    r = int(max(0, min(radius, (min(x1 - x0, y1 - y0)) // 2)))

    # Mittel- und Seitenflächen
    cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), bg, -1)
    cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), bg, -1)
    # Ecken
    cv2.circle(img, (x0 + r, y0 + r), r, bg, -1)
    cv2.circle(img, (x1 - r, y0 + r), r, bg, -1)
    cv2.circle(img, (x0 + r, y1 - r), r, bg, -1)
    cv2.circle(img, (x1 - r, y1 - r), r, bg, -1)

    # Text selbst (Baseline bei (x,y))
    img[:] = _draw_text_ttf(img, text, (x, y), font_px, color, thickness=thickness)


def show_and_continue(frame, msg: str):
    """
    Blendet eine Hinweisbox ein (z.B. 'Kein Marker erkannt').
    Zeichnet nur ins Bild – blockiert nicht und ruft kein imshow/waitKey auf.
    """
    h, _ = frame.shape[:2]
    scale = text_scale(h, base=0.5)
    put_text_bg(frame, msg, (30, int(40 * scale)), scale,
                (255, 255, 255), bg=(0, 0, 255))


# =========================
#  Pfeile & Drehrichtung
# =========================
def draw_arrows(frame, dirs, w, h, play_sounds=True,
                limiter: "SoundRateLimiter | None" = None):
    """
    Zeichnet kardinale Pfeile ('vorne','hinten','links','rechts') mit Rand-Padding.
    Spielt passende Sounds (inkl. Kombis) rate-limited.
    """
    if limiter is None:
        limiter = SoundRateLimiter(min_interval_sec=3.0)

    ARROW_COL = ARROW_COLOR

    # Strichstärke etwas mit Bildhöhe mitskalieren, ohne neuen Parameter
    TH = max(ARROW_THICKNESS, int(round(ARROW_THICKNESS * (h / float(REF_H)))))

    # interner Rand (kein neuer Config-Parameter)
    margin = max(40, int(0.06 * min(w, h)))          # ~6% vom kleineren Maß, mind. 40 px
    # effektive Pfeillänge (skalierend, aber nie kleiner als der eingestellte Wert)
    eff_len = max(ARROW_LENGTH, int(0.08 * min(w, h)))  # ~8% vom kleineren Maß

    present = set()
    for d in dirs:
        if d == 'vorne':
            start = (w // 2, margin + eff_len)
            end   = (w // 2, margin)
            cv2.arrowedLine(frame, start, end, ARROW_COL, TH, line_type=TEXT_LINE, tipLength=0.25)
            present.add('vorne')

        elif d == 'hinten':
            start = (w // 2, h - margin - eff_len)
            end   = (w // 2, h - margin)
            cv2.arrowedLine(frame, start, end, ARROW_COL, TH, line_type=TEXT_LINE, tipLength=0.25)
            present.add('hinten')

        elif d == 'links':
            start = (margin + eff_len, h // 2)
            end   = (margin, h // 2)
            cv2.arrowedLine(frame, start, end, ARROW_COL, TH, line_type=TEXT_LINE, tipLength=0.25)
            present.add('links')

        elif d == 'rechts':
            start = (w - margin - eff_len, h // 2)
            end   = (w - margin, h // 2)
            cv2.arrowedLine(frame, start, end, ARROW_COL, TH, line_type=TEXT_LINE, tipLength=0.25)
            present.add('rechts')

    if play_sounds and present:
        combo_key = None
        if 'vorne' in present and 'rechts' in present:
            combo_key = 'vorne_rechts'
        elif 'vorne' in present and 'links' in present:
            combo_key = 'vorne_links'
        elif 'hinten' in present and 'rechts' in present:
            combo_key = 'hinten_rechts'
        elif 'hinten' in present and 'links' in present:
            combo_key = 'hinten_links'

        if combo_key:
            _play_dir_sound_async(combo_key, limiter)
        else:
            for single in ('vorne', 'rechts', 'hinten', 'links'):
                if single in present:
                    _play_dir_sound_async(single, limiter)
                    break



def draw_rotation_arrow(frame, center, radius=100, clockwise=True,
                        color=(0, 0, 255), thickness=6, sweep_deg=45,
                        play_sound=True, limiter: "SoundRateLimiter | None" = None):
    """
    Zeichnet einen Bogenpfeil (Drehrichtung) und spielt optional einen Sound.
    """
    if limiter is None:
        limiter = SoundRateLimiter(min_interval_sec=3.0)

    cx, cy = int(center[0]), int(center[1])
    start_deg = -45
    end_deg = start_deg + (sweep_deg if clockwise else -sweep_deg)

    angles = np.linspace(start_deg, end_deg, num=48)
    pts = []
    for a in angles:
        rad = np.deg2rad(a)
        x = int(round(cx + radius * np.cos(rad)))
        y = int(round(cy + radius * np.sin(rad)))
        pts.append((x, y))

    if len(pts) >= 2:
        cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, color, thickness, lineType=TEXT_LINE)
        p_start = pts[-2]
        p_end = pts[-1]
        cv2.arrowedLine(frame, p_start, p_end, color, thickness, line_type=TEXT_LINE, tipLength=2)

    if play_sound:
        _play_dir_sound_async('drehen_rechts' if clockwise else 'drehen_links', limiter)
