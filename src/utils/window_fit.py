# window_fit.py
# Dynamisches OpenCV-Fenster in gleichem Prozent-Anteil wie deiner Referenz,
# ohne tkinter (macOS: CoreGraphics, Windows: Win32, Linux: xrandr).

import sys
import cv2
import ctypes
import subprocess
import re

# --- Deine Referenz (von dir gemeldet) ---
BASE_SCREEN_W, BASE_SCREEN_H = 3456, 2234
BASE_WIN_W,   BASE_WIN_H     = 1080, 1920  # Ziel-Fenster auf deiner Referenz

def _get_screen_size():
    """Gibt (screen_w, screen_h) des primären Monitors zurück."""
    # macOS
    if sys.platform == "darwin":
        try:
            core = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
            core.CGMainDisplayID.restype = ctypes.c_uint32
            core.CGDisplayPixelsWide.argtypes = [ctypes.c_uint32]
            core.CGDisplayPixelsWide.restype  = ctypes.c_size_t
            core.CGDisplayPixelsHigh.argtypes = [ctypes.c_uint32]
            core.CGDisplayPixelsHigh.restype  = ctypes.c_size_t
            did = core.CGMainDisplayID()
            w = int(core.CGDisplayPixelsWide(did))
            h = int(core.CGDisplayPixelsHigh(did))
            if w > 0 and h > 0:
                return w, h
        except Exception:
            pass

    # Windows
    if sys.platform.startswith("win"):
        try:
            user32 = ctypes.windll.user32
            # Optional: DPI-aware, sonst bekommst du ggf. skalierten Wert
            try:
                user32.SetProcessDPIAware()
            except Exception:
                pass
            return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
        except Exception:
            pass

    # Linux (X11) – best effort über xrandr
    try:
        out = subprocess.check_output(["xrandr", "--current"], stderr=subprocess.DEVNULL).decode()
        m = re.search(r"current\s+(\d+)\s+x\s+(\d+)", out)
        if m:
            return int(m.group(1)), int(m.group(2))
    except Exception:
        pass

    # Fallback
    return 1920, 1080

def make_window_same_percent(
    win_name: str = "Ausrichtung - Live",
    margin_frac: float = 0.0,
    min_w: int = 320,
    min_h: int = 240,
    center: bool = True,
):
    """
    Erzeugt ein resizables OpenCV-Fenster und setzt die Größe so,
    dass sie auf jedem System dem gleichen Prozent-Anteil entspricht
    wie (BASE_WIN / BASE_SCREEN) deiner Referenz.
    Returns: (win_w, win_h, (p_w, p_h))
    """
    # Prozentanteile aus der Referenz
    p_w = BASE_WIN_W / BASE_SCREEN_W  # ~31.25%
    p_h = BASE_WIN_H / BASE_SCREEN_H  # ~85.94%

    sw, sh = _get_screen_size()
    max_w = int(sw * (1.0 - margin_frac))
    max_h = int(sh * (1.0 - margin_frac))

    win_w = max(min_w, int(round(p_w * sw)))
    win_h = max(min_h, int(round(p_h * sh)))

    # Sicherheit: nicht größer als nutzbarer Bereich
    win_w = min(win_w, max_w)
    win_h = min(win_h, max_h)

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, win_w, win_h)
    if center:
        try:
            cv2.moveWindow(win_name, (sw - win_w) // 2, (sh - win_h) // 2)
        except Exception:
            pass

    return win_w, win_h, (p_w, p_h)

def make_window_scaled_from_base(
    win_name: str = "Ausrichtung - Live",
    base_win=(1080, 1920),
    base_screen=(3456, 2234),
    margin_frac: float = 0.0,
    center: bool = True,
):
    """
    Alternative: skaliert ein Basis-Fenster (1080x1920) gleichmäßig
    anhand des Verhältnisses aktueller Screen vs. Basis-Screen.
    Hält dabei das Seitenverhältnis exakt.
    """
    sw, sh = _get_screen_size()
    bw, bh = base_win
    bsw, bsh = base_screen

    scale = min(sw / bsw, sh / bsh)
    win_w = int(round(bw * scale))
    win_h = int(round(bh * scale))

    max_w = int(sw * (1.0 - margin_frac))
    max_h = int(sh * (1.0 - margin_frac))
    win_w = min(win_w, max_w)
    win_h = min(win_h, max_h)

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, win_w, win_h)
    if center:
        try:
            cv2.moveWindow(win_name, (sw - win_w) // 2, (sh - win_h) // 2)
        except Exception:
            pass

    return win_w, win_h, scale
