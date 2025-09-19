"""
Zentrale Konfiguration für Kamera, Kalibrierung (Lab ΔE), Masken-Postprocessing,
Ausrichtungs-Feedback und UI. HSV-Einstellungen entfernt.
"""

# =========================
#  UI / AUDIO / FEEDBACK
# =========================
SOUND_INTERVAL_SEC   = 5          # Mindestabstand zwischen Sounds (Sek.)
PHASE_SOUND_INTERVAL_SEC = 3.0
TEXT_THICK           = 5           # Liniendicke für UI-Text
REF_H                = 1920         # Referenz-Höhe fürs Text-Scaling (falls verwendet)
STABLE_ALIGN_SEC     = 2.5


# Pfeil-Anzeige (zentrieren etc.)
ARROW_LENGTH         = 150
ARROW_THICKNESS      = 10
ARROW_COLOR          = (0, 0, 255) # BGR


# =========================
#  ARUCO / GEOMETRIE
# =========================
CENTER_TOLERANCE_PX  = 20          # wie nah der Marker zur Bildmitte sein muss
TARGET_DISTANCE_M    = 1.03        # Zielabstand Kamera–Marker (Meter)
DISTANCE_TOLERANCE_M = 0.05       # erlaubte Abweichung (Meter)
CALIB_OFFSET_M       = 0.27        # Offset zwischen Marker- und Bett-Ebene (Meter)
MARKER_LENGTH_M      = 0.062       # reale Kantenlänge des ArUco (Meter)


# =========================
#  KAMERA / WORKFLOW
# =========================
ROTATE_90            = True        # True: Bild auf Hochformat drehen (90° CW)
WARMUP_FRAMES        = 30           # sofort starten (kein Warm-up)
CALIB_FRAMES         = 15          # 1 Frame für Erstkalibrierung
CENTER_STABLE_FRAMES = 8   # wie lange zentriert halten, bevor zum Abstand gewechselt wird
DIST_STABLE_FRAMES   = 8   # wie lange korrekten Abstand halten, bevor Kalibrierung/Align startet

# =========================
#  LAB-FARBSEGMENTIERUNG
# =========================
PRECALIB_STABLE_FRAMES = 20
ARUCO_EXCL_PAD_FRAC  = 0.35   
MARGIN_FRAC          = 0.20        # inneres Sichtfeld (Rand wird ignoriert)
CENTER_FRAC_W        = 0.06        # Breite des zentralen Patch (relativ)
CENTER_FRAC_H        = 0.06        # Höhe des zentralen Patch (relativ)
LAB_THR              = 19.0        # ΔE(Lab)-Schwelle (10–30 üblich)
LAB_EMA_ALPHA        = 0.25        # Glättung der Referenzfarbe (0..1)

# =========================
#  MASKEN-POSTPROCESSING
# =========================
USE_MEDIAN           = True
MEDIAN_KSIZE         = 5

USE_MORPH_CLOSE      = True
CLOSE_KSIZE          = 15
CLOSE_ITERS          = 2

USE_MORPH_OPEN       = True
OPEN_KSIZE           = 9
OPEN_ITERS           = 1

KEEP_LARGEST_COMP    = True
MIN_LCC_AREA         = 50000       # kleinste Fläche für größte Komponente

FILL_HOLES           = True
USE_CONVEX_HULL      = False

# =========================
#  RECHTECK & AUSRICHTUNG
# =========================
EDGE_INSET_PX        = 30          # vor Rechteck-Ermittlung etwas erodieren
MIN_DRAW_AREA_FRAC   = 0.20        # Mindestanteil der Bildfläche zum Zeichnen
ANGLE_TOLERANCE_DEG  = 3.0         # Toleranz für „parallel“
ALIGN_MODE           = "auto"      # "auto" | "horizontal" | "vertical"
