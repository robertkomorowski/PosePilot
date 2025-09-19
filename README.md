# Kameraausrichtung – Live-Ausrichtung einer Kamera (DE/EN)

> **Projektzweck**  
> Dieses Repository enthält eine Python‑Anwendung zur **Live‑Ausrichtung einer Kamera** über visuelle Overlays und **Sprachhinweise**. Für die Zentrierung und Abstandsbestimmung wird ein **ArUco‑Marker** verwendet; die **Parallelität (z‑Rotation)** wird über die **Segmentierung der grünen Bettfläche** im CIELAB‑Farbraum ermittelt. Das System ist als Komponente im Kontext des CP‑Diadem‑Projekts gedacht und arbeitet **lokal in Echtzeit**.

---

## 🇩🇪 Deutsch

### Features
- **Echtzeit‑Feedback** (OpenCV): Overlays für *Zentrierung*, *Abstand/Höhe* und *Rotation/Parallelität*  
- **Sprachhinweise** (deutsch): z. B. „oben“, „rechts“, „drehen links“, „ausgerichtet“ (siehe `data/sounds/`)  
- **Robuste Marker‑Erkennung**: ArUco 4x4‑50; Zentrierung + z‑Abstand via `solvePnP(IPPE_SQUARE)`  
- **Bettflächen‑Analyse (CIELAB)**: ΔE‑basierte Segmentierung mit EMA‑Refarbbestimmung und Morphologie  
- **Phasenlogik**: `CENTER → DIST → CALIB (Lab) → ALIGN`; Haltevorgaben für stabile Zustandswechsel  
- **Validierungs‑Modus**: `src/validation.py` speichert *genau 2* Screenshots + CSV (Winkelvergleich Code↔Bild)  
- **Fenster‑Auto‑Fit**: `utils/window_fit.py` passt die OpenCV‑Fenstergröße an die Bildschirmgröße an

### Verzeichnisstruktur
```
Kameraausrichtung_BA/
├─ src/
│  ├─ main.py                  # Hauptroutine (Live‑Ausrichtung + UI/Sounds)
│  ├─ validation.py            # 1‑Run‑Validierung (2 Screenshots + CSV)
│  ├─ config.py                # Zentrale Konfiguration/Schwellen
│  └─ utils/
│     ├─ bed_detection.py      # CIELAB‑Segmentierung + Hauptachse
│     ├─ calibration.py        # Laden der Kamera‑Intrinsics (NPZ)
│     ├─ draw_feedback.py      # Text, Pfeile, Rotation, Rate‑Limiter
│     ├─ marker_utils.py       # Zentrier‑Check
│     ├─ audio.py              # plattformabhängiges Abspielen von WAV
│     └─ window_fit.py         # Fenster‑Sizing
├─ data/
│  ├─ calibration/
│  │  ├─ images/               # (eigene Kalibrierfotos)
│  │  └─ intrinsics/
│  │     └─ camera_calibration_data.npz  # benötigt zur Laufzeit
│  ├─ sounds/                  # deutsche Sprachprompts (WAV)
│  └─ sounds_pack/             # Generator‑Skripte/Phrazes
└─ doc/                        # Marker/Chessboard‑Vorlagen (PDF) requirements.txt und README.md
```

### Systemvoraussetzungen
- **Python** 3.10–3.12
- **Pakete**:  
  ```bash
  pip install "opencv-contrib-python>=4.8" numpy pillow
  ```
  > *Hinweis:* `opencv-contrib-python` ist erforderlich (ArUco).  
- **Audio‑Tools** (für Sounds; je nach OS vorinstalliert):
  - Windows: `winsound` (WAV) – integriert
  - macOS: `afplay` (oder `say`) – vorinstalliert
  - Linux: `paplay` / `aplay` oder `ffplay` (bei Bedarf installieren)

### Kamerakalibrierung *(nur bei neuer/anderer Kamera oder geänderter Optik)*
> Wenn bereits eine **passende Intrinsics‑Datei** vorhanden ist (`data/calibration/intrinsics/camera_calibration_data.npz`), **kann dieser Abschnitt übersprungen werden**.  
> Eine Kalibrierung ist **kamera‑ und fokusspezifisch**; bei unveränderter Kamera (Brennweite/Zoom) genügt eine **einmalige** Kalibrierung.

1. **Kalibrierbilder aufnehmen** (Schachbrett **7×10 innere Ecken**, **19 mm** Kantenlänge):  
   ```bash
   python data/calibration/capture_chessboard_images.py
   ```
2. **Kalibrieren** und NPZ erzeugen:  
   ```bash
   python data/calibration/calibrate_camera.py
   # erzeugt: data/calibration/intrinsics/camera_calibration_data.npz
   ```
3. Optional: **Optimierung/Evaluierung** siehe weitere Skripte in `data/calibration/`.

### ArUco‑Marker & Setup
- **Marker:** ArUco **4x4‑50**, **Kantenlänge 62 mm** (siehe `doc/`), mittig auf dem Bett.  
- **Zielabstand:** `TARGET_DISTANCE_M = 1.03` m (siehe `src/config.py`).  
- **Bettlaken:** homogener **Grünton** (für die Segmentierung).  
- **Offset:** `CALIB_OFFSET_M = 0.27` kompensiert den Ebenenversatz Marker↔Bett.

### Nutzung (Live‑Ausrichtung)
```bash
python src/main.py
```
- Folgen Sie den Sprach- und Bildhinweisen.  
- Phasen: **Zentrieren** → **Abstand halten** → **Lab‑Kalibrierung** → **Rotation**.  
- Bei stabiler Ausrichtung („AUSGERICHTET“) ertönt ein Hinweis; die Box wird grün.

### Validierung (1 Lauf, 2 Screenshots + CSV)
```bash
python src/validation.py
```
- Erstes Bild bei **zentriert**, zweites bei **stabil ausgerichtet**.  
- Ergebnisdateien in `runs/`:
  - `runs/events.csv`
  - `runs/shots/centered_*.png`, `runs/shots/angle_ok_*.png`

### Konfiguration
Alle wichtigen Parameter finden sich in `src/config.py`, u. a.:
- **Marker/Geometrie:** `CENTER_TOLERANCE_PX`, `TARGET_DISTANCE_M`, `DISTANCE_TOLERANCE_M`, `MARKER_LENGTH_M`, `CALIB_OFFSET_M`
- **CIELAB‑Segmentierung:** `LAB_THR`, `LAB_EMA_ALPHA`, `MARGIN_FRAC`, Morphologie‑Kernel/Iterationen
- **Ausrichtung:** `ANGLE_TOLERANCE_DEG`, `ALIGN_MODE`
- **Phasen/Haltezeiten:** `CENTER_STABLE_FRAMES`, `DIST_STABLE_FRAMES`, `STABLE_ALIGN_SEC`

### Tipps & Troubleshooting
- **„Kalibrierdatei nicht gefunden“** → Lege eine bestehende **NPZ** in `data/calibration/intrinsics/` ab **oder** kalibriere **einmalig** (nur bei neuer/anderer Kamera).  
- **Kein Marker erkannt** → Beleuchtung, Druckqualität, Größe/Distanz prüfen; ArUco 4x4‑50 verwenden.  
- **„Kein Bett erkannt“/Instabile Rotation** → `LAB_THR` erhöhen (20–30 testen), `MARGIN_FRAC`/Morphologie anpassen, homogenes grünes Laken nutzen.  
- **Keine Sounds** → Linux: `paplay`/`aplay`/`ffplay` installieren; Pfade in `utils/audio.py` anpassen.  
- **Fenstergröße** → `utils/window_fit.py` ggf. Basis‑Auflösung (`BASE_SCREEN_W/H`) anpassen.

### Lizenz & Zitation  
- **Zitation/Referenz:** Diese Software entstand im Rahmen einer Bachelorarbeit (CP‑Diadem‑Kontext).

---

## 🇬🇧 English

### Purpose
This repository provides a Python application for **real‑time camera alignment** with visual overlays and **spoken prompts**. **ArUco markers** are used for **centering** and **z‑distance**; **parallelism (z‑rotation)** is estimated via **green bed surface segmentation** in the CIELAB color space. The component is intended to run **locally in real time** and to be embedded into the CP‑Diadem workflow.

### Features
- **Real‑time feedback** (OpenCV): overlays for *centering*, *distance/height*, *rotation/parallelism*  
- **Spoken prompts** (German, WAV in `data/sounds/`)  
- **Robust marker detection**: ArUco 4x4‑50; `solvePnP(IPPE_SQUARE)` for z‑distance  
- **Bed analysis (CIELAB)**: ΔE segmentation with EMA reference color + morphology  
- **Phase logic**: `CENTER → DIST → CALIB (Lab) → ALIGN` with hold requirements  
- **Validation mode**: `src/validation.py` stores *exactly two* screenshots + a CSV (code vs. screenshot angle)  
- **Auto‑sized window**: `utils/window_fit.py` scales the OpenCV window to your screen

### Requirements
- **Python** 3.10–3.12
- **Packages**:  
  ```bash
  pip install "opencv-contrib-python>=4.8" numpy pillow
  ```
- **Audio tools** (for WAV playback):
  - Windows: `winsound` (built‑in)
  - macOS: `afplay` (or `say`)
  - Linux: `paplay` / `aplay` or `ffplay`

### Camera Calibration *(only for a new/different camera or changed optics)*
> If you already have a **matching intrinsics file** (`data/calibration/intrinsics/camera_calibration_data.npz`), you can **skip calibration**.  
> Calibration is **camera‑ and focus‑specific**; with a fixed camera (focal length/zoom), you typically calibrate **once**.

1. **Capture chessboard images** (**7×10 inner corners**, **19 mm** square size):  
   ```bash
   python data/calibration/capture_chessboard_images.py
   ```
2. **Calibrate** and create NPZ:  
   ```bash
   python data/calibration/calibrate_camera.py
   # outputs: data/calibration/intrinsics/camera_calibration_data.npz
   ```

### ArUco Marker & Setup
- **Marker:** ArUco **4x4‑50**, **62 mm** edge length (see `doc/`), placed at bed center.  
- **Target distance:** `TARGET_DISTANCE_M = 1.03` m.  
- **Bed sheet:** homogeneous **green** surface for segmentation.  
- **Offset:** `CALIB_OFFSET_M = 0.27` compensates the marker‑to‑bed plane gap.

### Run (Live Alignment)
```bash
python src/main.py
```
Follow the on‑screen & audio guidance. Phases: **Centering** → **Hold distance** → **Lab calibration** → **Rotation**. When aligned, the box turns green and a spoken cue confirms success.

### Validation (1 run, 2 screenshots + CSV)
```bash
python src/validation.py
```
- First screenshot at **centered**, second at **stable aligned**.  
- Outputs in `runs/`:
  - `runs/events.csv`
  - `runs/shots/centered_*.png`, `runs/shots/angle_ok_*.png`

### Configuration
See `src/config.py` for all key parameters:
- **Marker/geometry:** `CENTER_TOLERANCE_PX`, `TARGET_DISTANCE_M`, `DISTANCE_TOLERANCE_M`, `MARKER_LENGTH_M`, `CALIB_OFFSET_M`
- **CIELAB segmentation:** `LAB_THR`, `LAB_EMA_ALPHA`, `MARGIN_FRAC` and morphology settings
- **Alignment:** `ANGLE_TOLERANCE_DEG`, `ALIGN_MODE`
- **Phase/hold times:** `CENTER_STABLE_FRAMES`, `DIST_STABLE_FRAMES`, `STABLE_ALIGN_SEC`

### Tips & Troubleshooting
- **“Calibration file not found”** → Place an existing **NPZ** in `data/calibration/intrinsics/` **or** calibrate **once** (only for a new/different camera).  
- **Marker not detected** → check lighting/print quality/size/distance; use ArUco 4x4‑50.  
- **“No bed detected”/unstable rotation** → increase `LAB_THR` (try 20–30), adjust `MARGIN_FRAC`/morphology, ensure a uniform green sheet.  
- **No audio** → on Linux, install `paplay`/`aplay`/`ffplay`; adjust paths in `utils/audio.py` if needed.  
- **Window size** → tweak `utils/window_fit.py` base resolution if necessary.

### License & Citation 
- **Citation/Reference:** Developed as part of a Bachelor’s project (CP‑Diadem context).
