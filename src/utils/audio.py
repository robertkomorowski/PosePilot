import platform, os, threading, time, shutil, subprocess
from pathlib import Path
from config import *

class SoundRateLimiter:
    def __init__(self, min_interval_sec=3.0):
        self.min_interval = float(min_interval_sec)
        self.last_time = {}

    def can_play(self, key: str) -> bool:
        now = time.time()
        t = self.last_time.get(key, 0.0)
        if now - t >= self.min_interval:
            self.last_time[key] = now
            return True
        return False

def play_sound_file(filepath):
    """
    Spielt eine Sounddatei plattformabhängig ab.
    -> Unter Windows ist WAV (PCM 16-bit) am sichersten.
    """
    p = Path(filepath).resolve()
    if not p.exists():
        print(f"[sound] Datei nicht gefunden: {p}")
        return

    sysname = platform.system()
    ext = p.suffix.lower()

    try:
        if sysname == 'Windows':
            # Bevorzugt: winsound (nur WAV)
            if ext == '.wav':
                try:
                    import winsound
                    winsound.PlaySound(str(p), winsound.SND_FILENAME | winsound.SND_ASYNC)
                    return
                except Exception as e:
                    print(f"[sound] winsound Fehler: {e}")

            # Fallback: PowerShell (WAV) blockiert intern bis Ende, aber wir starten separat
            if shutil.which('powershell') and ext == '.wav':
                subprocess.Popen([
                    'powershell', '-NoProfile', '-Command',
                    f'[System.Media.SoundPlayer]::new("{str(p)}").PlaySync()'
                ])
                return

            # Letzter Fallback: mit Standard-App öffnen (kann Fenster aufpoppen)
            subprocess.Popen(['cmd', '/c', 'start', '', str(p)])
            return

        elif sysname == 'Darwin':  # macOS
            subprocess.Popen(['afplay', str(p)])
            return

        else:  # Linux
            if shutil.which('paplay'):
                subprocess.Popen(['paplay', str(p)])
                return
            if shutil.which('aplay'):
                subprocess.Popen(['aplay', str(p)])
                return
            if shutil.which('ffplay'):
                subprocess.Popen(['ffplay', '-nodisp', '-autoexit', str(p)])
                return
            print("[sound] Kein Audio-Player (paplay/aplay/ffplay) gefunden.")

    except Exception as e:
        print(f"[sound] Fehler beim Abspielen: {e}")

SOUND_PATHS = {
    'vorne':                   'data/sounds/vorne.wav',
    'hinten':                  'data/sounds/hinten.wav',
    'links':                  'data/sounds/links.wav',
    'rechts':                 'data/sounds/rechts.wav',
    'vorne_rechts':            'data/sounds/vorne_rechts.wav',
    'hinten_rechts':           'data/sounds/hinten_rechts.wav',
    'vorne_links':             'data/sounds/vorne_links.wav',
    'hinten_links':            'data/sounds/hinten_links.wav',
    'drehen_rechts':          'data/sounds/drehen_rechts.wav',
    'drehen_links':           'data/sounds/drehen_links.wav',
    'ausgerichtet':           'data/sounds/ausgerichtet.wav',
    'kalibrieren_lab':        'data/sounds/kalibrieren_lab.wav',
    'warte_vor_kalibrierung': 'data/sounds/warte_vor_kalibrierung.wav',
    'warte_nach_ausrichtung': 'data/sounds/warte_nach_ausrichtung.wav',
}

def _play_dir_sound_async(direction: str, limiter: SoundRateLimiter = SoundRateLimiter(min_interval_sec=3.0)):
    path = SOUND_PATHS.get(direction)
    if not path:
        print(f"[sound] Kein Mapping für Key: {direction}")
        return
    if limiter.can_play(direction):
        threading.Thread(target=play_sound_file, args=(path,), daemon=True).start()
