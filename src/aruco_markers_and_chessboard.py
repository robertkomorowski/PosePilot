import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Einstellungen
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
marker_id = 0  # Fester Marker mit ID 0
marker_sizes_mm = [40, 60, 80, 100]  # Verschiedene Größen in mm
dpi = 300  # Druckauflösung

# PDF-Pfad
pdf_path = "./data/aruco_markers_and_chessboard.pdf"
pdf = PdfPages(pdf_path)

# ArUco Marker in verschiedenen Größen generieren
for size in marker_sizes_mm:
    # Berechne die Pixel-Größe basierend auf der gewünschten physischen Größe
    pixels = int(size * dpi / 25.4)  # Konvertiere mm in Pixel
    
    # Marker generieren mit der berechneten Pixel-Größe
    marker_image = aruco_dict.generateImageMarker(marker_id, pixels)
    
    # Figur mit korrekter Größe für den Druck erstellen
    fig, ax = plt.subplots(figsize=(size / 25.4, size / 25.4), dpi=dpi)
    ax.imshow(marker_image, cmap='gray')
    ax.axis('off')
    
    # Speichern mit korrekten Rändern
    pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# Schachbrettmuster (7x10 Ecken, 8x11 Felder)
rows, cols = 7, 10
square_size_mm = 30
# Konvertiere mm in Pixel unter Berücksichtigung der DPI
square_size_px = int(square_size_mm * dpi / 25.4)  # 25.4 mm = 1 inch
board_size_px = ((rows + 1) * square_size_px, (cols + 1) * square_size_px)
board = np.zeros(board_size_px, dtype=np.uint8)

for i in range(rows + 1):
    for j in range(cols + 1):
        if (i + j) % 2 == 0:
            y = i * square_size_px
            x = j * square_size_px
            board[y:y+square_size_px, x:x+square_size_px] = 255

fig, ax = plt.subplots(figsize=(board.shape[1] / dpi, board.shape[0] / dpi), dpi=dpi)
ax.imshow(board, cmap='gray')
ax.axis('off')
pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
plt.close(fig)

pdf.close()
print(f"PDF gespeichert unter: {pdf_path}")
