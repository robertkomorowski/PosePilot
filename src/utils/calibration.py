import os
import numpy as np

def load_calibration():
    path = os.path.join("data", "calibration", "intrinsics", "camera_calibration_data.npz")
    if not os.path.exists(path):
        raise FileNotFoundError("Kalibrierdatei nicht gefunden: " + path)
    data = np.load(path)
    return data['camera_matrix'], data['dist_coeffs']
