import numpy as np
from config import CENTER_TOLERANCE_PX

def is_centered(marker_center, frame_shape):
    h, w = frame_shape[:2]
    img_ctr = np.array([w//2, h//2])
    mc = np.array(marker_center)
    diff = mc - img_ctr
    dirs = []
    if abs(diff[1]) > CENTER_TOLERANCE_PX:
        dirs.append('vorne' if diff[1] < 0 else 'hinten')
    if abs(diff[0]) > CENTER_TOLERANCE_PX:
        dirs.append('links' if diff[0] < 0 else 'rechts')
    return len(dirs) == 0, dirs
