import numpy as np


def compute_hillshade(elevation, azimuth=315, altitude=45, ambient=0.3):
    dy, dx = np.gradient(elevation)
    aspect = np.arctan2(-dy, dx)
    slope = np.arctan(np.sqrt(dx * dx + dy * dy))

    azimuth_rad = np.radians(360 - azimuth)
    altitude_rad = np.radians(altitude)

    shaded = (np.sin(altitude_rad) * np.sin(slope)
              + np.cos(altitude_rad) * np.cos(slope)
              * np.cos(azimuth_rad - aspect))

    shaded = np.clip(shaded, 0, 1)
    return ambient + (1 - ambient) * shaded
