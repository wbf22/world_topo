import numpy as np


def compute_hillshade(elevation, azimuth=315, altitude=45, ambient=0.3,
                      resolution=None):
    dy, dx = np.gradient(elevation)
    if resolution is not None:
        y_res, x_res = resolution
        dx = dx / x_res
        dy = dy / y_res
    aspect = np.arctan2(-dy, dx)
    slope = np.arctan(np.sqrt(dx * dx + dy * dy))

    azimuth_rad = np.radians(360 - azimuth)
    altitude_rad = np.radians(altitude)

    slope_min = 0.01
    slope_max = 0.05
    t = np.clip((slope - slope_min) / (slope_max - slope_min), 0, 1)
    aspect_weight = t * t * (3 - 2 * t)

    shaded = (np.cos(altitude_rad) * np.cos(slope)
              + aspect_weight * np.sin(altitude_rad) * np.sin(slope)
              * np.cos(azimuth_rad - aspect))

    shaded = np.clip(shaded, 0, 1)
    return ambient + (1 - ambient) * shaded
