import numpy as np
import matplotlib.pyplot as plt


EARTH_ELEVATION_MIN = -500
EARTH_ELEVATION_MAX = 9000


def color_elevation_map(ax, dem, extent, cmap='terrain', alpha=0.35,
                        vmin=EARTH_ELEVATION_MIN, vmax=EARTH_ELEVATION_MAX,
                        colorbar=True):
    im = ax.imshow(dem, extent=extent, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
    if colorbar:
        plt.colorbar(im, ax=ax, label='Elevation (m)', shrink=0.8)
