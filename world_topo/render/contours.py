import numpy as np


def add_contours(ax, dem, extent, interval=100):
    xmin, xmax, ymin, ymax = extent
    ny, nx = dem.shape

    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    X = np.linspace(xmin + dx / 2, xmax - dx / 2, nx)
    Y = np.linspace(ymax - dy / 2, ymin + dy / 2, ny)

    vmin = np.floor(np.nanmin(dem) / interval) * interval
    vmax = np.ceil(np.nanmax(dem) / interval) * interval
    levels = np.arange(vmin, vmax + interval, interval)

    cs = ax.contour(X, Y, dem, levels=levels, colors='black', linewidths=0.3, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=6, fmt='%d')
