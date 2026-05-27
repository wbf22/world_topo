import os
import sys
import tempfile

import numpy as np
import rasterio
from matplotlib import rcParams, pyplot as plt
from matplotlib.colors import Normalize

from world_topo.fetch.dem import fetch_dem
from world_topo.fetch.imagery import fetch_imagery
from world_topo.render.hillshade import compute_hillshade
from world_topo.render.contours import add_contours

rcParams['figure.dpi'] = 150


def make_elevation_map(lat, lon, width, height, output,
                       contour_interval=100, no_imagery=False, clean=False, debug=False):
    west = lon - width / 2
    east = lon + width / 2
    south = lat - height / 2
    north = lat + height / 2

    print(f"Area: {lat:.4f}°, {lon:.4f}°  (±{width/2:.4f}° × ±{height/2:.4f}°)",
          file=sys.stderr)
    print(file=sys.stderr)

    with tempfile.TemporaryDirectory(prefix='world_topo_') as tmp_dir:
        dem_path = fetch_dem(west, south, east, north, tmp_dir)

        imagery_path = None
        if not no_imagery:
            imagery_path = fetch_imagery(west, south, east, north, tmp_dir)

        print(file=sys.stderr)
        print("Rendering...", file=sys.stderr)

        with rasterio.open(dem_path) as src:
            dem = src.read(1).astype(np.float64)
            nodata = src.nodata
            if nodata is not None:
                dem[dem == nodata] = np.nan
            bounds = src.bounds
            extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

        ny, nx = dem.shape
        if debug:
            nan_count = np.count_nonzero(np.isnan(dem))
            valid = dem[~np.isnan(dem)]
            print(f"  DEM: {nx}x{ny}, valid pixels: {len(valid)}, "
                  f"NaN: {nan_count} ({100*nan_count/(ny*nx):.1f}%)",
                  file=sys.stderr)
            if len(valid):
                print(f"  DEM range: {valid.min():.0f}m to {valid.max():.0f}m",
                      file=sys.stderr)
                print(f"  DEM mean: {valid.mean():.0f}m, std: {valid.std():.0f}m",
                      file=sys.stderr)

        aspect_ratio = ny / nx
        figsize = (12, 12 * aspect_ratio)
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # --- Render base: satellite imagery underlay ---
        if imagery_path and os.path.exists(imagery_path):
            with rasterio.open(imagery_path) as src:
                if src.count >= 3:
                    rgb = src.read([1, 2, 3]).astype(np.float32)
                    for i in range(3):
                        band = rgb[i]
                        p2, p98 = np.nanpercentile(band, (2, 98))
                        if p98 > p2:
                            rgb[i] = np.clip((band - p2) / (p98 - p2), 0, 1)
                    rgb = np.clip(rgb, 0, 1)
                    img_extent = [
                        src.bounds.left, src.bounds.right,
                        src.bounds.bottom, src.bounds.top,
                    ]
                    ax.imshow(np.transpose(rgb, (1, 2, 0)),
                              extent=img_extent, alpha=0.9)

        # --- Render middle: terrain elevation + hillshade composite ---
        shade = compute_hillshade(dem)

        norm = Normalize(vmin=-500, vmax=9000)
        colors = plt.cm.terrain(norm(dem))

        nan_mask = np.isnan(dem)
        if nan_mask.any():
            colors[nan_mask] = (0, 0, 0, 0)

        shade_stack = np.stack(
            [shade, shade, shade, np.ones_like(shade)], axis=2,
        )
        colors *= shade_stack

        composite_alpha = 0.55 if imagery_path else 1.0
        ax.imshow(colors, extent=extent, alpha=composite_alpha)

        if not clean:
            sm = plt.cm.ScalarMappable(norm=norm, cmap='terrain')
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label='Elevation (m)', shrink=0.8)

        # --- Render top: contour lines ---
        add_contours(ax, dem, extent, interval=contour_interval)

        # --- Labels & output ---
        if clean:
            ax.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        else:
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            center_lat = (bounds.top + bounds.bottom) / 2
            center_lon = (bounds.left + bounds.right) / 2
            ax.set_title(f'Elevation Map ({center_lat:.4f}°, {center_lon:.4f}°)')
            plt.tight_layout()

        # --- Debug: save individual layers ---
        if debug:
            fig_dem, ax_dem = plt.subplots(1, 1, figsize=figsize)
            ax_dem.imshow(dem, extent=extent, cmap='terrain',
                          vmin=-500, vmax=9000)
            ax_dem.set_title('DEM (raw)')
            fig_dem.savefig(output.replace('.png', '_dem.png'), dpi=150,
                            bbox_inches='tight')
            plt.close(fig_dem)

            fig_shade, ax_shade = plt.subplots(1, 1, figsize=figsize)
            ax_shade.imshow(shade, extent=extent, cmap='gray')
            ax_shade.set_title('Hillshade')
            fig_shade.savefig(output.replace('.png', '_hillshade.png'), dpi=150,
                              bbox_inches='tight')
            plt.close(fig_shade)

        plt.savefig(output, dpi=150, bbox_inches='tight',
                    pad_inches=0 if clean else 0.1)
        plt.close()

    print(f"Done — map saved to: {output}", file=sys.stderr)
