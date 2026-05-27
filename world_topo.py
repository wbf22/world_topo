#!/usr/bin/env python3
"""Generate elevation maps from satellite data.

Usage:
    python world_topo.py --lat 46.85 --lon -121.76 --width 0.08 --height 0.08 -o map.png
    python world_topo.py --lat 40.511 --lon -112.258 --height 0.15 --no-imagery --clean
"""

import argparse
import gzip
import math
import os
import sys
import tempfile

import numpy as np
import rasterio
import requests
from matplotlib import rcParams, pyplot as plt
from rasterio.merge import merge
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds

SRTM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi"
MAX_OUTPUT_PIXELS = 50_000_000
EARTH_ELEVATION_MIN = -500
EARTH_ELEVATION_MAX = 9000

rcParams['figure.dpi'] = 150


# ── DEM fetching ──────────────────────────────────────────────────────────

def _tile_name(lat_idx, lon_idx):
    lat_p = 'N' if lat_idx >= 0 else 'S'
    lon_p = 'E' if lon_idx >= 0 else 'W'
    return f"{lat_p}{abs(lat_idx):02d}{lon_p}{abs(lon_idx):03d}"


def _lat_dir(lat_idx):
    lat_p = 'N' if lat_idx >= 0 else 'S'
    return f"{lat_p}{abs(lat_idx):02d}"


def _download_tile(lat_idx, lon_idx, cache_dir):
    name = _tile_name(lat_idx, lon_idx)
    hgt_path = os.path.join(cache_dir, f"{name}.hgt")
    if os.path.exists(hgt_path):
        return hgt_path
    for ext in ('.hgt.gz', '.hgt'):
        url = f"{SRTM_URL}/{_lat_dir(lat_idx)}/{name}{ext}"
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            if ext == '.hgt.gz':
                gz_path = os.path.join(cache_dir, f"{name}.hgt.gz")
                with open(gz_path, 'wb') as f:
                    for chunk in resp.iter_content(65536):
                        f.write(chunk)
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(hgt_path, 'wb') as f_out:
                        f_out.write(f_in.read())
                os.unlink(gz_path)
            else:
                with open(hgt_path, 'wb') as f:
                    for chunk in resp.iter_content(65536):
                        f.write(chunk)
            return hgt_path
    return None


def fetch_dem(west, south, east, north, work_dir=None):
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix='world_topo_dem_')
    lat_start = max(int(math.floor(south)), -56)
    lat_end = min(int(math.ceil(north)), 60)
    lon_start = int(math.floor(west))
    lon_end = int(math.ceil(east))
    tile_paths = []
    missing = []
    for lat in range(lat_start, lat_end):
        for lon in range(lon_start, lon_end):
            path = _download_tile(lat, lon, work_dir)
            if path:
                tile_paths.append(path)
            else:
                missing.append(_tile_name(lat, lon))
    if missing:
        print(f"  Skipped {len(missing)} tile(s) over water: {', '.join(missing[:3])}...",
              file=sys.stderr)
    if not tile_paths:
        raise RuntimeError("No SRTM tiles found. Coverage: 60°N–56°S.")
    if len(tile_paths) == 1:
        output_path = os.path.join(work_dir, 'dem.tif')
        with rasterio.open(tile_paths[0]) as src:
            window = src.window(west, south, east, north)
            if window.width < 1 or window.height < 1:
                raise RuntimeError("DEM tile does not overlap the requested area")
            window = window.round_lengths().round_offsets()
            data = src.read(1, window=window)
            profile = src.profile.copy()
            profile.update(driver='GTiff', height=window.height,
                           width=window.width, transform=src.window_transform(window))
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data, 1)
        return output_path
    srcs = [rasterio.open(p) for p in tile_paths]
    merged_array, merged_transform = merge(srcs)
    for s in srcs:
        s.close()
    profile = rasterio.open(tile_paths[0]).profile.copy()
    profile.update(driver='GTiff', height=merged_array.shape[1],
                   width=merged_array.shape[2], transform=merged_transform)
    merged_path = os.path.join(work_dir, 'dem_merged.tif')
    with rasterio.open(merged_path, 'w', **profile) as dst:
        dst.write(merged_array)
    with rasterio.open(merged_path) as src:
        window = src.window(west, south, east, north)
        if window.width < 1 or window.height < 1:
            raise RuntimeError("Merged DEM does not overlap the requested area")
        window = window.round_lengths().round_offsets()
        data = src.read(1, window=window)
        profile = src.profile.copy()
        profile.update(driver='GTiff', height=window.height,
                       width=window.width, transform=src.window_transform(window))
        output_path = os.path.join(work_dir, 'dem.tif')
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data, 1)
    os.unlink(merged_path)
    return output_path


# ── Satellite imagery ─────────────────────────────────────────────────────

def _window_in_src_crs(src, west, south, east, north):
    if src.crs and src.crs.to_string() != 'EPSG:4326':
        try:
            left, bottom, right, top = transform_bounds(
                'EPSG:4326', src.crs, west, south, east, north)
        except Exception:
            return None
    else:
        left, bottom, right, top = west, south, east, north
    margin = 0.02 * max(right - left, top - bottom)
    window = src.window(left - margin, bottom - margin, right + margin, top + margin)
    if window.width < 10 or window.height < 10:
        return None
    return window.round_lengths().round_offsets()


def fetch_imagery(west, south, east, north, work_dir=None):
    try:
        import pystac_client
        import planetary_computer
    except ImportError:
        print("  pystac-client / planetary-computer not installed; skipping imagery",
              file=sys.stderr)
        return None
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix='world_topo_imagery_')
    try:
        print("  Searching for cloud-free Sentinel-2 scene...", file=sys.stderr)
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace)
        search = catalog.search(
            bbox=[west, south, east, north],
            collections=["sentinel-2-l2a"],
            max_items=3,
            query={"eo:cloud_cover": {"lt": 20}},
            sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}])
        items = list(search.items())
        if not items:
            print("  No suitable imagery found", file=sys.stderr)
            return None
        item = items[0]
        cc = item.properties.get('eo:cloud_cover', '?')
        print(f"  Scene: {item.id} (cloud cover: {cc}%)", file=sys.stderr)
        href = item.assets.get('true-color', item.assets.get('visual', None))
        if href is None:
            print("  No RGB visual asset available", file=sys.stderr)
            return None
        href = href.href
        output_path = os.path.join(work_dir, 'imagery.tif')
        print("  Downloading imagery...", file=sys.stderr)
        with rasterio.open(href) as src:
            window = _window_in_src_crs(src, west, south, east, north)
            if window is None:
                print("  Could not locate area in imagery tile", file=sys.stderr)
                return None
            data = src.read(window=window)
            src_transform = src.window_transform(window)
            if src.crs and src.crs.to_string() != 'EPSG:4326':
                dst_crs = 'EPSG:4326'
                win_left, win_bottom, win_right, win_top = array_bounds(
                    window.height, window.width, src_transform)
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    src.crs, dst_crs, window.width, window.height,
                    win_left, win_bottom, win_right, win_top)
                if dst_width * dst_height > MAX_OUTPUT_PIXELS:
                    scale = (MAX_OUTPUT_PIXELS / (dst_width * dst_height)) ** 0.5
                    dst_width = max(1, int(dst_width * scale))
                    dst_height = max(1, int(dst_height * scale))
                    dst_transform, dst_width, dst_height = calculate_default_transform(
                        src.crs, dst_crs, window.width, window.height,
                        win_left, win_bottom, win_right, win_top,
                        dst_width=dst_width, dst_height=dst_height)
                profile = src.profile.copy()
                profile.update(driver='GTiff', crs=dst_crs, transform=dst_transform,
                               width=dst_width, height=dst_height,
                               blockxsize=256, blockysize=256, tiled=True)
                with rasterio.open(output_path, 'w', **profile) as dst:
                    for i in range(1, data.shape[0] + 1):
                        reproject(source=data[i - 1], destination=rasterio.band(dst, i),
                                  src_transform=src_transform, src_crs=src.crs,
                                  dst_transform=dst_transform, dst_crs=dst_crs,
                                  resampling=Resampling.bilinear)
            else:
                profile = src.profile.copy()
                profile.update(driver='GTiff', height=window.height, width=window.width,
                               transform=src_transform, blockxsize=256, blockysize=256, tiled=True)
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(data)
        print(f"  Imagery saved to: {output_path}", file=sys.stderr)
        return output_path
    except Exception as e:
        print(f"  Could not fetch satellite imagery: {e}", file=sys.stderr)
        return None


# ── Hillshade ─────────────────────────────────────────────────────────────

def compute_hillshade(elevation, azimuth=315, altitude=45, ambient=0.3, resolution=None):
    dy, dx = np.gradient(elevation)
    if resolution is not None:
        y_res, x_res = resolution
        dx /= x_res
        dy /= y_res
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


# ── Contours ──────────────────────────────────────────────────────────────

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


# ── Color elevation overlay ──────────────────────────────────────────────

def color_elevation_map(ax, dem, extent, cmap='terrain', alpha=0.35,
                        vmin=EARTH_ELEVATION_MIN, vmax=EARTH_ELEVATION_MAX,
                        colorbar=True):
    im = ax.imshow(dem, extent=extent, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
    if colorbar:
        plt.colorbar(im, ax=ax, label='Elevation (m)', shrink=0.8)


# ── Orchestrator ──────────────────────────────────────────────────────────

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
        size = (12 / 0.1) * min(width, height)
        figsize = (size, size * aspect_ratio)
        fig, ax = plt.subplots(1, 1, figsize=figsize)

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
                    img_extent = [src.bounds.left, src.bounds.right,
                                  src.bounds.bottom, src.bounds.top]
                    ax.imshow(np.transpose(rgb, (1, 2, 0)), extent=img_extent, alpha=0.9)

        center_lat_rad = np.radians((bounds.top + bounds.bottom) / 2)
        dlon = (bounds.right - bounds.left) / nx
        dlat = (bounds.top - bounds.bottom) / ny
        x_res = dlon * 111320 * np.cos(center_lat_rad)
        y_res = dlat * 111320

        shade = compute_hillshade(dem, resolution=(y_res, x_res))
        shade_alpha = 0.35 if imagery_path else 0.85
        ax.imshow(shade, extent=extent, cmap='gray', alpha=shade_alpha)

        color_elevation_map(ax, dem, extent, alpha=0.35, colorbar=not clean)
        add_contours(ax, dem, extent, interval=contour_interval)

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

        plt.savefig(output, dpi=150, bbox_inches='tight',
                    pad_inches=0 if clean else 0.1)
        plt.close()

    print(f"Done — map saved to: {output}", file=sys.stderr)


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Generate elevation maps from satellite data.')
    parser.add_argument('--lat', type=float, required=True,
                        help='Center latitude')
    parser.add_argument('--lon', type=float, required=True,
                        help='Center longitude')
    parser.add_argument('--width', type=float, default=0.15,
                        help='Width in degrees (default: 0.15)')
    parser.add_argument('--height', type=float, default=0.15,
                        help='Height in degrees (default: 0.15)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output image path (default: auto-named)')
    parser.add_argument('--contour-interval', type=float, default=100,
                        help='Contour interval in meters (default: 100)')
    parser.add_argument('--no-imagery', action='store_true',
                        help='Skip satellite imagery underlay')
    parser.add_argument('--clean', action='store_true',
                        help='No labels, axis, legend, or title')
    parser.add_argument('--debug', action='store_true',
                        help='Print DEM stats')
    return parser.parse_args(argv)


def main():
    args = parse_args()
    output = args.output
    if output is None:
        lat_s = f"{args.lat:+.4f}".replace("+", "n").replace("-", "s")
        lon_s = f"{args.lon:+.4f}".replace("+", "e").replace("-", "w")
        output = f"elevation_{lat_s}_{lon_s}_{args.width}x{args.height}.png"
    make_elevation_map(lat=args.lat, lon=args.lon, width=args.width,
                       height=args.height, output=output,
                       contour_interval=args.contour_interval,
                       no_imagery=args.no_imagery, clean=args.clean,
                       debug=args.debug)


if __name__ == '__main__':
    main()
