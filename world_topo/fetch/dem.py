import gzip
import math
import os
import sys
import tempfile

import requests
import rasterio
from rasterio.merge import merge

SRTM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi"


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

    lat_start = int(math.floor(south))
    lat_end = int(math.ceil(north))
    lon_start = int(math.floor(west))
    lon_end = int(math.ceil(east))

    lat_start = max(lat_start, -56)
    lat_end = min(lat_end, 60)

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
        print(f"  Skipped {len(missing)} tile(s) over water (no data): {', '.join(missing[:3])}...",
              file=sys.stderr)

    if not tile_paths:
        raise RuntimeError(
            "No SRTM tiles found for this area. "
            "Coverage is limited to 60°N–56°S."
        )

    if len(tile_paths) == 1:
        output_path = os.path.join(work_dir, 'dem.tif')
        with rasterio.open(tile_paths[0]) as src:
            window = src.window(west, south, east, north)
            if window.width < 1 or window.height < 1:
                raise RuntimeError("DEM tile does not overlap the requested area")
            window = window.round_lengths().round_offsets()
            transform = src.window_transform(window)
            data = src.read(1, window=window)
            profile = src.profile.copy()
            profile.update(
                driver='GTiff',
                height=window.height,
                width=window.width,
                transform=transform,
            )
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data, 1)
        return output_path

    srcs = [rasterio.open(p) for p in tile_paths]
    merged_array, merged_transform = merge(srcs)
    for s in srcs:
        s.close()

    profile = rasterio.open(tile_paths[0]).profile.copy()
    profile.update(
        driver='GTiff',
        height=merged_array.shape[1],
        width=merged_array.shape[2],
        transform=merged_transform,
    )
    merged_path = os.path.join(work_dir, 'dem_merged.tif')
    with rasterio.open(merged_path, 'w', **profile) as dst:
        dst.write(merged_array)

    with rasterio.open(merged_path) as src:
        window = src.window(west, south, east, north)
        if window.width < 1 or window.height < 1:
            raise RuntimeError("Merged DEM does not overlap the requested area")
        window = window.round_lengths().round_offsets()
        transform = src.window_transform(window)
        data = src.read(1, window=window)
        profile = src.profile.copy()
        profile.update(
            driver='GTiff',
            height=window.height,
            width=window.width,
            transform=transform,
        )
        output_path = os.path.join(work_dir, 'dem.tif')
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data, 1)
    os.unlink(merged_path)

    return output_path
