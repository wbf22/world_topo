import os
import sys
import tempfile

import rasterio
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds

MAX_OUTPUT_PIXELS = 50_000_000


def _window_in_src_crs(src, west, south, east, north):
    src_crs = src.crs
    if src_crs and src_crs.to_string() != 'EPSG:4326':
        try:
            left, bottom, right, top = transform_bounds(
                'EPSG:4326', src_crs, west, south, east, north,
            )
        except Exception:
            return None
    else:
        left, bottom, right, top = west, south, east, north

    margin = 0.02 * max(right - left, top - bottom)
    left -= margin
    bottom -= margin
    right += margin
    top += margin

    window = src.window(left, bottom, right, top)
    if window.width < 10 or window.height < 10:
        return None

    window = window.round_lengths().round_offsets()
    return window


def fetch_imagery(west, south, east, north, work_dir=None):
    try:
        import pystac_client
        import planetary_computer
    except ImportError:
        print("  pystac-client or planetary-computer not installed; skipping imagery",
              file=sys.stderr)
        return None

    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix='world_topo_imagery_')

    try:
        print("  Searching for cloud-free Sentinel-2 scene...", file=sys.stderr)

        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )

        search = catalog.search(
            bbox=[west, south, east, north],
            collections=["sentinel-2-l2a"],
            max_items=3,
            query={"eo:cloud_cover": {"lt": 20}},
            sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        )

        items = list(search.items())
        if not items:
            print("  No suitable imagery found in area", file=sys.stderr)
            return None

        item = items[0]
        cloud_cover = item.properties.get('eo:cloud_cover', 'unknown')
        print(f"  Found scene: {item.id} (cloud cover: {cloud_cover}%)", file=sys.stderr)

        if 'true-color' in item.assets:
            href = item.assets['true-color'].href
        elif 'visual' in item.assets:
            href = item.assets['visual'].href
        else:
            print("  No RGB visual asset available", file=sys.stderr)
            return None

        output_path = os.path.join(work_dir, 'imagery.tif')

        print("  Downloading imagery...", file=sys.stderr)

        with rasterio.open(href) as src:
            window = _window_in_src_crs(src, west, south, east, north)
            if window is None:
                print("  Could not locate target area within imagery tile", file=sys.stderr)
                return None

            data = src.read(window=window)
            src_transform = src.window_transform(window)

            if src.crs and src.crs.to_string() != 'EPSG:4326':
                dst_crs = 'EPSG:4326'

                win_left, win_bottom, win_right, win_top = array_bounds(
                    window.height, window.width, src_transform,
                )

                dst_transform, dst_width, dst_height = calculate_default_transform(
                    src.crs, dst_crs,
                    window.width, window.height,
                    win_left, win_bottom, win_right, win_top,
                )

                if dst_width * dst_height > MAX_OUTPUT_PIXELS:
                    scale = (MAX_OUTPUT_PIXELS / (dst_width * dst_height)) ** 0.5
                    dst_width = max(1, int(dst_width * scale))
                    dst_height = max(1, int(dst_height * scale))
                    dst_transform, dst_width, dst_height = calculate_default_transform(
                        src.crs, dst_crs,
                        window.width, window.height,
                        win_left, win_bottom, win_right, win_top,
                        dst_width=dst_width,
                        dst_height=dst_height,
                    )

                profile = src.profile.copy()
                profile.update(
                    driver='GTiff',
                    crs=dst_crs,
                    transform=dst_transform,
                    width=dst_width,
                    height=dst_height,
                    blockxsize=256,
                    blockysize=256,
                    tiled=True,
                )

                with rasterio.open(output_path, 'w', **profile) as dst:
                    for i in range(1, data.shape[0] + 1):
                        reproject(
                            source=data[i - 1],
                            destination=rasterio.band(dst, i),
                            src_transform=src_transform,
                            src_crs=src.crs,
                            dst_transform=dst_transform,
                            dst_crs=dst_crs,
                            resampling=Resampling.bilinear,
                        )
            else:
                profile = src.profile.copy()
                profile.update(
                    driver='GTiff',
                    height=window.height,
                    width=window.width,
                    transform=src_transform,
                    blockxsize=256,
                    blockysize=256,
                    tiled=True,
                )
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(data)

        print(f"  Imagery saved to: {output_path}", file=sys.stderr)
        return output_path

    except Exception as e:
        print(f"  Could not fetch satellite imagery: {e}", file=sys.stderr)
        return None
