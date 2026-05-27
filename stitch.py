#!/usr/bin/env python3
"""Stitch a 2D grid of tile images into one image.

Usage:
    # tiles named tile_0_0.png, tile_0_1.png, ... tile_4_3.png
    python stitch.py 'tile_*.png' -o mosaic.png

    # or with explicit row-major list
    python stitch.py row0/*.png row1/*.png row2/*.png -o mosaic.png
"""

import argparse
import glob
import sys
from PIL import Image
import re


def parse_args():
    p = argparse.ArgumentParser(description='Stitch tiles into a mosaic')
    p.add_argument('patterns', nargs='+',
                   help='Glob patterns, one per row, or a single pattern')
    p.add_argument('-o', default='mosaic.png', help='Output file')
    return p.parse_args()


def numeric_sort(paths):
    return sorted(paths, key=lambda p: [int(n) if n.isdigit() else n
                                        for n in re.split(r'(\d+)', p)])


def main():
    args = parse_args()

    if len(args.patterns) == 1:
        files = numeric_sort(glob.glob(args.patterns[0]))
        if not files:
            print('No files matched.', file=sys.stderr)
            sys.exit(1)
        # Auto-detect grid dimensions from filenames
    else:
        files = []
        for pat in args.patterns:
            row = numeric_sort(glob.glob(pat))
            if not row:
                print(f'No files matched: {pat}', file=sys.stderr)
                sys.exit(1)
            files.append(row)

    if isinstance(files[0], list):
        rows_raw = files
    else:
        print(f'Found {len(files)} tiles', file=sys.stderr)
        tile_map = {}
        cols_dedup = {}
        for f in files:
            nums = [int(n) for n in re.findall(r'(\d+)', f)]
            if len(nums) >= 2:
                r, c = nums[0], nums[1]
                tile_map[(r, c)] = f
                cols_dedup.setdefault(r, set()).add(c)
        if tile_map:
            rows = sorted(set(r for r, _ in tile_map))
            cols_per_row = {r: len(cols_dedup[r]) for r in rows}
            ref_cols = cols_per_row[rows[0]]
            assert all(c == ref_cols for c in cols_per_row.values()), \
                'Rows have different column counts'
            cols = sorted(set(c for _, c in tile_map))
            rows_raw = [[tile_map[(r, c)] for c in cols] for r in rows]
            print(f'Auto-grid: {len(rows)} rows x {len(cols)} cols', file=sys.stderr)
        else:
            n = int(len(files) ** 0.5)
            while len(files) % n != 0:
                n -= 1
            rows_raw = [files[i:i + n] for i in range(0, len(files), n)]
            print(f'Auto-grid: {len(rows_raw)} rows x {n} cols', file=sys.stderr)

    tiles = []
    for r, row in enumerate(rows_raw):
        row_imgs = []
        for path in row:
            im = Image.open(path).convert('RGB')
            row_imgs.append(im)
        widths = [im.width for im in row_imgs]
        if len(set(widths)) > 1:
            print(f'Row {r}: varying widths {widths}', file=sys.stderr)
        total_w = sum(widths)
        max_h = max(im.height for im in row_imgs)
        row_canvas = Image.new('RGB', (total_w, max_h))
        x = 0
        for im in row_imgs:
            row_canvas.paste(im, (x, 0))
            x += im.width
        tiles.append(row_canvas)

    heights = [im.height for im in tiles]
    if len(set(heights)) > 1:
        print(f'Row heights differ: {heights}', file=sys.stderr)
    total_h = sum(heights)
    max_w = max(im.width for im in tiles)
    result = Image.new('RGB', (max_w, total_h))
    y = 0
    for im in tiles:
        result.paste(im, (0, y))
        y += im.height

    result.save(args.o)
    print(f'Saved {result.width}x{result.height} to {args.o}', file=sys.stderr)


if __name__ == '__main__':
    main()
