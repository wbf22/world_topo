import argparse
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Generate elevation maps from satellite data.',
    )

    parser.add_argument(
        '--lat',
        type=float,
        required=True,
        help='Center latitude of the area',
    )
    parser.add_argument(
        '--lon',
        type=float,
        required=True,
        help='Center longitude of the area',
    )
    parser.add_argument(
        '--width',
        type=float,
        default=0.15,
        help='Width of the area in degrees (default: 0.15)',
    )
    parser.add_argument(
        '--height',
        type=float,
        default=0.15,
        help='Height of the area in degrees (default: 0.15)',
    )
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Output image path (default: auto-named from coordinates)',
    )
    parser.add_argument(
        '--contour-interval',
        type=float,
        default=100,
        help='Contour interval in meters (default: 100)',
    )
    parser.add_argument(
        '--no-imagery',
        action='store_true',
        help='Skip satellite imagery underlay',
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Output just the map — no labels, axis, legend, or title',
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Save individual layers and print DEM stats for debugging',
    )

    return parser.parse_args(argv)


def main():
    args = parse_args()
    from world_topo.main import make_elevation_map

    output = args.output
    if output is None:
        lat_s = f"{args.lat:+.4f}".replace("+", "n").replace("-", "s")
        lon_s = f"{args.lon:+.4f}".replace("+", "e").replace("-", "w")
        output = f"elevation_{lat_s}_{lon_s}_{args.width}x{args.height}.png"

    make_elevation_map(
        lat=args.lat,
        lon=args.lon,
        width=args.width,
        height=args.height,
        output=output,
        contour_interval=args.contour_interval,
        no_imagery=args.no_imagery,
        clean=args.clean,
        debug=args.debug,
    )
