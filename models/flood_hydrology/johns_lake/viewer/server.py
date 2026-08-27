"""
Flask viewer server — Johns Lake Digital Twin
Port 5050.  Run: python3 viewer/server.py

On first start (or when --reprocess is passed), runs the three preprocess
scripts to generate viewer/data/*.bin / *.json / *.png.
"""
import os, sys, subprocess, argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Files that must exist before we consider preprocessing done
REQUIRED = [
    "dem.bin",
    "geo_meta.json",
    "voxels.bin",
    "fwc_bed.bin",
    "naip_rgb.png",
    "ssurgo.png",
    "ssurgo_legend.json",
    "lake_mask.png",
    "hydrography.png",
    "floodplain.png",
    "roads_buildings.png",
    "simulation_index.json",
]

PREPROCESS_SCRIPTS = [
    os.path.join(BASE_DIR, "preprocess", "export_dem.py"),
    os.path.join(BASE_DIR, "preprocess", "export_voxels.py"),
    os.path.join(BASE_DIR, "preprocess", "export_overlays.py"),
    os.path.join(BASE_DIR, "preprocess", "export_simulation.py"),
]


def run_preprocessing():
    print("=" * 60)
    print("Preprocessing viewer data …")
    os.makedirs(DATA_DIR, exist_ok=True)
    for script in PREPROCESS_SCRIPTS:
        print(f"\n→ {os.path.basename(script)}")
        result = subprocess.run(
            [sys.executable, script],
            cwd=os.path.dirname(BASE_DIR),
        )
        if result.returncode != 0:
            print(f"  ERROR: {script} exited with code {result.returncode}")
            sys.exit(1)
    print("\nPreprocessing complete.\n" + "=" * 60)


def data_ready():
    return all(os.path.exists(os.path.join(DATA_DIR, f)) for f in REQUIRED)


from flask import Flask, send_from_directory, render_template

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def index():
    return render_template("index.html")


# JS shared with the sibling site, served from ../floodtwin/viewer_js so there is one
# canonical copy rather than a duplicate per viewer.
SHARED_JS_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "floodtwin", "viewer_js")


@app.route("/shared/<path:filename>")
def serve_shared(filename):
    return send_from_directory(SHARED_JS_DIR, filename)


@app.route("/data/<path:filename>")
def serve_data(filename):
    return send_from_directory(DATA_DIR, filename)


@app.route("/api/scenarios")
def api_scenarios():
    """Return simulation_index.json — list of available simulation scenarios."""
    return send_from_directory(DATA_DIR, "simulation_index.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reprocess", action="store_true", help="Re-run preprocessing even if data exists")
    parser.add_argument("--port", type=int, default=5050)
    # Binds to localhost by default: this dev server has no authentication, so exposing
    # it on all interfaces would publish the AOI data to the whole local network.
    # Pass --host 0.0.0.0 deliberately if you need access from another device.
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    if args.reprocess or not data_ready():
        run_preprocessing()

    print(f"\nStarting viewer at http://localhost:{args.port}/\n")
    app.run(host=args.host, port=args.port, debug=False)
