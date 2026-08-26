"""
Flask viewer server — CFX SR417 Corridor Digital Twin
Port 5051 (flood_hydrology's Johns Lake viewer uses 5050 — different port so
both can run side by side).  Run: python3 viewer/server.py

On first start (or when --reprocess is passed), runs the preprocess scripts
to generate viewer/data/*.bin / *.json / *.png.
"""
import os, sys, subprocess, argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Files that must exist before we consider preprocessing done. No voxels
# at this stage —.
REQUIRED = [
    "dem.bin",
    "geo_meta.json",
    "naip_rgb.png",
    "ssurgo.png",
    "ssurgo_legend.json",
    "hydrography.png",
    "floodplain.png",
    "boundary.png",
    "roads_buildings.png",
    "terrain_elevation.png",
    "terrain_slope.png",
    "terrain_hillshade.png",
    "terrain_tpi.png",
    "terrain_curvature.png",
    "terrain_tri.png",
    "hydro_hand.png",
    "hydro_flow_accum.png",
    "hydro_streams.png",
    "fema_hand_risk.png",
    "fema_sim_extent_overlay.png",
    "ian_flood_viewer.png",
    "simulation_ian_frames.bin",
    "simulation_ian_infiltration.bin",
    "simulation_ian_hydrograph.json",
    "simulation_index.json",
    "ps_max1_water.png",
    "ps_max1_water.png", "ps_max2_water.png", "ps_max3_water.png",
    "ps_avg1_water.png", "ps_avg2_water.png", "ps_avg3_water.png", "ps_avg4_water.png",
    "ps_min1_water.png", "ps_min2_water.png", "ps_min3_water.png",
    "ps_max1_rgb.png",   "ps_max2_rgb.png",   "ps_max3_rgb.png",
    "ps_avg1_rgb.png",   "ps_avg2_rgb.png",   "ps_avg3_rgb.png",   "ps_avg4_rgb.png",
    "ps_min1_rgb.png",   "ps_min2_rgb.png",   "ps_min3_rgb.png",
    "ps_max1_ndvi.png",  "ps_max2_ndvi.png",  "ps_max3_ndvi.png",
    "ps_avg1_ndvi.png",  "ps_avg2_ndvi.png",  "ps_avg3_ndvi.png",  "ps_avg4_ndvi.png",
    "ps_min1_ndvi.png",  "ps_min2_ndvi.png",  "ps_min3_ndvi.png",
    "bridge_mesh_town_loop_blvd.obj",
    "bridge_mesh_john_young_pkwy.obj",
    "lidar_pointcloud.bin",
]

PREPROCESS_SCRIPTS = [
    os.path.join(BASE_DIR, "preprocess", "export_dem.py"),
    os.path.join(BASE_DIR, "preprocess", "export_overlays.py"),
    os.path.join(BASE_DIR, "preprocess", "export_ian_simulation.py"),
    # Requires lidar/build_lidar_pointcloud.py + lidar/export_full_pointcloud.py to have been
    # run first (downloads the raw LAZ point cloud, builds the bridge-crossing meshes, and
    # exports the decimated full-AOI point cloud) — same one-time-prerequisite convention as
    # export_ian_simulation.py's flood_sim_ian.py dependency.
    os.path.join(BASE_DIR, "preprocess", "export_lidar.py"),
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
# Without this, Flask caches the compiled index.html in memory for the life of the process —
# template edits (unlike static JS/CSS, read fresh from disk each request) silently never take
# effect until the server is restarted, which is an easy-to-miss trap; worth keeping on for a
# long-running local dev viewer that gets iterated on.
app.config["TEMPLATES_AUTO_RELOAD"] = True


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/site3")
def site3():
    # Gee Creek gauge-matched validation site (37km from the main AOI) — a separate page/
    # scene, not a layer in the main one, since it has its own terrain/coordinate space.
    # Not gated on REQUIRED/data_ready() — this page's own data
    # (dem_site3.bin, geo_meta_site3.json, depth_frames_ian_site3.bin, etc.) is optional and
    # built by a separate preprocessing step (viewer/preprocess/export_dem_site3.py +
    # simulation/run_site3_ian.py --save-frames), not part of the main AOI's startup gate.
    return render_template("site3.html")


@app.route("/data/<path:filename>")
def serve_data(filename):
    return send_from_directory(DATA_DIR, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reprocess", action="store_true", help="Re-run preprocessing even if data exists")
    parser.add_argument("--port", type=int, default=5051)
    # Binds to localhost by default: this dev server has no authentication, so exposing
    # it on all interfaces would publish the AOI data to the whole local network.
    # Pass --host 0.0.0.0 deliberately if you need access from another device.
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    if args.reprocess or not data_ready():
        run_preprocessing()

    print(f"\nStarting viewer at http://localhost:{args.port}/\n")
    app.run(host=args.host, port=args.port, debug=False)
