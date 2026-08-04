"""
Export Ian simulation frames → viewer/data/

Two scenarios are exported:
  ian_noinfil — no Horton infiltration (Pe = rain; water stays on surface; DEFAULT)
  ian         — with per-cell SSURGO Horton infiltration (fc_eff 0.7-62.3 mm/hr by soil
                unit, AMC-III correction; spatially varying, less runoff overall)

Run flood_sim_ian.py twice before exporting:
  python3 simulation/flood_sim_ian.py --cell-size 5 --dt 20 --frame-interval 30 --save-frames --no-infiltration
  python3 simulation/flood_sim_ian.py --cell-size 5 --dt 20 --frame-interval 30 --save-frames

Then run this script:
  python3 viewer/preprocess/export_ian_simulation.py
"""
import os, json, shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # viewer/
PROJ_DIR = os.path.dirname(BASE_DIR)
SIM_OUT  = os.path.join(PROJ_DIR, "simulation", "outputs")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


def _copy(src, dst, required=True):
    if not os.path.exists(src):
        msg = f"  {'MISSING' if required else 'optional'}: {os.path.basename(src)}"
        if required:
            msg += " — run flood_sim_ian.py --save-frames"
        print(msg)
        return False
    shutil.copy2(src, dst)
    print(f"  {os.path.basename(dst)}  ({os.path.getsize(dst)//1024} KB)")
    return True


def export_scenario(tag, label):
    """tag = '' (with infiltration) or '_noinfil' (without)."""
    src_bin  = os.path.join(SIM_OUT, f"depth_frames_ian{tag}.bin")
    dst_bin  = os.path.join(DATA_DIR, f"simulation_ian{tag}_frames.bin")
    src_json = os.path.join(SIM_OUT, f"simulation_ian{tag}_hydrograph.json")
    dst_json = os.path.join(DATA_DIR, f"simulation_ian{tag}_hydrograph.json")

    print(f"\n  [{label}]")
    ok_bin  = _copy(src_bin,  dst_bin)
    ok_json = _copy(src_json, dst_json)
    if ok_json:
        with open(dst_json) as f:
            hj = json.load(f)
        n       = len(hj.get("times_min", []))
        peak_ha = hj.get("peak_flooded_ha", 0)
        print(f"    {n} frames  total={hj.get('total_rain_mm',0):.0f}mm  peak={peak_ha:.1f}ha")
    return ok_bin, ok_json


def main():
    print("Exporting Ian simulation files → viewer/data/ …")

    # No-infiltration scenario (default viewer scenario)
    ok_noinfil, _ = export_scenario("_noinfil", "No infiltration — water stays on surface")

    # With-infiltration scenario
    ok_infil, _   = export_scenario("", "With per-cell SSURGO Horton infiltration (fc_eff 0.7-62.3 mm/hr by soil unit)")

    # simulation_index.json: list both scenarios for the viewer toggle
    entries = []
    for tag, scenario_id, label, use_infil in [
        ("_noinfil", "ian_noinfil", "Hurricane Ian · no infiltration (surface runoff only)", False),
        ("",         "ian",         "Hurricane Ian · with soil infiltration (Horton/SSURGO)",  True),
    ]:
        summary_src = os.path.join(SIM_OUT, f"ian{tag}_sim_summary.json")
        if os.path.exists(summary_src):
            with open(summary_src) as f:
                entry = json.load(f)
        else:
            entry = {
                "id":              scenario_id,
                "label":           label,
                "use_infiltration": use_infil,
                "n_frames":        145,
                "frame_interval_min": 30,
            }
        entries.append(entry)

    index_path = os.path.join(DATA_DIR, "simulation_index.json")
    with open(index_path, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"\n  simulation_index.json  ({len(entries)} scenarios)")
    print("Done.")


if __name__ == "__main__":
    main()
