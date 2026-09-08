"""Flask server for the twin viewer.

The layer list comes from what is on disk, not a hardcoded manifest. Its predecessor refused to
start unless all 55 named files were present -- including a 1,011 MB point cloud and 30
PlanetScope textures the page renders perfectly well without.
"""

import json
from pathlib import Path
from typing import Dict, List

from flask import Flask, jsonify, render_template, send_from_directory

from sites import SiteConfig, get_site

BASE = Path(__file__).resolve().parent

LAYERS = [
    {"id": "naip", "label": "NAIP aerial imagery", "file": "naip.png", "kind": "drape"},
    {"id": "hand", "label": "Height above nearest drainage", "file": "hand.png", "kind": "drape"},
    {"id": "impervious", "label": "NLCD impervious surface", "file": "impervious.png",
     "kind": "drape"},
    {"id": "flood", "label": "Hurricane Ian flood depth", "file": "flood_ian.bin", "kind": "flood"},
]
"""Every optional layer, each declaring the one file it needs. Missing means absent from the
panel, never a refusal to start."""


def available(site: SiteConfig) -> List[Dict]:
    """Layers whose data is actually present for this site."""
    data = BASE / "data" / site.name
    return [dict(layer, available=(data / layer["file"]).exists()) for layer in LAYERS]


def create_app(site: SiteConfig) -> Flask:
    """Build the app for one site."""
    data_dir = BASE / "data" / site.name
    # dem.bin is guarded alongside meta.json because it is the one file the page cannot render
    # without: terrain.js fetches it unconditionally, and a 404 there becomes a Float32Array
    # over an HTML error body -- garbage terrain rather than a missing layer.
    for name in ("meta.json", "dem.bin"):
        assert (data_dir / name).exists(), (
            f"{site.name} viewer payload is missing {name}. Rebuild it with:\n"
            f"    python3 cli.py export --site {site.name}")

    app = Flask(__name__, static_folder=str(BASE / "static"),
                template_folder=str(BASE / "templates"))

    @app.route("/")
    def index() -> str:
        """The single page."""
        return render_template("index.html", site=site.name, label=site.label)

    @app.route("/api/layers")
    def layers() -> object:
        """Scene metadata and which optional layers have data on disk."""
        return jsonify({"site": site.name,
                        "meta": json.loads((data_dir / "meta.json").read_text()),
                        "layers": available(site)})

    @app.route("/data/<path:name>")
    def data(name: str) -> object:
        """Serve the committed viewer payload."""
        return send_from_directory(data_dir, name)

    return app


def serve(site: SiteConfig, port: int = 5051) -> None:
    """Run the viewer on localhost. Bound to 127.0.0.1 deliberately: there is no auth here."""
    app = create_app(site)
    print(f"  {site.label}\n  http://127.0.0.1:{port}")
    for layer in available(site):
        print(f"    [{'x' if layer['available'] else ' '}] {layer['label']}")
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    import sys
    serve(get_site(sys.argv[1] if len(sys.argv) > 1 else "site3"))
