#!/usr/bin/env python3
"""
DeepEarth Dashboard Server Launcher

Simple script to start the DeepEarth Dashboard with appropriate
configuration and startup information display.
"""

from deepearth_dashboard import app, CONFIG

def main():
    """Start the DeepEarth Dashboard server."""
    print("\n" + "="*80)
    print("🌍 DeepEarth Multimodal Geospatial Dashboard")
    print("="*80)
    print(f"Dataset: {CONFIG['dataset_name']}")
    print(f"Version: {CONFIG['dataset_version']}")
    print(f"\nStarting server on http://localhost:5000")
    print("="*80 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000)


if __name__ == '__main__':
    main()