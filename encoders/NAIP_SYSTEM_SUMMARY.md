# NAIP-3DEP System Location

All NAIP-3DEP files have been organized into:

```
encoders/xyzt/data/naip-lidar/
```

## Contents

### Production Scripts
- `naip_download.py` - Download and parse NAIP data by (lat, lon)
- `naip_train.py` - Train Earth4D with chip-based splits  
- `naip_utils.py` - Geographic utility functions

### Documentation
- `NAIP_README.md` - Complete usage guide
- `naip-3dep-chm.md` - Nature Scientific Data paper (extracted)

### Data
- `naip_files_metadata.csv` - Partial metadata download (38 MB)
- `archive/` - Old exploration scripts

## Quick Start

```bash
cd encoders/xyzt/data/naip-lidar

# Download data
python naip_download.py --location Stanford_CA --max-chips 100 --parse --output stanford

# Train model  
python naip_train.py --data stanford/parsed_xyztrgb.pt --output runs/stanford
```

See `NAIP_README.md` for full documentation.
