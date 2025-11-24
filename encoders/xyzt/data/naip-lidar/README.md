# NAIP-3DEP Production System

Production-ready system for downloading and training Earth4D on USGS 3DEP LiDAR + USDA NAIP aerial imagery.

**Dataset:** 22.8 million paired chips (256×256) across entire US
**Source:** [Nature Scientific Data (2025)](https://doi.org/10.1038/s41597-025-04655-z)
**Objective:** Input (lat, lon, elevation, time) → Output (R, G, B)

---

## Quick Start

### 1. Download Data for a Location

```bash
# Download 100 chips near Stanford (auto-downloads metadata on first run)
python naip_download.py --location Stanford_CA --max-chips 100 --parse --output data/stanford

# Or use custom lat/lon
python naip_download.py --lat 37.43 --lon -122.17 --radius 5 --max-chips 100 --parse --output data/custom
```

**First-time setup:** The script will automatically download the 3.2 GB metadata index (`files.csv`) which catalogs all 22.8M chips. This is cached at `~/.cache/naip/files.csv` and only happens once.

**Output:**
- `data/stanford/parsed_xyztrgb.pt` - Parsed data with chip metadata
- `data/stanford/chip_metadata.json` - Chip locations and properties

### 2. Train Earth4D Model

```bash
# Baseline model
python naip_train.py --data data/stanford/parsed_xyztrgb.pt --output runs/stanford_baseline --epochs 25

# With learned probing
python naip_train.py --data data/stanford/parsed_xyztrgb.pt --learned-probing --probing-range 32 --output runs/stanford_learned --epochs 25
```

**Output:**
- `runs/stanford_baseline/best_model.pt` - Best checkpoint
- `runs/stanford_baseline/history.json` - Training curves

---

## System Architecture

### Core Scripts

| Script | Purpose |
|--------|---------|
| `naip_download.py` | Download + parse NAIP data by location |
| `naip_train.py` | Train Earth4D with chip-based splits |
| `naip_utils.py` | Lat/lon ↔ UTM conversion utilities |

### Data Flow

```
User Input: (lat, lon) + radius
           ↓
   naip_download.py
           ↓
   [Auto-download files.csv on first use]
           ↓
   Find chips within radius
           ↓
   Download tar files
           ↓
   Extract & parse chips
           ↓
   parsed_xyztrgb.pt
   (with chip metadata)
           ↓
   naip_train.py
           ↓
   Chip-based train/test split
   (prevents spatial leakage)
           ↓
   Trained model
```

---

## Features

### Automatic Metadata Caching
- First run: Downloads 3.2 GB `files.csv` (~30-40 min)
- Subsequent runs: Uses cached version instantly
- Cache location: `~/.cache/naip/files.csv` (customizable with `--cache-dir`)

### Chip-Based Train/Test Split
- **Problem:** Random point splits leak spatial information across train/test
- **Solution:** Split by entire chips (256×256 grids)
- Each chip is 65,536 points - all go to train OR test, never split

### Geographic Targeting
- Specify any (latitude, longitude) anywhere in US
- Automatic UTM zone detection
- Find chips within radius (default 10 km)

---

## Target Locations

Pre-defined locations for easy testing:

```bash
# ASU Campus, Tempe AZ
python naip_download.py --location ASU_Tempe --max-chips 100 --parse --output data/asu

# Boulder, CO (Flatirons)
python naip_download.py --location Boulder_CO --max-chips 100 --parse --output data/boulder

# NYC High Line
python naip_download.py --location NYC_HighLine --max-chips 100 --parse --output data/nyc

# Stanford University
python naip_download.py --location Stanford_CA --max-chips 100 --parse --output data/stanford

# Miami South Beach
python naip_download.py --location Miami_SouthBeach --max-chips 100 --parse --output data/miami
```

---

## Data Format

### Parsed Data (`parsed_xyztrgb.pt`)

```python
{
    'data': torch.Tensor,          # (N, 7): [lat, lon, elev, time, r, g, b]
    'chip_ids': List[str],         # Chip identifiers
    'chip_sizes': List[int],       # Points per chip (typically 65,536)
    'chip_metadata': List[dict],   # Full metadata per chip
    'columns': List[str],          # Column names
    'n_chips': int,                # Number of chips
    'n_points': int,               # Total points
}
```

### Chip Metadata

Each chip includes:
- `chip_id`: Unique identifier (format: `zone_x_y_date`)
- `utm_zone`: UTM zone number
- `x`, `y`: UTM easting/northing coordinates
- `chm_date`: LiDAR acquisition date (YYYY-MM-DD)
- `naip_date`: NAIP aerial imagery date (YYYY-MM-DD)
- `land_cover`: Dominant land cover type
- `ecoregion`: EPA Level III ecoregion
- `distance`: Distance from query point (meters)

---

## Advanced Usage

### Download Without Parsing

```bash
# Just download raw tar files
python naip_download.py --location Stanford_CA --max-chips 100 --output data/stanford
# Parse later manually or with custom logic
```

### Custom Cache Directory

```bash
# Store metadata elsewhere
python naip_download.py --location Boulder_CO --cache-dir /mnt/data/naip_cache --max-chips 100 --parse --output data/boulder
```

### Larger Regions

```bash
# Get 1000 chips within 20 km radius
python naip_download.py --lat 40.01 --lon -105.27 --radius 20 --max-chips 1000 --parse --output data/boulder_large
```

### Training Configuration

```bash
# Custom hyperparameters
python naip_train.py \
    --data data/stanford/parsed_xyztrgb.pt \
    --output runs/stanford_custom \
    --learned-probing \
    --probing-range 64 \
    --epochs 50 \
    --batch-size 16384 \
    --lr 5e-5 \
    --train-split 0.9
```

---

## Technical Details

### Coordinate Systems
- **Filenames:** WGS84 (EPSG:326XX)
- **GeoTIFFs:** NAD83 (EPSG:269XX)
- **Script handles conversion automatically**

### Data Scaling
- **CHM (elevation):** Stored as `int16` scaled by 100 → divide by 100 to get meters
- **NAIP (RGB):** Stored as `uint8` → divide by 255 to get [0, 1]
- **Script handles normalization automatically**

### Chip Structure
- **CHM:** 256×256 pixels, 1m resolution, single-band (elevation)
- **NAIP:** Variable resolution (0.3-1.0m GSD), 5 bands (RGBN + mask)
- **Resampling:** NAIP resampled to match CHM 256×256 grid
- **Result:** 65,536 (lat, lon, elev, time, r, g, b) points per chip

### Train/Test Split Strategy
```python
# Pseudocode for chip-based split
chips = [chip1, chip2, ..., chipN]
random.shuffle(chips)
train_chips = chips[:80%]
test_chips = chips[80%:]

# All points from train_chips → train set
# All points from test_chips → test set
# Zero spatial leakage between sets
```

---

## Requirements

```bash
pip install torch numpy pandas rasterio tqdm
```

---

## File Organization

```
encoders/xyzt/
├── earth4d.py              # Earth4D encoder
└── data/
    └── naip-lidar/
        ├── naip_download.py       # Main download script
        ├── naip_train.py          # Training script
        ├── naip_utils.py          # Utility functions
        ├── NAIP_README.md         # This file
        ├── naip-3dep-chm.md       # Extracted Nature paper
        ├── archive/               # Old exploration scripts
        └── {location}/
            ├── parsed_xyztrgb.pt       # Parsed data
            ├── chip_metadata.json      # Chip info
            └── .cache/
                ├── files.csv            # Metadata index (auto-downloaded)
                ├── chm/{zone}/*.tar     # Downloaded CHM tars
                └── naip/{zone}/*.tar    # Downloaded NAIP tars
```

---

## Citation

If you use this data, please cite:

```bibtex
@article{allred2025canopy,
  title={Canopy height model and NAIP imagery pairs across CONUS},
  author={Allred, Brady W and McCord, Sarah E and Morford, Scott L},
  journal={Scientific Data},
  volume={12},
  number={322},
  year={2025},
  publisher={Nature Publishing Group},
  doi={10.1038/s41597-025-04655-z}
}
```

**License:** CC BY 4.0

---

## Troubleshooting

### "No chips found in UTM zone"
- Location may be outside US (CONUS only)
- Try increasing `--radius` parameter

### "files.csv download timeout"
- Run again - wget will resume from where it stopped
- Or manually download: `wget -c http://rangeland.ntsg.umt.edu/data/rap/chm-naip/files.csv`

### Training OOM (Out of Memory)
- Reduce `--batch-size` (try 4096 or 2048)
- Reduce number of chips: `--max-chips 50`

### Slow download
- Increase `--radius` to get more chips per tar file
- Server may be rate-limiting - downloads resume automatically

---

## Contact

For issues related to:
- **This code:** File an issue in the repository
- **Dataset:** Contact Brady Allred (allredbw@gmail.com)
