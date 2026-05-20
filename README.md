# LiDAR Metrics Pipeline

**Pipeline designed by Harshana Wedagedara**

A robust, production-ready Python pipeline for extracting 72 vegetation and terrain metrics from airborne LiDAR point clouds. Optimized for Alaska boreal forest analysis with built-in QA/QC validation and metadata generation.

---

## Features

- **Single-step meter output** — Processes LAS files in US Survey Feet, outputs GeoTIFF rasters in meters with proper CRS
- **72 comprehensive metrics** — Height distributions, canopy structure, terrain, and surface complexity
- **Noise filtering** — Classification-based (7, 18) + AGL threshold (60m) for robust outlier removal
- **Grid-snapped extents** — Ensures perfect alignment between rasters and point cloud footprints
- **Parallel processing** — Leverages multiprocessing for efficient computation (80 cores default)
- **Embedded QA/QC** — Pre- and post-processing validation with configurable tolerances
- **Rich metadata** — JSON metadata files with processing parameters, statistics, and QA results

---

## Metrics Produced (72 Total)

| Category | Count | Metrics |
|----------|-------|---------|
| **Height bins** | 12 | `hb_0_0.5`, `hb_0.5_1`, `hb_1_2`, ... `hb_more30` |
| **Cumulative density** | 9 | `dens_0_1`, `dens_0_2`, ... `dens_0_30` |
| **Percentiles** | 20 | `p5`, `p10`, `p15`, ... `p95`, `p99` |
| **Distribution stats** | 5 | `lida_mn`, `lida_med`, `lida_stdv`, `lida_skw`, `lida_kurt` |
| **Central tendency** | 2 | `q_mean` (quadratic), `c_mean` (cubic) |
| **Canopy height** | 1 | `chm` (canopy height model) |
| **Canopy/Understory** | 6 | `cnpy_min/max/mean`, `und_min/max/mean` |
| **Variability** | 4 | `cv_canopy/understory`, `var_canopy/understory` |
| **Vertical structure** | 1 | `crr` (canopy relief ratio) |
| **Cover** | 4 | `first_ccov`, `all_ccov`, `first_ucov`, `all_ucov` |
| **Surface complexity** | 2 | `rugosity`, `roughness` |
| **Terrain** | 2 | `slope` (degrees), `DEM` |

All height-based metrics output in **meters**. Proportion metrics (height bins, density, cover) are unitless (0-1).

---

## Quick Start

### Prerequisites

```bash
pip install numpy scipy rasterio laspy --break-system-packages
```

**Python 3.x** required. Tested on Python 3.8+.

### Basic Usage

**Single LAS file:**

```bash
python lidar_metrics_parallel.py
```

Edit configuration in script:
```python
INPUT_LAS = r"path/to/your_file.las"
OUTPUT_FOLDER = r"path/to/output"
```

**Batch processing:**

```bash
python batch_process_lidar.py
```

Configure paths:
```python
LAS_FOLDER = r"/path/to/las_files/"
OUTPUT_FOLDER = r"/path/to/output/"
```

**Optional QA validation:**

```bash
python validate_lidar_outputs.py
```

---

## Output Structure

```
output_folder/
├── batch_metadata.json              # Batch summary
├── batch_processing_log.txt         # Processing log
├── qa_qc_report.json                # QA validation results
├── qa_qc_summary.txt                # Human-readable QA report
└── FB17_3666/                       # Per-file outputs
    ├── FB17_3666_metadata.json     # File-specific metadata
    ├── FB17_3666_DEM.tif           # Digital elevation model
    ├── FB17_3666_chm.tif           # Canopy height model
    ├── FB17_3666_slope.tif         # Terrain slope (degrees)
    ├── FB17_3666_q_mean.tif        # Quadratic mean height
    ├── FB17_3666_c_mean.tif        # Cubic mean height
    └── ... (67 more metric rasters)
```

### Output Specifications

- **Format:** GeoTIFF (LZW compressed)
- **Resolution:** 10m cell size
- **CRS:** Meter-based (e.g., EPSG:6396 for NAD83(2011) Alaska Zone 3)
- **NoData:** -9999
- **Data type:** Float32

---

## Configuration

### Key Parameters

Edit in `lidar_metrics_parallel.py`:

```python
CELL_SIZE = 10                # Metric grid resolution (meters)
DEM_RESOLUTION = 1            # DEM resolution (meters)
CANOPY_THRESHOLD = 1.37       # Canopy/understory threshold (meters)
MAX_AGL_THRESHOLD = 60.0      # Maximum AGL for noise filtering (meters)
MIN_AGL_THRESHOLD = 0.0       # Minimum AGL (meters)
NUM_PROCESSES = 80            # Parallel processing cores

GROUND_CLASS = 2              # LAS classification for ground points
NOISE_CLASSIFICATIONS = [7, 18]  # Noise classes to remove
```

### QA/QC Tolerances

```python
QA_MAX_CHM = 50.0                 # Warning if CHM exceeds (meters)
QA_MAX_NODATA_PCT = 20.0          # Warning if NoData exceeds (%)
QA_MIN_GROUND_POINTS = 1000       # Minimum ground points required
QA_MIN_POINT_DENSITY = 1.0        # Minimum points per m²
```

---

## Technical Details

### Pipeline Workflow

1. **Pre-processing QA** — Validates LAS file (CRS, ground points, classification)
2. **Coordinate conversion** — US Survey Feet → Meters (0.3048006096...)
3. **Noise filtering** — Remove classifications 7, 18 + AGL > 60m
4. **DEM creation** — 1m resolution from ground points (class 2)
5. **Height normalization** — AGL = Z - DEM
6. **Extent processing** — Use filtered points + grid snapping
7. **Metric calculation** — Parallel processing per 10m cell
8. **Post-processing QA** — Validate outputs (CRS, ranges, completeness)
9. **Metadata generation** — JSON with parameters, stats, QA results

### CRS Detection

The pipeline automatically detects the meter-equivalent CRS using:
1. **Name matching** — Parses WKT for Alaska zone identifiers
2. **Parameter extraction** — Builds custom CRS from projection parameters if needed
3. **Verification** — Validates CRS before writing outputs

Supported: NAD83(2011) and NAD83 Alaska zones 2-10 (EPSG:6395-6403, 26932-26940).

### Extent Alignment Fix

The pipeline ensures raster extents perfectly match the LiDAR footprint:
- Uses **filtered point extents** (not LAS header)
- **Grid snapping**: `floor(min/CELL_SIZE)*CELL_SIZE`, `ceil(max/CELL_SIZE)*CELL_SIZE`
- **Floor indexing**: `np.floor((x-xmin)/CELL_SIZE)` for precise pixel assignment

## Requirements

### Python Libraries

- **numpy** — Array operations and numeric computation
- **scipy** — Interpolation (griddata), statistics, image filters (sobel)
- **rasterio** — GeoTIFF I/O and CRS handling
- **laspy** — LAS file reading
- **multiprocessing** — Parallel execution (standard library)

### Input Data Requirements

- **Format:** LAS 1.2+ (not LAZ — decompress first)
- **CRS:** NAD83 or NAD83(2011) Alaska zones
- **Units:** US Survey Feet (horizontal and vertical)
- **Classification:** Required field with ground points (class 2)
- **Returns:** Return number and number of returns fields

---

##  QA/QC

### Two-Level Validation

**1. Embedded QA (runs automatically)**
- Pre-processing: CRS, classification, ground points, point density
- Post-processing: Completeness, CRS, CHM range, NoData %, extent match
- Per-file metadata with QA results

**2. Standalone validation (optional)**
```bash
python validate_lidar_outputs.py
```
- Validates all 72 metrics per file
- Checks value ranges per metric type
- Generates comprehensive JSON + text reports

### Validation Checks

| Check | Threshold | Action |
|-------|-----------|--------|
| CHM max | 50m | Warning |
| NoData % | 20% | Warning |
| Ground points | 1,000 | Error if fewer |
| Point density | 1 pt/m² | Warning |
| Extent mismatch | 15m | Warning |

---

## Metadata

Each processed LAS file generates a JSON metadata file containing:

```json
{
  "pipeline": {
    "name": "LiDAR Metrics Pipeline",
    "version": "3.0",
    "designed_by": "Harshana Wedagedara"
  },
  "processing": {
    "date": "2026-05-11 14:32:15",
    "software": { "python_version": "3.12.0", ... }
  },
  "input": {
    "file": "FB17_3666.las",
    "point_count": 16115208,
    "source_epsg": null
  },
  "parameters": {
    "cell_size_meters": 10,
    "dem_resolution_meters": 1,
    "canopy_threshold_meters": 1.37,
    "max_agl_meters": 60
  },
  "output": {
    "target_epsg": 6396,
    "grid_rows": 92,
    "grid_cols": 92,
    "total_rasters": 73
  },
  "qa_qc": { ... }
}
```

---


## Citation

If you use this pipeline in your research, please cite:

```
Wedagedara, H. (2026). LiDAR Metrics Pipeline v1.0 
Automated extraction of vegetation and terrain metrics from airborne LiDAR.
GitHub: https://github.com/Harsha17uconn/LiDAR_Metrics_Pipeline
```


## Author

**Harshana Wedagedara**

Pipeline designed for Alaska boreal forest LiDAR analysis.

---

## Contact

For questions, issues, or contributions:
- Open an issue on GitHub
- Email: harshana.wedagedara@uconn.edu
