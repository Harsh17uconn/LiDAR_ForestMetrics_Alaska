"""
LiDAR Metrics Calculation Pipeline
====================================
Pipeline designed by Harshana Wedagedara

Creates 72 vegetation/terrain metrics from LiDAR point clouds.
All outputs in METERS with proper meter-based CRS.

Features:
- Noise filtering (classifications + AGL thresholds)
- DEM creation from ground points
- 72 metrics including q_mean, c_mean, slope
- Grid-snapped extents for perfect alignment
- QA/QC validation
- JSON metadata generation
"""
import os
import re
import sys
import json
import platform
from datetime import datetime
import numpy as np
import laspy
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
from scipy import stats
from scipy.interpolate import griddata
from scipy.ndimage import sobel
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# ==================== PIPELINE METADATA ====================
PIPELINE_NAME = "LiDAR Metrics Pipeline"
PIPELINE_VERSION = "3.0"
PIPELINE_AUTHOR = "Harshana Wedagedara"

# ==================== CONFIGURATION ====================
INPUT_LAS = r"E:\Biomass\norm_las\your_file.las"
OUTPUT_FOLDER = r"E:\Biomass\metrics_output"

# All values in METERS
CELL_SIZE = 10                # Cell size in meters
DEM_RESOLUTION = 1            # DEM resolution in meters
CANOPY_THRESHOLD = 1.37       # Canopy/understory threshold in meters
MAX_AGL_THRESHOLD = 60.0      # Max AGL in meters
MIN_AGL_THRESHOLD = 0.0       # Min AGL in meters
NUM_PROCESSES = 80

GROUND_CLASS = 2
DEM_AGGREGATION = 'min'
DEM_INTERPOLATION = 'linear'
SAVE_DEM = True

NOISE_CLASSIFICATIONS = [7, 18]

# ==================== QA/QC TOLERANCES ====================
QA_MAX_CHM = 50.0                 # Warning if CHM exceeds this (meters)
QA_MAX_NODATA_PCT = 20.0          # Warning if NoData exceeds this %
QA_MIN_GROUND_POINTS = 1000       # Minimum ground points required
QA_MIN_POINT_DENSITY = 1.0        # Minimum points per m²
QA_MAX_EXTENT_DIFF = 15.0         # Max extent difference (meters)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==================== VERIFIED EPSG CODES ====================
# Maps CRS NAME PATTERNS to verified EPSG codes (in meters)

CRS_NAME_TO_METER_EPSG = [
    # NAD83(2011) Alaska zones - VERIFIED on epsg.io
    # IMPORTANT: Check zone 10 BEFORE zone 1 (so 10 doesn't match "zone 1" first)
    ('alaska zone 10', {'2011': 6403, 'default': 26940}),
    ('alaska zone 2', {'2011': 6395, 'default': 26932}),
    ('alaska zone 3', {'2011': 6396, 'default': 26933}),  # ← Fairbanks
    ('alaska zone 4', {'2011': 6397, 'default': 26934}),
    ('alaska zone 5', {'2011': 6398, 'default': 26935}),
    ('alaska zone 6', {'2011': 6399, 'default': 26936}),
    ('alaska zone 7', {'2011': 6400, 'default': 26937}),
    ('alaska zone 8', {'2011': 6401, 'default': 26938}),
    ('alaska zone 9', {'2011': 6402, 'default': 26939}),
    ('alaska zone 1', {'2011': None, 'default': None}),  # Zone 1 is special
]


# ==================== CRS DETECTION ====================

def detect_crs_from_name(wkt):
    """Detect CRS by parsing the name in the WKT - most reliable method."""
    if not wkt:
        return None, None
    
    wkt_lower = wkt.lower()
    is_nad83_2011 = '2011' in wkt or 'nsrs2011' in wkt_lower
    
    for zone_pattern, epsg_dict in CRS_NAME_TO_METER_EPSG:
        if zone_pattern in wkt_lower:
            if is_nad83_2011:
                epsg = epsg_dict.get('2011')
            else:
                epsg = epsg_dict.get('default')
            
            zone_name_pretty = zone_pattern.title()
            if is_nad83_2011:
                zone_name_pretty = f"NAD83(2011) {zone_name_pretty}"
            else:
                zone_name_pretty = f"NAD83 {zone_name_pretty}"
            
            return zone_name_pretty, epsg
    
    return None, None


def extract_projection_parameters(crs):
    """Extract projection parameters from CRS WKT."""
    params = {
        'central_meridian': None,
        'lat_origin': None,
        'scale_factor': None,
        'false_easting': None,
        'false_northing': None,
        'projection': None,
        'is_feet': False,
        'is_us_feet': False,
        'is_nad83_2011': False,
        'raw_wkt': '',
    }
    
    if crs is None:
        return params
    
    try:
        wkt = crs.to_wkt()
        params['raw_wkt'] = wkt
        wkt_lower = wkt.lower()
        
        if 'transverse_mercator' in wkt_lower or 'transverse mercator' in wkt_lower:
            params['projection'] = 'transverse_mercator'
        elif 'lambert_conformal_conic' in wkt_lower:
            params['projection'] = 'lambert_conformal_conic'
        
        if '2011' in wkt or 'nsrs2011' in wkt_lower:
            params['is_nad83_2011'] = True
        
        if ('foot_us' in wkt_lower or 'us survey foot' in wkt_lower or 
            'ussurveyfoot' in wkt_lower or '"us survey foot"' in wkt_lower):
            params['is_feet'] = True
            params['is_us_feet'] = True
        elif 'foot' in wkt_lower:
            params['is_feet'] = True
        
        param_patterns = {
            'central_meridian': [
                r'PARAMETER\["[Cc]entral_[Mm]eridian"\s*,\s*([-\d.eE+]+)\]',
                r'PARAMETER\[\s*"central_meridian"\s*,\s*([-\d.eE+]+)',
                r'"central_meridian"[^,]*,\s*([-\d.eE+]+)',
                r'"Longitude of natural origin"\s*,\s*([-\d.eE+]+)',
            ],
            'lat_origin': [
                r'PARAMETER\["[Ll]atitude_[Oo]f_[Oo]rigin"\s*,\s*([-\d.eE+]+)\]',
                r'PARAMETER\[\s*"latitude_of_origin"\s*,\s*([-\d.eE+]+)',
                r'"latitude_of_origin"[^,]*,\s*([-\d.eE+]+)',
                r'"Latitude of natural origin"\s*,\s*([-\d.eE+]+)',
            ],
            'scale_factor': [
                r'PARAMETER\["[Ss]cale_[Ff]actor"\s*,\s*([-\d.eE+]+)\]',
                r'PARAMETER\[\s*"scale_factor"\s*,\s*([-\d.eE+]+)',
                r'"scale_factor"[^,]*,\s*([-\d.eE+]+)',
            ],
            'false_easting': [
                r'PARAMETER\["[Ff]alse_[Ee]asting"\s*,\s*([-\d.eE+]+)\]',
                r'PARAMETER\[\s*"false_easting"\s*,\s*([-\d.eE+]+)',
                r'"false_easting"[^,]*,\s*([-\d.eE+]+)',
            ],
            'false_northing': [
                r'PARAMETER\["[Ff]alse_[Nn]orthing"\s*,\s*([-\d.eE+]+)\]',
                r'PARAMETER\[\s*"false_northing"\s*,\s*([-\d.eE+]+)',
                r'"false_northing"[^,]*,\s*([-\d.eE+]+)',
            ],
        }
        
        for key, patterns in param_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, wkt, re.IGNORECASE | re.DOTALL)
                if match:
                    try:
                        params[key] = float(match.group(1))
                        break
                    except ValueError:
                        continue
        
    except Exception as e:
        print(f"  WARNING: Error parsing CRS: {e}")
    
    return params


def verify_crs(crs_obj):
    """Verify CRS is valid."""
    if crs_obj is None:
        return False
    try:
        wkt = crs_obj.to_wkt()
        return wkt is not None and len(wkt) >= 10
    except Exception:
        return False


US_FOOT_TO_METER = 0.3048006096012192
INTL_FOOT_TO_METER = 0.3048


def get_crs_info(las_crs):
    """Detect CRS units and find correct meter-equivalent CRS."""
    if las_crs is None:
        raise ValueError("LAS file has no CRS - cannot proceed")
    
    print(f"  Detecting CRS from LAS file...")
    
    params = extract_projection_parameters(las_crs)
    
    wkt_preview = params['raw_wkt'][:200] if params['raw_wkt'] else 'EMPTY'
    print(f"  WKT preview: {wkt_preview}")
    print()
    
    print(f"  Detected information:")
    print(f"    Projection: {params['projection']}")
    print(f"    NAD83(2011)?: {params['is_nad83_2011']}")
    print(f"    Units: {'US Survey Feet' if params['is_us_feet'] else ('Feet' if params['is_feet'] else 'Meters')}")
    
    if params['is_us_feet']:
        h_unit = 'us_feet'
        h_to_meter = US_FOOT_TO_METER
    elif params['is_feet']:
        h_unit = 'intl_feet'
        h_to_meter = INTL_FOOT_TO_METER
    else:
        h_unit = 'meters'
        h_to_meter = 1.0
    
    source_epsg = None
    try:
        source_epsg = las_crs.to_epsg()
    except:
        pass
    
    meter_crs = None
    target_epsg = None
    
    # Strategy 1: Match by CRS NAME
    print(f"\n  Strategy 1: Matching by CRS name...")
    zone_name, target_epsg = detect_crs_from_name(params['raw_wkt'])
    
    if zone_name and target_epsg:
        print(f"    ✓ Found by name: {zone_name}")
        print(f"    ✓ Target EPSG: {target_epsg}")
        try:
            meter_crs = CRS.from_epsg(target_epsg)
            if verify_crs(meter_crs):
                print(f"    ✓ CRS created successfully")
            else:
                meter_crs = None
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            meter_crs = None
    
    # Strategy 2: Already in meters
    if meter_crs is None and h_unit == 'meters':
        print(f"\n  Strategy 2: CRS already in meters, using original")
        meter_crs = las_crs
        target_epsg = source_epsg
    
    # Strategy 3: Build custom CRS
    if meter_crs is None:
        print(f"\n  Strategy 3: Building custom CRS from parameters...")
        meter_crs = build_meter_crs_from_params(params)
    
    if not verify_crs(meter_crs):
        raise ValueError(
            "Could not create a valid meter-based CRS!\n"
            f"  Source EPSG: {source_epsg}\n"
            f"  Please check the LAS file's CRS"
        )
    
    print(f"\n  >>> Final CRS: EPSG:{target_epsg if target_epsg else 'custom'} <<<")
    
    v_to_meter = h_to_meter
    return h_unit, h_to_meter, v_to_meter, meter_crs, source_epsg, target_epsg


def build_meter_crs_from_params(params):
    """Build a custom meter-based CRS from projection parameters."""
    if (params['central_meridian'] is None or 
        params['lat_origin'] is None or
        params['projection'] != 'transverse_mercator'):
        return None
    
    fe = params['false_easting'] if params['false_easting'] is not None else 0
    fn = params['false_northing'] if params['false_northing'] is not None else 0
    sf = params['scale_factor'] if params['scale_factor'] is not None else 0.9999
    
    if params['is_us_feet']:
        fe = fe * US_FOOT_TO_METER
        fn = fn * US_FOOT_TO_METER
    elif params['is_feet']:
        fe = fe * INTL_FOOT_TO_METER
        fn = fn * INTL_FOOT_TO_METER
    
    proj4 = (
        f"+proj=tmerc +lat_0={params['lat_origin']} +lon_0={params['central_meridian']} "
        f"+k={sf} +x_0={fe} +y_0={fn} +datum=NAD83 +units=m +no_defs"
    )
    
    try:
        return CRS.from_proj4(proj4)
    except Exception:
        return None


# ==================== METRIC DEFINITIONS (in METERS) ====================

HEIGHT_BINS = [
    ('hb_0_0.5', 0, 0.5),
    ('hb_0.5_1', 0.5, 1),
    ('hb_1_2', 1, 2),
    ('hb_2_3', 2, 3),
    ('hb_3_4', 3, 4),
    ('hb_4_5', 4, 5),
    ('hb_5_10', 5, 10),
    ('hb_10_15', 10, 15),
    ('hb_15_20', 15, 20),
    ('hb_20_25', 20, 25),
    ('hb_25_30', 25, 30),
    ('hb_more30', 30, 1000),
]

DENS_THRESHOLDS = [
    ('dens_0_1', 1), ('dens_0_2', 2), ('dens_0_3', 3), ('dens_0_5', 5),
    ('dens_0_10', 10), ('dens_0_15', 15), ('dens_0_20', 20),
    ('dens_0_25', 25), ('dens_0_30', 30),
]

PERCENTILES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]


# ==================== QA/QC FUNCTIONS ====================

def validate_las_file(las_path):
    """Pre-processing validation of LAS file."""
    qa_results = {'passed': True, 'warnings': [], 'errors': []}
    
    if not os.path.exists(las_path):
        qa_results['errors'].append(f"File not found: {las_path}")
        qa_results['passed'] = False
        return qa_results
    
    try:
        las = laspy.read(las_path)
        
        try:
            crs = las.header.parse_crs()
            if crs is None:
                qa_results['errors'].append("No CRS in LAS header")
                qa_results['passed'] = False
        except Exception as e:
            qa_results['errors'].append(f"Cannot parse CRS: {e}")
            qa_results['passed'] = False
        
        try:
            classification = np.array(las.classification)
            n_ground = np.sum(classification == GROUND_CLASS)
            if n_ground < QA_MIN_GROUND_POINTS:
                qa_results['warnings'].append(
                    f"Few ground points: {n_ground:,} (min: {QA_MIN_GROUND_POINTS:,})"
                )
        except Exception as e:
            qa_results['errors'].append(f"No classification field: {e}")
            qa_results['passed'] = False
        
        n_points = las.header.point_count
        if n_points < 1000:
            qa_results['warnings'].append(f"Very few points: {n_points:,}")
    
    except Exception as e:
        qa_results['errors'].append(f"Error reading LAS: {e}")
        qa_results['passed'] = False
    
    return qa_results


def validate_outputs(output_folder, base_name, expected_metrics):
    """Post-processing validation of output rasters."""
    qa_results = {'passed': True, 'warnings': [], 'errors': [], 'stats': {}}
    
    missing = []
    for metric in expected_metrics:
        filepath = os.path.join(output_folder, f"{base_name}_{metric}.tif")
        if not os.path.exists(filepath):
            missing.append(metric)
    
    if missing:
        qa_results['errors'].append(f"Missing rasters: {missing[:10]}")
        qa_results['passed'] = False
    
    # Validate CHM
    chm_path = os.path.join(output_folder, f"{base_name}_chm.tif")
    if os.path.exists(chm_path):
        try:
            with rasterio.open(chm_path) as src:
                data = src.read(1)
                nodata = src.nodata
                
                if src.crs is None:
                    qa_results['errors'].append("CHM has no CRS!")
                    qa_results['passed'] = False
                else:
                    qa_results['stats']['crs'] = str(src.crs)
                
                valid = data[data != nodata] if nodata is not None else data
                
                if len(valid) > 0:
                    qa_results['stats']['chm_max_m'] = float(np.max(valid))
                    qa_results['stats']['chm_mean_m'] = float(np.mean(valid))
                    
                    if np.max(valid) > QA_MAX_CHM:
                        qa_results['warnings'].append(
                            f"CHM max ({np.max(valid):.1f}m) exceeds threshold ({QA_MAX_CHM}m)"
                        )
                
                n_total = data.size
                n_nodata = np.sum(data == nodata) if nodata is not None else 0
                nodata_pct = 100 * n_nodata / n_total
                qa_results['stats']['nodata_pct'] = float(nodata_pct)
                
                if nodata_pct > QA_MAX_NODATA_PCT:
                    qa_results['warnings'].append(
                        f"High NoData: {nodata_pct:.1f}% (max: {QA_MAX_NODATA_PCT}%)"
                    )
        except Exception as e:
            qa_results['errors'].append(f"Error validating CHM: {e}")
            qa_results['passed'] = False
    
    return qa_results


def generate_metadata(las_path, output_folder, processing_info, qa_results):
    """Generate JSON metadata file."""
    base_name = os.path.splitext(os.path.basename(las_path))[0]
    
    metadata = {
        "pipeline": {
            "name": PIPELINE_NAME,
            "version": PIPELINE_VERSION,
            "designed_by": PIPELINE_AUTHOR,
            "description": "LiDAR-based vegetation and terrain metrics pipeline"
        },
        "processing": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "software": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "numpy_version": np.__version__,
                "rasterio_version": rasterio.__version__,
                "laspy_version": laspy.__version__,
            }
        },
        "input": {
            "file": os.path.basename(las_path),
            "path": las_path,
            "size_mb": round(os.path.getsize(las_path) / (1024*1024), 2) if os.path.exists(las_path) else 0,
            **processing_info.get('input', {})
        },
        "parameters": {
            "cell_size_meters": CELL_SIZE,
            "dem_resolution_meters": DEM_RESOLUTION,
            "canopy_threshold_meters": CANOPY_THRESHOLD,
            "min_agl_meters": MIN_AGL_THRESHOLD,
            "max_agl_meters": MAX_AGL_THRESHOLD,
            "noise_classifications_removed": NOISE_CLASSIFICATIONS,
            "ground_classification": GROUND_CLASS,
            "dem_aggregation": DEM_AGGREGATION,
            "dem_interpolation": DEM_INTERPOLATION,
        },
        "output": {
            "folder": output_folder,
            "crs_units": "Meters",
            **processing_info.get('output', {})
        },
        "qa_qc": {
            "pre_processing": processing_info.get('pre_qa', {}),
            "post_processing": qa_results,
            "tolerances": {
                "max_chm_meters": QA_MAX_CHM,
                "max_nodata_percent": QA_MAX_NODATA_PCT,
                "min_ground_points": QA_MIN_GROUND_POINTS,
                "min_point_density_per_m2": QA_MIN_POINT_DENSITY,
                "max_extent_diff_meters": QA_MAX_EXTENT_DIFF,
            }
        },
        "metrics_categories": {
            "height_bins": [name for name, _, _ in HEIGHT_BINS],
            "cumulative_density": [name for name, _ in DENS_THRESHOLDS],
            "percentiles": [f"p{p}" for p in PERCENTILES],
            "distribution_stats": ["lida_mn", "lida_med", "lida_stdv", "lida_skw", "lida_kurt"],
            "central_tendency": ["q_mean", "c_mean"],
            "canopy_height": ["chm"],
            "canopy_understory": ["cnpy_min", "cnpy_max", "cnpy_mean",
                                  "und_min", "und_max", "und_mean"],
            "variability": ["cv_canopy", "cv_understory", "var_canopy", "var_understory"],
            "vertical_structure": ["crr"],
            "cover": ["first_ccov", "all_ccov", "first_ucov", "all_ucov"],
            "surface_complexity": ["rugosity", "roughness"],
            "terrain": ["slope", "DEM"],
        }
    }
    
    metadata_path = os.path.join(output_folder, f"{base_name}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_path


# ==================== DEM CREATION ====================

def create_dem_from_lidar(x_m, y_m, z_m, classification, dem_resolution=1.0):
    """Create DEM from ground points. All inputs in METERS."""
    print(f"\nCreating DEM from LiDAR ground points...")
    
    ground_mask = classification == GROUND_CLASS
    n_ground = np.sum(ground_mask)
    
    print(f"  Total points: {len(x_m):,}")
    print(f"  Ground points (class {GROUND_CLASS}): {n_ground:,}")
    
    if n_ground == 0:
        raise ValueError("No ground points found!")
    
    x_ground = x_m[ground_mask]
    y_ground = y_m[ground_mask]
    z_ground = z_m[ground_mask]
    
    xmin = x_m.min()
    xmax = x_m.max()
    ymin = y_m.min()
    ymax = y_m.max()
    
    buffer = dem_resolution
    xmin -= buffer
    ymax += buffer
    xmax += buffer
    ymin -= buffer
    
    ncols = int(np.ceil((xmax - xmin) / dem_resolution))
    nrows = int(np.ceil((ymax - ymin) / dem_resolution))
    
    print(f"  DEM grid: {nrows} x {ncols} at {dem_resolution}m")
    
    cols = np.floor((x_ground - xmin) / dem_resolution).astype(int)
    rows = np.floor((ymax - y_ground) / dem_resolution).astype(int)
    
    inside = (rows >= 0) & (rows < nrows) & (cols >= 0) & (cols < ncols)
    rows = rows[inside]
    cols = cols[inside]
    z_ground = z_ground[inside]
    
    print(f"  Aggregating using '{DEM_AGGREGATION}'...")
    dem_grid = np.full((nrows, ncols), np.nan, dtype=np.float32)
    
    if DEM_AGGREGATION == 'min':
        cell_min = np.full((nrows, ncols), np.inf, dtype=np.float32)
        np.minimum.at(cell_min, (rows, cols), z_ground)
        valid = cell_min != np.inf
        dem_grid[valid] = cell_min[valid]
    elif DEM_AGGREGATION == 'mean':
        cell_sum = np.zeros((nrows, ncols), dtype=np.float64)
        cell_count = np.zeros((nrows, ncols), dtype=np.int32)
        np.add.at(cell_sum, (rows, cols), z_ground)
        np.add.at(cell_count, (rows, cols), 1)
        valid = cell_count > 0
        dem_grid[valid] = cell_sum[valid] / cell_count[valid]
    
    n_filled = np.sum(~np.isnan(dem_grid))
    n_empty = np.sum(np.isnan(dem_grid))
    print(f"  Cells with data: {n_filled:,} ({100*n_filled/(nrows*ncols):.1f}%)")
    
    if n_empty > 0:
        print(f"  Interpolating {n_empty:,} empty cells using '{DEM_INTERPOLATION}'...")
        
        filled_mask = ~np.isnan(dem_grid)
        empty_mask = np.isnan(dem_grid)
        
        rows_filled, cols_filled = np.where(filled_mask)
        rows_empty, cols_empty = np.where(empty_mask)
        
        if len(rows_filled) >= 4:
            points_known = np.column_stack([rows_filled, cols_filled])
            values_known = dem_grid[filled_mask]
            points_unknown = np.column_stack([rows_empty, cols_empty])
            
            try:
                interpolated = griddata(points_known, values_known, points_unknown,
                                       method=DEM_INTERPOLATION)
                still_nan = np.isnan(interpolated)
                if np.any(still_nan):
                    interpolated_nn = griddata(points_known, values_known, 
                                              points_unknown[still_nan], method='nearest')
                    interpolated[still_nan] = interpolated_nn
                dem_grid[empty_mask] = interpolated
            except Exception:
                interpolated = griddata(points_known, values_known, points_unknown,
                                       method='nearest')
                dem_grid[empty_mask] = interpolated
    
    dem_extent = (xmin, ymin, xmax, ymax)
    print(f"  DEM elevation range: {np.nanmin(dem_grid):.2f} to {np.nanmax(dem_grid):.2f} m")
    
    return dem_grid, dem_extent


def sample_dem_at_points(x_m, y_m, dem_grid, dem_extent, dem_resolution):
    """Sample DEM values at point locations."""
    xmin, ymin, xmax, ymax = dem_extent
    nrows, ncols = dem_grid.shape
    
    cols = np.floor((x_m - xmin) / dem_resolution).astype(int)
    rows = np.floor((ymax - y_m) / dem_resolution).astype(int)
    
    cols = np.clip(cols, 0, ncols - 1)
    rows = np.clip(rows, 0, nrows - 1)
    
    return dem_grid[rows, cols]


def calculate_slope_from_dem(dem_grid, dem_resolution, nodata=-9999):
    """Calculate slope (degrees) from DEM using Sobel filter."""
    print(f"\nCalculating slope from DEM...")
    
    valid_mask = ~np.isnan(dem_grid)
    if not np.any(valid_mask):
        return np.full_like(dem_grid, nodata)
    
    dem_filled = dem_grid.copy()
    dem_filled[~valid_mask] = np.nanmean(dem_grid)
    
    dz_dx = sobel(dem_filled, axis=1) / (8.0 * dem_resolution)
    dz_dy = sobel(dem_filled, axis=0) / (8.0 * dem_resolution)
    
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)
    
    slope_result = np.full_like(dem_grid, nodata, dtype=np.float32)
    slope_result[valid_mask] = slope_deg[valid_mask]
    
    print(f"  Slope range: {np.nanmin(slope_result[slope_result != nodata]):.2f}° to "
          f"{np.nanmax(slope_result[slope_result != nodata]):.2f}°")
    
    return slope_result


def resample_to_target_grid(source_grid, source_extent, source_resolution,
                             target_xmin, target_ymax, target_nrows, target_ncols,
                             target_resolution, nodata=-9999):
    """Resample a higher-resolution grid to match target metric grid."""
    target_grid = np.full((target_nrows, target_ncols), nodata, dtype=np.float32)
    src_xmin, src_ymin, src_xmax, src_ymax = source_extent
    src_nrows, src_ncols = source_grid.shape
    
    for tr in range(target_nrows):
        for tc in range(target_ncols):
            tx = target_xmin + (tc + 0.5) * target_resolution
            ty = target_ymax - (tr + 0.5) * target_resolution
            
            sc = int((tx - src_xmin) / source_resolution)
            sr = int((src_ymax - ty) / source_resolution)
            
            if 0 <= sr < src_nrows and 0 <= sc < src_ncols:
                val = source_grid[sr, sc]
                if not np.isnan(val) and val != nodata:
                    target_grid[tr, tc] = val
    
    return target_grid


# ==================== MAIN PROCESSING ====================

def normalize_lidar(las_path, output_folder=None):
    """Read LAS, filter noise, convert to meters, create DEM, normalize heights."""
    print(f"\nReading LiDAR file: {las_path}")
    las = laspy.read(las_path)
    
    try:
        las_crs = las.header.parse_crs()
    except:
        las_crs = None
    
    h_unit, h_to_meter, v_to_meter, meter_crs, source_epsg, target_epsg = get_crs_info(las_crs)
    
    x_orig = np.array(las.x)
    y_orig = np.array(las.y)
    z_orig = np.array(las.z)
    
    try:
        classification = np.array(las.classification)
    except:
        raise ValueError("LAS file has no classification field!")
    
    try:
        return_num = np.array(las.return_number)
        num_returns = np.array(las.number_of_returns)
    except:
        return_num = np.ones_like(x_orig, dtype=int)
        num_returns = np.ones_like(x_orig, dtype=int)
    
    n_initial = len(x_orig)
    print(f"\nInitial points: {n_initial:,}")
    
    # Filter noise classifications
    print(f"\nFilter 1: Removing noise classifications {NOISE_CLASSIFICATIONS}...")
    n_noise_removed = 0
    for cls in NOISE_CLASSIFICATIONS:
        n_noise = np.sum(classification == cls)
        n_noise_removed += int(n_noise)
        if n_noise > 0:
            print(f"  Class {cls} (noise): {n_noise:,} points")
    
    not_noise_mask = ~np.isin(classification, NOISE_CLASSIFICATIONS)
    x_orig = x_orig[not_noise_mask]
    y_orig = y_orig[not_noise_mask]
    z_orig = z_orig[not_noise_mask]
    classification = classification[not_noise_mask]
    return_num = return_num[not_noise_mask]
    num_returns = num_returns[not_noise_mask]
    
    print(f"  Points after noise filter: {len(x_orig):,}")
    
    # Convert to METERS (ONE-WAY conversion)
    print("\nConverting all coordinates to METERS...")
    x_m = x_orig * h_to_meter
    y_m = y_orig * h_to_meter
    z_m = z_orig * v_to_meter
    
    print(f"  X range: {x_m.min():.2f} to {x_m.max():.2f} m")
    print(f"  Y range: {y_m.min():.2f} to {y_m.max():.2f} m")
    print(f"  Z range: {z_m.min():.2f} to {z_m.max():.2f} m")
    
    # Create DEM in meters
    dem_grid, dem_extent = create_dem_from_lidar(
        x_m, y_m, z_m, classification, DEM_RESOLUTION
    )
    
    # Calculate slope from DEM
    slope_grid = calculate_slope_from_dem(dem_grid, DEM_RESOLUTION)
    
    # Save DEM
    if SAVE_DEM and output_folder is not None:
        base_name = os.path.splitext(os.path.basename(las_path))[0]
        dem_output_path = os.path.join(output_folder, f"{base_name}_DEM.tif")
        
        xmin_m, ymin_m, xmax_m, ymax_m = dem_extent
        dem_transform = from_origin(xmin_m, ymax_m, DEM_RESOLUTION, DEM_RESOLUTION)
        
        with rasterio.open(
            dem_output_path, 'w', driver='GTiff',
            height=dem_grid.shape[0], width=dem_grid.shape[1],
            count=1, dtype='float32', crs=meter_crs,
            transform=dem_transform, nodata=-9999, compress='lzw'
        ) as dst:
            dst.write(dem_grid.astype(np.float32), 1)
        
        print(f"  Saved DEM: {dem_output_path}")
    
    # Normalize heights (AGL)
    print("\nNormalizing heights (AGL)...")
    ground_elevation = sample_dem_at_points(x_m, y_m, dem_grid, dem_extent, DEM_RESOLUTION)
    z_agl = z_m - ground_elevation
    
    # Apply AGL thresholds
    print(f"\nFilter 2: AGL thresholds")
    print(f"  Min AGL: {MIN_AGL_THRESHOLD}m")
    print(f"  Max AGL: {MAX_AGL_THRESHOLD}m (noise safety net)")
    
    n_too_high = int(np.sum(z_agl > MAX_AGL_THRESHOLD))
    if n_too_high > 0:
        print(f"  Removed {n_too_high:,} points > {MAX_AGL_THRESHOLD}m (noise)")
    
    valid = ((z_agl >= MIN_AGL_THRESHOLD) & 
             (z_agl <= MAX_AGL_THRESHOLD) & 
             (~np.isnan(z_agl)))
    
    x_m = x_m[valid]
    y_m = y_m[valid]
    z_agl = z_agl[valid]
    return_num = return_num[valid]
    num_returns = num_returns[valid]
    
    print(f"  Final valid AGL points: {len(z_agl):,}")
    print(f"  AGL range: {z_agl.min():.2f} to {z_agl.max():.2f} m")
    
    # ============================================
    # USE FILTERED POINT EXTENTS + GRID SNAPPING
    # ============================================
    # Use actual filtered point extents (not LAS header)
    las_xmin = np.min(x_m)
    las_xmax = np.max(x_m)
    las_ymin = np.min(y_m)
    las_ymax = np.max(y_m)
    
    print(f"\nUsing FILTERED POINT extents:")
    print(f"  X: {las_xmin:.2f} to {las_xmax:.2f} m")
    print(f"  Y: {las_ymin:.2f} to {las_ymax:.2f} m")
    
    # Snap to grid for clean alignment
    las_xmin = np.floor(las_xmin / CELL_SIZE) * CELL_SIZE
    las_ymin = np.floor(las_ymin / CELL_SIZE) * CELL_SIZE
    las_xmax = np.ceil(las_xmax / CELL_SIZE) * CELL_SIZE
    las_ymax = np.ceil(las_ymax / CELL_SIZE) * CELL_SIZE
    
    print(f"After grid snapping to {CELL_SIZE}m cells:")
    print(f"  X: {las_xmin:.2f} to {las_xmax:.2f} m")
    print(f"  Y: {las_ymin:.2f} to {las_ymax:.2f} m")
    
    # Grid dimensions (exact multiples after snapping)
    ncols = int((las_xmax - las_xmin) / CELL_SIZE)
    nrows = int((las_ymax - las_ymin) / CELL_SIZE)
    
    print(f"  Metric grid: {nrows} x {ncols} (cell = {CELL_SIZE}m)")
    
    grid_info = {
        'xmin': las_xmin,
        'xmax': las_xmax,
        'ymin': las_ymin,
        'ymax': las_ymax,
        'ncols': ncols,
        'nrows': nrows,
        'crs': meter_crs,
        'h_unit_orig': h_unit,
        'source_epsg': source_epsg,
        'target_epsg': target_epsg,
        'dem_grid': dem_grid,
        'dem_extent': dem_extent,
        'slope_grid': slope_grid,
        'processing_info': {
            'initial_points': n_initial,
            'noise_points_removed': n_noise_removed,
            'high_agl_points_removed': n_too_high,
            'valid_agl_points': len(z_agl),
        }
    }
    
    return x_m, y_m, z_agl, return_num, num_returns, grid_info


def assign_points_to_pixels(x, y, z_agl, return_num, num_returns, grid_info):
    """Assign each point to its grid cell."""
    print("\nAssigning points to pixels...")
    
    xmin = grid_info['xmin']
    ymax = grid_info['ymax']
    nrows = grid_info['nrows']
    ncols = grid_info['ncols']
    
    cols = np.floor((x - xmin) / CELL_SIZE).astype(int)
    rows = np.floor((ymax - y) / CELL_SIZE).astype(int)
    
    inside = (rows >= 0) & (rows < nrows) & (cols >= 0) & (cols < ncols)
    
    n_outside = int(np.sum(~inside))
    if n_outside > 0:
        print(f"  Note: {n_outside:,} points outside snapped grid (clipped)")
    
    rows = rows[inside]
    cols = cols[inside]
    z_agl = z_agl[inside]
    return_num = return_num[inside]
    num_returns = num_returns[inside]
    
    pixel_dict = {}
    for i in range(len(rows)):
        pixel_key = (rows[i], cols[i])
        if pixel_key not in pixel_dict:
            pixel_dict[pixel_key] = {'z': [], 'return_num': [], 'num_returns': []}
        pixel_dict[pixel_key]['z'].append(z_agl[i])
        pixel_dict[pixel_key]['return_num'].append(return_num[i])
        pixel_dict[pixel_key]['num_returns'].append(num_returns[i])
    
    for key in pixel_dict:
        pixel_dict[key]['z'] = np.array(pixel_dict[key]['z'])
        pixel_dict[key]['return_num'] = np.array(pixel_dict[key]['return_num'])
        pixel_dict[key]['num_returns'] = np.array(pixel_dict[key]['num_returns'])
    
    print(f"  Pixels with data: {len(pixel_dict):,} / {nrows*ncols:,} ({100*len(pixel_dict)/(nrows*ncols):.1f}%)")
    return pixel_dict


def calculate_pixel_metrics(pixel_data):
    """Calculate all metrics for one pixel. All inputs/outputs in METERS."""
    z = pixel_data['z']
    return_num = pixel_data['return_num']
    num_returns = pixel_data['num_returns']
    
    n_total = len(z)
    if n_total == 0:
        return None
    
    metrics = {}
    
    # HEIGHT BINS (in meters)
    for name, z_min, z_max in HEIGHT_BINS:
        count = np.sum((z >= z_min) & (z < z_max))
        metrics[name] = count / n_total
    
    # CUMULATIVE DENSITY (in meters)
    for name, threshold in DENS_THRESHOLDS:
        count = np.sum(z < threshold)
        metrics[name] = count / n_total
    
    # PERCENTILES
    if n_total >= 2:
        percentile_values = np.percentile(z, PERCENTILES)
        for i, p in enumerate(PERCENTILES):
            metrics[f'p{p}'] = percentile_values[i]
    else:
        for p in PERCENTILES:
            metrics[f'p{p}'] = z[0] if n_total == 1 else 0
    
    # DISTRIBUTION STATS
    metrics['lida_mn'] = np.mean(z)
    metrics['lida_med'] = np.median(z)
    metrics['lida_stdv'] = np.std(z) if n_total > 1 else 0
    metrics['lida_skw'] = stats.skew(z) if n_total > 2 else 0
    metrics['lida_kurt'] = stats.kurtosis(z) if n_total > 3 else 0
    
    # Q_MEAN - Quadratic (RMS) mean height
    metrics['q_mean'] = np.sqrt(np.mean(z**2))
    
    # C_MEAN - Cubic mean height
    mean_cubed = np.mean(z**3)
    if mean_cubed >= 0:
        metrics['c_mean'] = mean_cubed ** (1.0/3.0)
    else:
        metrics['c_mean'] = -((-mean_cubed) ** (1.0/3.0))
    
    # CHM
    metrics['chm'] = np.max(z)
    
    # CANOPY/UNDERSTORY (1.37m threshold)
    canopy_mask = z > CANOPY_THRESHOLD
    understory_mask = z < CANOPY_THRESHOLD
    z_canopy = z[canopy_mask]
    z_understory = z[understory_mask]
    
    if len(z_canopy) > 0:
        metrics['cnpy_min'] = np.min(z_canopy)
        metrics['cnpy_max'] = np.max(z_canopy)
        metrics['cnpy_mean'] = np.mean(z_canopy)
    else:
        metrics['cnpy_min'] = 0
        metrics['cnpy_max'] = 0
        metrics['cnpy_mean'] = 0
    
    if len(z_understory) > 0:
        metrics['und_min'] = np.min(z_understory)
        metrics['und_max'] = np.max(z_understory)
        metrics['und_mean'] = np.mean(z_understory)
    else:
        metrics['und_min'] = 0
        metrics['und_max'] = 0
        metrics['und_mean'] = 0
    
    # VARIABILITY
    if len(z_canopy) > 1:
        metrics['cv_canopy'] = np.std(z_canopy) / np.mean(z_canopy) if np.mean(z_canopy) > 0 else 0
        metrics['var_canopy'] = np.var(z_canopy)
    else:
        metrics['cv_canopy'] = 0
        metrics['var_canopy'] = 0
    
    if len(z_understory) > 1:
        metrics['cv_understory'] = np.std(z_understory) / np.mean(z_understory) if np.mean(z_understory) > 0 else 0
        metrics['var_understory'] = np.var(z_understory)
    else:
        metrics['cv_understory'] = 0
        metrics['var_understory'] = 0
    
    # CRR - Canopy Relief Ratio
    if len(z_canopy) > 0:
        z_max = np.max(z_canopy)
        z_min = np.min(z_canopy)
        z_mean = np.mean(z_canopy)
        if (z_max - z_min) > 0:
            metrics['crr'] = (z_mean - z_min) / (z_max - z_min)
        else:
            metrics['crr'] = 0
    else:
        metrics['crr'] = 0
    
    # COVER METRICS
    first_return_mask = return_num == 1
    first_returns_canopy = np.sum(first_return_mask & canopy_mask)
    first_returns_understory = np.sum(first_return_mask & understory_mask)
    first_returns_total = np.sum(first_return_mask)
    
    metrics['first_ccov'] = first_returns_canopy / first_returns_total if first_returns_total > 0 else 0
    metrics['first_ucov'] = first_returns_understory / first_returns_total if first_returns_total > 0 else 0
    metrics['all_ccov'] = np.sum(canopy_mask) / n_total
    metrics['all_ucov'] = np.sum(understory_mask) / n_total
    
    return metrics


def process_pixel_chunk(pixel_keys, pixel_dict):
    results = {}
    for key in pixel_keys:
        metrics = calculate_pixel_metrics(pixel_dict[key])
        if metrics is not None:
            results[key] = metrics
    return results


def calculate_surface_area_3d(heights, cell_size):
    """3D surface area in a window using triangulation."""
    rows, cols = heights.shape
    if rows < 2 or cols < 2:
        return cell_size * cell_size * rows * cols
    
    surface_area = 0.0
    
    for i in range(rows - 1):
        for j in range(cols - 1):
            z00 = heights[i, j]
            z01 = heights[i, j + 1]
            z10 = heights[i + 1, j]
            z11 = heights[i + 1, j + 1]
            
            if np.isnan(z00) or np.isnan(z01) or np.isnan(z10) or np.isnan(z11):
                continue
            
            p1 = np.array([j * cell_size, i * cell_size, z00])
            p2 = np.array([(j + 1) * cell_size, i * cell_size, z01])
            p3 = np.array([j * cell_size, (i + 1) * cell_size, z10])
            v1 = p2 - p1
            v2 = p3 - p1
            area1 = 0.5 * np.linalg.norm(np.cross(v1, v2))
            
            p1 = np.array([(j + 1) * cell_size, i * cell_size, z01])
            p2 = np.array([(j + 1) * cell_size, (i + 1) * cell_size, z11])
            p3 = np.array([j * cell_size, (i + 1) * cell_size, z10])
            v1 = p2 - p1
            v2 = p3 - p1
            area2 = 0.5 * np.linalg.norm(np.cross(v1, v2))
            
            surface_area += area1 + area2
    
    return surface_area


def calculate_surface_complexity(chm_grid, cell_size, nodata=-9999):
    """Calculate rugosity and roughness from CHM."""
    nrows, ncols = chm_grid.shape
    rugosity = np.full((nrows, ncols), nodata, dtype=np.float32)
    roughness = np.full((nrows, ncols), nodata, dtype=np.float32)
    
    window_size = 2  # 5x5 window
    print("Calculating surface complexity (rugosity & roughness)...")
    
    for i in range(nrows):
        for j in range(ncols):
            i_min = max(0, i - window_size)
            i_max = min(nrows, i + window_size + 1)
            j_min = max(0, j - window_size)
            j_max = min(ncols, j + window_size + 1)
            
            window = chm_grid[i_min:i_max, j_min:j_max]
            valid_heights = window[window != nodata]
            
            if len(valid_heights) < 4:
                continue
            
            roughness[i, j] = np.std(valid_heights)
            
            window_heights = np.full((i_max - i_min, j_max - j_min), np.nan)
            for wi in range(i_max - i_min):
                for wj in range(j_max - j_min):
                    val = window[wi, wj]
                    if val != nodata:
                        window_heights[wi, wj] = val
            
            surface_area_3d = calculate_surface_area_3d(window_heights, cell_size)
            n_valid = len(valid_heights)
            surface_area_2d = n_valid * cell_size * cell_size
            
            if surface_area_2d > 0:
                rugosity[i, j] = surface_area_3d / surface_area_2d
            else:
                rugosity[i, j] = 1.0
    
    return rugosity, roughness


def save_raster(data, output_path, grid_info, nodata=-9999):
    """Save raster with proper meter CRS."""
    if grid_info['crs'] is None:
        raise ValueError("Cannot save raster: CRS is None!")
    
    transform = from_origin(grid_info['xmin'], grid_info['ymax'], CELL_SIZE, CELL_SIZE)
    
    with rasterio.open(
        output_path, 'w', driver='GTiff',
        height=grid_info['nrows'], width=grid_info['ncols'],
        count=1, dtype='float32', crs=grid_info['crs'],
        transform=transform, nodata=nodata, compress='lzw'
    ) as dst:
        dst.write(data.astype(np.float32), 1)


def get_all_metric_names():
    """Return list of all metric names produced."""
    metric_names = []
    metric_names.extend([name for name, _, _ in HEIGHT_BINS])
    metric_names.extend([name for name, _ in DENS_THRESHOLDS])
    metric_names.extend([f'p{p}' for p in PERCENTILES])
    metric_names.extend(['lida_mn', 'lida_med', 'lida_stdv', 'lida_skw', 'lida_kurt'])
    metric_names.extend(['q_mean', 'c_mean'])
    metric_names.extend(['chm'])
    metric_names.extend(['cnpy_min', 'cnpy_max', 'cnpy_mean'])
    metric_names.extend(['und_min', 'und_max', 'und_mean'])
    metric_names.extend(['cv_canopy', 'cv_understory', 'var_canopy', 'var_understory'])
    metric_names.extend(['crr'])
    metric_names.extend(['first_ccov', 'first_ucov', 'all_ccov', 'all_ucov'])
    metric_names.extend(['rugosity', 'roughness'])
    metric_names.extend(['slope'])
    return metric_names


# ==================== MAIN ====================

def main():
    print("="*70)
    print(f"{PIPELINE_NAME} v{PIPELINE_VERSION}")
    print(f"Pipeline designed by {PIPELINE_AUTHOR}")
    print("="*70)
    print(f"Input: {INPUT_LAS}")
    print(f"Output: {OUTPUT_FOLDER}")
    print(f"Cell size: {CELL_SIZE}m | DEM resolution: {DEM_RESOLUTION}m")
    print(f"Canopy threshold: {CANOPY_THRESHOLD}m | Max AGL: {MAX_AGL_THRESHOLD}m")
    print(f"Noise classes: {NOISE_CLASSIFICATIONS}")
    print("="*70)
    
    # Pre-processing QA
    print("\n[QA] Pre-processing validation...")
    pre_qa = validate_las_file(INPUT_LAS)
    
    if pre_qa['errors']:
        for err in pre_qa['errors']:
            print(f"  ✗ {err}")
        if not pre_qa['passed']:
            print("\n[QA] Pre-processing FAILED. Aborting.")
            return
    
    if pre_qa['warnings']:
        for warn in pre_qa['warnings']:
            print(f"  ⚠ {warn}")
    
    if pre_qa['passed'] and not pre_qa['warnings']:
        print("[QA] ✓ Pre-processing validation passed")
    
    # Main processing
    x, y, z_agl, return_num, num_returns, grid_info = normalize_lidar(
        INPUT_LAS, output_folder=OUTPUT_FOLDER
    )
    
    pixel_dict = assign_points_to_pixels(x, y, z_agl, return_num, num_returns, grid_info)
    
    print(f"\nProcessing {len(pixel_dict):,} pixels with {NUM_PROCESSES} processes...")
    
    pixel_keys = list(pixel_dict.keys())
    chunk_size = len(pixel_keys) // NUM_PROCESSES + 1
    pixel_chunks = [pixel_keys[i:i + chunk_size] for i in range(0, len(pixel_keys), chunk_size)]
    
    process_func = partial(process_pixel_chunk, pixel_dict=pixel_dict)
    
    with Pool(processes=NUM_PROCESSES) as pool:
        chunk_results = pool.map(process_func, pixel_chunks)
    
    all_metrics = {}
    for chunk_result in chunk_results:
        all_metrics.update(chunk_result)
    
    print(f"Processed {len(all_metrics):,} pixels")
    
    nrows = grid_info['nrows']
    ncols = grid_info['ncols']
    nodata = -9999
    
    first_pixel_metrics = next(iter(all_metrics.values()))
    metric_names = list(first_pixel_metrics.keys())
    
    metric_grids = {}
    for metric_name in metric_names:
        metric_grids[metric_name] = np.full((nrows, ncols), nodata, dtype=np.float32)
    
    for (row, col), metrics in all_metrics.items():
        for metric_name, value in metrics.items():
            metric_grids[metric_name][row, col] = value
    
    # Surface complexity
    chm_grid = metric_grids['chm']
    rugosity, roughness = calculate_surface_complexity(chm_grid, CELL_SIZE, nodata)
    metric_grids['rugosity'] = rugosity
    metric_grids['roughness'] = roughness
    
    # Slope (resampled from DEM resolution to metric resolution)
    print("\nResampling slope to metric grid...")
    slope_resampled = resample_to_target_grid(
        grid_info['slope_grid'], grid_info['dem_extent'], DEM_RESOLUTION,
        grid_info['xmin'], grid_info['ymax'], nrows, ncols, CELL_SIZE, nodata
    )
    metric_grids['slope'] = slope_resampled
    
    print("\nSaving rasters...")
    base_name = os.path.splitext(os.path.basename(INPUT_LAS))[0]
    
    for metric_name, grid in metric_grids.items():
        output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_{metric_name}.tif")
        save_raster(grid, output_path, grid_info, nodata)
    
    # Post-processing QA
    print("\n[QA] Post-processing validation...")
    expected_metrics = get_all_metric_names()
    post_qa = validate_outputs(OUTPUT_FOLDER, base_name, expected_metrics)
    
    if post_qa['warnings']:
        for warn in post_qa['warnings']:
            print(f"  ⚠ {warn}")
    if post_qa['errors']:
        for err in post_qa['errors']:
            print(f"  ✗ {err}")
    if post_qa['passed'] and not post_qa['warnings']:
        print("[QA] ✓ Post-processing validation passed")
    
    if post_qa['stats']:
        print("\n[QA] Statistics:")
        for key, val in post_qa['stats'].items():
            if isinstance(val, (int, float)):
                print(f"  {key}: {val:.3f}")
            else:
                print(f"  {key}: {val}")
    
    # Generate metadata
    print("\nGenerating metadata...")
    processing_info = {
        'input': {
            'point_count': grid_info['processing_info']['initial_points'],
            'crs_wkt_preview': str(grid_info['crs'].to_wkt())[:200],
            'source_epsg': grid_info['source_epsg'],
        },
        'output': {
            'grid_rows': nrows,
            'grid_cols': ncols,
            'total_rasters': len(metric_grids) + 1,  # +1 for DEM
            'target_epsg': grid_info['target_epsg'],
            'metrics_produced': sorted(list(metric_grids.keys()) + ['DEM']),
            'noise_points_removed': grid_info['processing_info']['noise_points_removed'],
            'high_agl_points_removed': grid_info['processing_info']['high_agl_points_removed'],
            'valid_agl_points': grid_info['processing_info']['valid_agl_points'],
        },
        'pre_qa': pre_qa,
    }
    
    metadata_path = generate_metadata(INPUT_LAS, OUTPUT_FOLDER, processing_info, post_qa)
    print(f"  Saved metadata: {metadata_path}")
    
    print("\n" + "="*70)
    print(f"COMPLETED! {len(metric_grids) + 1} rasters in METERS")
    print(f"Output CRS: EPSG:{grid_info['target_epsg']}")
    print(f"Pipeline designed by {PIPELINE_AUTHOR}")
    print("="*70)


if __name__ == "__main__":
    main()
