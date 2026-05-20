"""
QA/QC Validation Script for LiDAR Metrics Output
==================================================
Pipeline designed by Harshana Wedagedara

Validates output rasters (in METERS) from the LiDAR metrics pipeline.
Generates a comprehensive QA report.
"""
import os
import glob
import json
from datetime import datetime
import numpy as np
import rasterio
import warnings
warnings.filterwarnings('ignore')

PIPELINE_AUTHOR = "Harshana Wedagedara"

# ==================== CONFIGURATION ====================

INPUT_FOLDER = r"/gpfs/sharedfs1/chw06003/Shashika/Test/Metricsall"
QA_REPORT_FILE = os.path.join(INPUT_FOLDER, "qa_qc_report.json")
QA_SUMMARY_FILE = os.path.join(INPUT_FOLDER, "qa_qc_summary.txt")

# Validation tolerances (values in meters)
QA_MAX_CHM = 50.0
QA_MAX_NODATA_PCT = 20.0


# ==================== EXPECTED METRICS ====================

EXPECTED_METRICS = [
    # Height bins
    'hb_0_0.5', 'hb_0.5_1', 'hb_1_2', 'hb_2_3', 'hb_3_4', 'hb_4_5',
    'hb_5_10', 'hb_10_15', 'hb_15_20', 'hb_20_25', 'hb_25_30', 'hb_more30',
    # Cumulative density
    'dens_0_1', 'dens_0_2', 'dens_0_3', 'dens_0_5', 'dens_0_10',
    'dens_0_15', 'dens_0_20', 'dens_0_25', 'dens_0_30',
    # Percentiles
    'p5', 'p10', 'p15', 'p20', 'p25', 'p30', 'p35', 'p40', 'p45',
    'p50', 'p55', 'p60', 'p65', 'p70', 'p75', 'p80', 'p85', 'p90', 'p95', 'p99',
    # Distribution stats
    'lida_mn', 'lida_med', 'lida_stdv', 'lida_skw', 'lida_kurt',
    # Central tendency
    'q_mean', 'c_mean',
    # CHM
    'chm',
    # Canopy/understory
    'cnpy_min', 'cnpy_max', 'cnpy_mean',
    'und_min', 'und_max', 'und_mean',
    # Variability
    'cv_canopy', 'cv_understory', 'var_canopy', 'var_understory',
    # Vertical structure
    'crr',
    # Cover
    'first_ccov', 'first_ucov', 'all_ccov', 'all_ucov',
    # Surface complexity
    'rugosity', 'roughness',
    # Terrain
    'slope', 'DEM',
]

# Metrics that should be in meters (heights)
HEIGHT_METRICS = {
    'chm', 'cnpy_max', 'cnpy_mean', 'cnpy_min',
    'und_max', 'und_mean', 'und_min',
    'lida_mn', 'lida_med', 'lida_stdv',
    'q_mean', 'c_mean', 'roughness',
    'p5', 'p10', 'p15', 'p20', 'p25', 'p30', 'p35', 'p40', 'p45',
    'p50', 'p55', 'p60', 'p65', 'p70', 'p75', 'p80', 'p85', 'p90', 'p95', 'p99',
    'DEM',
}

# Metrics that should be 0-1 (ratios/proportions)
PROPORTION_METRICS = {
    'hb_0_0.5', 'hb_0.5_1', 'hb_1_2', 'hb_2_3', 'hb_3_4', 'hb_4_5',
    'hb_5_10', 'hb_10_15', 'hb_15_20', 'hb_20_25', 'hb_25_30', 'hb_more30',
    'dens_0_1', 'dens_0_2', 'dens_0_3', 'dens_0_5', 'dens_0_10',
    'dens_0_15', 'dens_0_20', 'dens_0_25', 'dens_0_30',
    'first_ccov', 'first_ucov', 'all_ccov', 'all_ucov',
    'crr',
}


def validate_raster(raster_path, metric_name):
    """Validate a single raster file."""
    result = {
        'file': os.path.basename(raster_path),
        'metric': metric_name,
        'passed': True,
        'warnings': [],
        'errors': [],
        'stats': {}
    }
    
    try:
        with rasterio.open(raster_path) as src:
            data = src.read(1)
            nodata = src.nodata
            crs = src.crs
            
            if crs is None:
                result['errors'].append("Missing CRS")
                result['passed'] = False
                return result
            
            result['stats']['crs'] = str(crs)
            result['stats']['shape'] = list(data.shape)
            result['stats']['cell_size'] = abs(src.transform[0])
            
            if nodata is not None:
                valid = data[data != nodata]
            else:
                valid = data[~np.isnan(data)]
            
            n_total = data.size
            n_valid = len(valid)
            nodata_pct = 100 * (n_total - n_valid) / n_total
            
            result['stats']['total_pixels'] = int(n_total)
            result['stats']['valid_pixels'] = int(n_valid)
            result['stats']['nodata_percent'] = round(nodata_pct, 2)
            
            if nodata_pct > QA_MAX_NODATA_PCT:
                result['warnings'].append(
                    f"High NoData: {nodata_pct:.1f}% (max: {QA_MAX_NODATA_PCT}%)"
                )
            
            if n_valid == 0:
                result['errors'].append("No valid data")
                result['passed'] = False
                return result
            
            result['stats']['min'] = float(np.min(valid))
            result['stats']['max'] = float(np.max(valid))
            result['stats']['mean'] = float(np.mean(valid))
            result['stats']['median'] = float(np.median(valid))
            result['stats']['std'] = float(np.std(valid))
            result['stats']['p95'] = float(np.percentile(valid, 95))
            result['stats']['p99'] = float(np.percentile(valid, 99))
            
            # Height-specific validation (values in meters)
            if metric_name in HEIGHT_METRICS:
                if metric_name in ['chm', 'cnpy_max']:
                    if result['stats']['max'] > QA_MAX_CHM:
                        result['warnings'].append(
                            f"Max {metric_name} ({result['stats']['max']:.1f}m) exceeds threshold ({QA_MAX_CHM}m)"
                        )
                    if result['stats']['min'] < -0.5:
                        result['warnings'].append(
                            f"Negative values: min = {result['stats']['min']:.2f}m"
                        )
            
            # Proportion validation
            elif metric_name in PROPORTION_METRICS:
                if result['stats']['min'] < -0.01 or result['stats']['max'] > 1.01:
                    result['warnings'].append(
                        f"Outside [0,1]: min={result['stats']['min']:.3f}, "
                        f"max={result['stats']['max']:.3f}"
                    )
            
            # Slope validation
            elif metric_name == 'slope':
                if result['stats']['min'] < 0 or result['stats']['max'] > 90:
                    result['warnings'].append(
                        f"Slope outside [0°, 90°]: min={result['stats']['min']:.1f}, "
                        f"max={result['stats']['max']:.1f}"
                    )
            
    except Exception as e:
        result['errors'].append(f"Error reading: {str(e)}")
        result['passed'] = False
    
    return result


def validate_folder(folder_path):
    """Validate all rasters in a folder."""
    folder_name = os.path.basename(folder_path.rstrip('/'))
    
    folder_result = {
        'folder': folder_name,
        'passed': True,
        'total_expected': len(EXPECTED_METRICS),
        'total_found': 0,
        'missing_metrics': [],
        'rasters': [],
        'summary': {
            'with_warnings': 0,
            'with_errors': 0,
            'no_crs': 0,
        }
    }
    
    for metric in EXPECTED_METRICS:
        possible_files = glob.glob(os.path.join(folder_path, f"*_{metric}.tif"))
        
        if not possible_files:
            direct_file = os.path.join(folder_path, f"{metric}.tif")
            if os.path.exists(direct_file):
                possible_files = [direct_file]
        
        if not possible_files:
            folder_result['missing_metrics'].append(metric)
            continue
        
        raster_result = validate_raster(possible_files[0], metric)
        folder_result['rasters'].append(raster_result)
        folder_result['total_found'] += 1
        
        if raster_result['warnings']:
            folder_result['summary']['with_warnings'] += 1
        if raster_result['errors']:
            folder_result['summary']['with_errors'] += 1
            folder_result['passed'] = False
        if 'Missing CRS' in str(raster_result.get('errors', [])):
            folder_result['summary']['no_crs'] += 1
    
    if folder_result['missing_metrics']:
        folder_result['passed'] = False
    
    return folder_result


def generate_summary_report(report):
    """Generate human-readable summary."""
    lines = []
    lines.append("="*70)
    lines.append(f"LiDAR Pipeline QA/QC Report")
    lines.append(f"Pipeline designed by {PIPELINE_AUTHOR}")
    lines.append("="*70)
    lines.append(f"Date: {report['date']}")
    lines.append(f"Input folder: {report['input_folder']}")
    lines.append(f"Folders processed: {len(report['folders'])}")
    lines.append("")
    
    total_passed = sum(1 for f in report['folders'] if f['passed'])
    total_failed = len(report['folders']) - total_passed
    
    lines.append(f"PASSED: {total_passed}")
    lines.append(f"FAILED: {total_failed}")
    lines.append("")
    
    lines.append("-"*70)
    lines.append("PER-FOLDER SUMMARY")
    lines.append("-"*70)
    
    for folder in report['folders']:
        status = "✓ PASS" if folder['passed'] else "✗ FAIL"
        lines.append(f"\n{status} - {folder['folder']}")
        lines.append(f"  Metrics found: {folder['total_found']}/{folder['total_expected']}")
        
        if folder['missing_metrics']:
            lines.append(f"  Missing: {', '.join(folder['missing_metrics'][:10])}")
        
        lines.append(f"  Rasters with warnings: {folder['summary']['with_warnings']}")
        lines.append(f"  Rasters with errors: {folder['summary']['with_errors']}")
        
        if folder['summary']['no_crs'] > 0:
            lines.append(f"  ⚠ Rasters missing CRS: {folder['summary']['no_crs']}")
        
        for raster in folder['rasters']:
            if raster['errors'] or raster['warnings']:
                lines.append(f"\n  {raster['metric']}:")
                for err in raster['errors']:
                    lines.append(f"    ✗ ERROR: {err}")
                for warn in raster['warnings']:
                    lines.append(f"    ⚠ WARNING: {warn}")
    
    lines.append("")
    lines.append("="*70)
    lines.append(f"Pipeline designed by {PIPELINE_AUTHOR}")
    lines.append("="*70)
    
    return "\n".join(lines)


def main():
    print("="*70)
    print(f"LiDAR Pipeline QA/QC Validation")
    print(f"Pipeline designed by {PIPELINE_AUTHOR}")
    print("="*70)
    print(f"Input folder: {INPUT_FOLDER}")
    print(f"Expected metrics per folder: {len(EXPECTED_METRICS)}")
    print("="*70)
    
    subfolders = [f for f in glob.glob(os.path.join(INPUT_FOLDER, "*")) 
                  if os.path.isdir(f)]
    
    if len(subfolders) == 0:
        print(f"\nNo subfolders found. Validating files in {INPUT_FOLDER}...")
        subfolders = [INPUT_FOLDER]
    
    print(f"\nValidating {len(subfolders)} folder(s)...")
    
    report = {
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'pipeline_author': PIPELINE_AUTHOR,
        'input_folder': INPUT_FOLDER,
        'expected_metrics': EXPECTED_METRICS,
        'folders': []
    }
    
    for i, folder in enumerate(subfolders, 1):
        print(f"\n[{i}/{len(subfolders)}] Validating: {os.path.basename(folder)}")
        folder_result = validate_folder(folder)
        report['folders'].append(folder_result)
        
        status = "PASS" if folder_result['passed'] else "FAIL"
        print(f"  Status: {status}")
        print(f"  Metrics: {folder_result['total_found']}/{folder_result['total_expected']}")
        print(f"  Warnings: {folder_result['summary']['with_warnings']}")
        print(f"  Errors: {folder_result['summary']['with_errors']}")
    
    with open(QA_REPORT_FILE, 'w') as f:
        json.dump(report, f, indent=2)
    
    summary = generate_summary_report(report)
    with open(QA_SUMMARY_FILE, 'w') as f:
        f.write(summary)
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)
    
    total_passed = sum(1 for f in report['folders'] if f['passed'])
    total_failed = len(report['folders']) - total_passed
    
    print(f"Total folders: {len(report['folders'])}")
    print(f"PASSED: {total_passed}")
    print(f"FAILED: {total_failed}")
    print(f"\nDetailed JSON report: {QA_REPORT_FILE}")
    print(f"Human-readable summary: {QA_SUMMARY_FILE}")
    print(f"\nPipeline designed by {PIPELINE_AUTHOR}")
    print("="*70)


if __name__ == "__main__":
    main()
