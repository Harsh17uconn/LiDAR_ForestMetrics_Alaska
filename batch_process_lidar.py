"""
Batch LiDAR Metrics Processing
================================
Pipeline designed by Harshana Wedagedara

Processes multiple LAS files producing meter-based rasters with QA/QC and metadata.
"""
import os
import glob
import json
import time
from datetime import datetime
import sys

sys.path.append(os.path.dirname(__file__))
from lidar_metrics_parallel import (
    PIPELINE_NAME, PIPELINE_VERSION, PIPELINE_AUTHOR,
    normalize_lidar, 
    assign_points_to_pixels,
    process_pixel_chunk,
    calculate_surface_complexity,
    resample_to_target_grid,
    save_raster,
    validate_las_file,
    validate_outputs,
    generate_metadata,
    get_all_metric_names,
    HEIGHT_BINS, DENS_THRESHOLDS, PERCENTILES,
    CELL_SIZE, DEM_RESOLUTION,
    NUM_PROCESSES, NOISE_CLASSIFICATIONS,
    MAX_AGL_THRESHOLD,
)

from multiprocessing import Pool
from functools import partial
import numpy as np

# ==================== BATCH CONFIGURATION ====================

LAS_FOLDER = r"/gpfs/sharedfs1/chw06003/SHL/AK_LiDAR/2017_Lidar/2017_Aerial_Lidar_Creamers_Field/"
OUTPUT_FOLDER = r"/gpfs/sharedfs1/chw06003/Shashika/Test/Metricsall_2"

LAS_PATTERN = "*.las"
SKIP_EXISTING = True
LOG_FILE = os.path.join(OUTPUT_FOLDER, "batch_processing_log.txt")
BATCH_METADATA_FILE = os.path.join(OUTPUT_FOLDER, "batch_metadata.json")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    with open(LOG_FILE, 'a') as f:
        f.write(log_entry + "\n")


def process_single_file(las_path, output_folder):
    start_time = time.time()
    base_name = os.path.splitext(os.path.basename(las_path))[0]
    
    file_record = {
        'file': base_name,
        'status': 'unknown',
        'processing_time_seconds': 0,
        'qa_passed': False,
        'qa_warnings': [],
        'qa_errors': [],
        'metrics_produced': 0,
    }
    
    try:
        log_message(f"Processing: {base_name}")
        
        # Pre-processing QA
        pre_qa = validate_las_file(las_path)
        file_record['qa_warnings'].extend(pre_qa.get('warnings', []))
        file_record['qa_errors'].extend(pre_qa.get('errors', []))
        
        if not pre_qa['passed']:
            log_message(f"  ✗ Pre-QA failed: {pre_qa['errors']}")
            file_record['status'] = 'failed_qa'
            file_record['processing_time_seconds'] = round(time.time() - start_time, 1)
            return False, file_record
        
        if pre_qa['warnings']:
            for warn in pre_qa['warnings']:
                log_message(f"  ⚠ QA warning: {warn}")
        
        # Main processing
        x, y, z_agl, return_num, num_returns, grid_info = normalize_lidar(
            las_path, output_folder=output_folder
        )
        
        log_message(f"  CRS: EPSG:{grid_info['source_epsg']} -> EPSG:{grid_info['target_epsg']}")
        log_message(f"  Valid AGL points: {len(z_agl):,}")
        log_message(f"  Max AGL: {z_agl.max():.2f}m")
        
        pixel_dict = assign_points_to_pixels(x, y, z_agl, return_num, num_returns, grid_info)
        
        pixel_keys = list(pixel_dict.keys())
        chunk_size = len(pixel_keys) // NUM_PROCESSES + 1
        pixel_chunks = [pixel_keys[i:i + chunk_size] for i in range(0, len(pixel_keys), chunk_size)]
        
        process_func = partial(process_pixel_chunk, pixel_dict=pixel_dict)
        
        with Pool(processes=NUM_PROCESSES) as pool:
            chunk_results = pool.map(process_func, pixel_chunks)
        
        all_metrics = {}
        for chunk_result in chunk_results:
            all_metrics.update(chunk_result)
        
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
        slope_resampled = resample_to_target_grid(
            grid_info['slope_grid'], grid_info['dem_extent'], DEM_RESOLUTION,
            grid_info['xmin'], grid_info['ymax'], nrows, ncols, CELL_SIZE, nodata
        )
        metric_grids['slope'] = slope_resampled
        
        # Save all rasters
        for metric_name, grid in metric_grids.items():
            output_path = os.path.join(output_folder, f"{base_name}_{metric_name}.tif")
            save_raster(grid, output_path, grid_info, nodata)
        
        # Post-processing QA
        expected_metrics = get_all_metric_names()
        post_qa = validate_outputs(output_folder, base_name, expected_metrics)
        
        file_record['qa_warnings'].extend(post_qa.get('warnings', []))
        file_record['qa_errors'].extend(post_qa.get('errors', []))
        file_record['qa_stats'] = post_qa.get('stats', {})
        
        for warn in post_qa.get('warnings', []):
            log_message(f"  ⚠ QA warning: {warn}")
        for err in post_qa.get('errors', []):
            log_message(f"  ✗ QA error: {err}")
        
        # Generate per-file metadata
        processing_info = {
            'input': {
                'point_count': grid_info['processing_info']['initial_points'],
                'crs_wkt_preview': str(grid_info['crs'].to_wkt())[:200],
                'source_epsg': grid_info['source_epsg'],
            },
            'output': {
                'grid_rows': nrows,
                'grid_cols': ncols,
                'total_rasters': len(metric_grids) + 1,
                'target_epsg': grid_info['target_epsg'],
                'metrics_produced': sorted(list(metric_grids.keys()) + ['DEM']),
                'noise_points_removed': grid_info['processing_info']['noise_points_removed'],
                'high_agl_points_removed': grid_info['processing_info']['high_agl_points_removed'],
                'valid_agl_points': grid_info['processing_info']['valid_agl_points'],
            },
            'pre_qa': pre_qa,
        }
        
        generate_metadata(las_path, output_folder, processing_info, post_qa)
        
        elapsed_time = time.time() - start_time
        log_message(f"  ✓ Completed {base_name} in {elapsed_time:.1f}s ({len(metric_grids)+1} metrics)")
        
        file_record['status'] = 'success'
        file_record['processing_time_seconds'] = round(elapsed_time, 1)
        file_record['qa_passed'] = post_qa['passed']
        file_record['metrics_produced'] = len(metric_grids) + 1
        file_record['target_epsg'] = grid_info['target_epsg']
        
        return True, file_record
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        error_msg = f"ERROR processing {base_name}: {str(e)}"
        log_message(f"  ✗ {error_msg}")
        import traceback
        log_message(f"  Traceback: {traceback.format_exc()}")
        
        file_record['status'] = 'error'
        file_record['processing_time_seconds'] = round(elapsed_time, 1)
        file_record['qa_errors'].append(str(e))
        
        return False, file_record


def find_las_files():
    las_files = glob.glob(os.path.join(LAS_FOLDER, LAS_PATTERN))
    las_files.sort()
    
    file_list = []
    
    for las_path in las_files:
        base_name = os.path.splitext(os.path.basename(las_path))[0]
        file_output_folder = os.path.join(OUTPUT_FOLDER, base_name)
        
        if SKIP_EXISTING and os.path.exists(file_output_folder):
            existing_files = glob.glob(os.path.join(file_output_folder, "*.tif"))
            if len(existing_files) >= 70:
                log_message(f"SKIP: {base_name} already processed")
                continue
        
        os.makedirs(file_output_folder, exist_ok=True)
        file_list.append((las_path, file_output_folder))
    
    return file_list


def save_batch_metadata(batch_info):
    with open(BATCH_METADATA_FILE, 'w') as f:
        json.dump(batch_info, f, indent=2)


def main():
    log_message("="*70)
    log_message(f"{PIPELINE_NAME} v{PIPELINE_VERSION} - BATCH PROCESSING")
    log_message(f"Pipeline designed by {PIPELINE_AUTHOR}")
    log_message("="*70)
    log_message(f"LAS folder: {LAS_FOLDER}")
    log_message(f"Output folder: {OUTPUT_FOLDER}")
    log_message(f"Cell size: {CELL_SIZE}m | DEM resolution: {DEM_RESOLUTION}m")
    log_message(f"Max AGL: {MAX_AGL_THRESHOLD}m | Noise classes: {NOISE_CLASSIFICATIONS}")
    log_message(f"Processes: {NUM_PROCESSES}")
    log_message("="*70)
    
    file_list = find_las_files()
    
    if len(file_list) == 0:
        log_message("No files to process!")
        return
    
    log_message(f"\nFound {len(file_list)} LAS files\n")
    
    batch_info = {
        'pipeline': {
            'name': PIPELINE_NAME,
            'version': PIPELINE_VERSION,
            'designed_by': PIPELINE_AUTHOR,
        },
        'batch_run': {
            'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'input_folder': LAS_FOLDER,
            'output_folder': OUTPUT_FOLDER,
            'total_files': len(file_list),
        },
        'files': []
    }
    
    total_start = time.time()
    success_count = 0
    fail_count = 0
    total_processing_time = 0
    
    for i, (las_path, output_folder) in enumerate(file_list, 1):
        log_message(f"\n[{i}/{len(file_list)}] Starting...")
        success, file_record = process_single_file(las_path, output_folder)
        
        batch_info['files'].append(file_record)
        
        if success:
            success_count += 1
            total_processing_time += file_record['processing_time_seconds']
        else:
            fail_count += 1
    
    total_elapsed = time.time() - total_start
    
    batch_info['batch_run']['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    batch_info['batch_run']['total_time_minutes'] = round(total_elapsed / 60, 2)
    batch_info['batch_run']['successful'] = success_count
    batch_info['batch_run']['failed'] = fail_count
    batch_info['batch_run']['avg_time_per_file_seconds'] = round(
        total_processing_time / max(success_count, 1), 1
    )
    
    n_with_warnings = sum(1 for f in batch_info['files'] if f.get('qa_warnings'))
    n_with_errors = sum(1 for f in batch_info['files'] if f.get('qa_errors'))
    
    batch_info['batch_run']['qa_summary'] = {
        'files_with_warnings': n_with_warnings,
        'files_with_errors': n_with_errors,
    }
    
    save_batch_metadata(batch_info)
    
    log_message("\n" + "="*70)
    log_message("BATCH PROCESSING COMPLETE")
    log_message("="*70)
    log_message(f"Total files: {len(file_list)}")
    log_message(f"Successful: {success_count}")
    log_message(f"Failed: {fail_count}")
    log_message(f"Files with QA warnings: {n_with_warnings}")
    log_message(f"Files with QA errors: {n_with_errors}")
    log_message(f"Total time: {total_elapsed/60:.1f} minutes")
    log_message(f"Average per file: {total_processing_time/max(success_count,1):.1f} seconds")
    log_message(f"\nBatch metadata: {BATCH_METADATA_FILE}")
    log_message(f"\nPipeline designed by {PIPELINE_AUTHOR}")
    log_message("="*70)


if __name__ == "__main__":
    main()
