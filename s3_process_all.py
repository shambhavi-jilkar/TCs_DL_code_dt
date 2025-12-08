"""
Multi-Year TC Track Forecasting - S3/SageMaker Production Pipeline
===================================================================

Process 1979-2021 IBTrACS + ERA5 data from S3.
Optimized for SageMaker with progress tracking and checkpointing.

Author: Adapted for production use
"""

import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
from pathlib import Path
import logging
import time
import boto3
from botocore.exceptions import ClientError
import tempfile
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# S3 Configuration
S3_BUCKET = 'storm-track-project-data'
S3_PREFIX = 'data/era5'

# Local paths (SageMaker)
IBTRACS_FILE = "IBTrACS_fore72.txt"  # Assumed to be local or download first
OUTPUT_DIR = Path("processed_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Processing parameters
START_YEAR = 1979
END_YEAR = 2021
PRESSURE_LEVELS = [300, 500, 700, 850]

# Spatial grid parameters
WIND_SST_RADIUS = 10  # degrees
GEOPOTENTIAL_BOUNDS = {
    'lat_min_offset': -10,
    'lat_max_offset': 35,
    'lon_min_offset': -20,
    'lon_max_offset': 20
}

# Temporal parameters
INPUT_HOURS = 24
FORECAST_HOURS = [6, 12, 24, 48, 72]
TIME_INTERVAL = 6

# Auto-detected time dimension
TIME_DIM = None

# Initialize S3 client
s3_client = boto3.client('s3')

# ============================================================================
# S3 UTILITIES
# ============================================================================

def download_from_s3(s3_key, local_path):
    """Download file from S3 to local path."""
    try:
        s3_client.download_file(S3_BUCKET, s3_key, local_path)
        return True
    except ClientError as e:
        logger.error(f"Failed to download {s3_key}: {e}")
        return False


def upload_to_s3(local_path, s3_key):
    """Upload file from local path to S3."""
    try:
        s3_client.upload_file(local_path, S3_BUCKET, s3_key)
        logger.info(f"Uploaded to s3://{S3_BUCKET}/{s3_key}")
        return True
    except ClientError as e:
        logger.error(f"Failed to upload {s3_key}: {e}")
        return False


def check_processed(year):
    """Check if year has already been processed."""
    s3_key = f"processed_data/processed_samples_{year}.pkl"
    try:
        s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
        return True
    except ClientError:
        return False


# ============================================================================
# DATA LOADING
# ============================================================================

def detect_time_dimension(dataset):
    """Auto-detect time dimension name."""
    possible_names = ['time', 'valid_time', 'forecast_time', 't']
    
    for dim_name in possible_names:
        if dim_name in dataset.dims:
            return dim_name
    
    for coord_name in dataset.coords:
        if 'time' in coord_name.lower():
            return coord_name
    
    raise ValueError(f"No time dimension found. Dims: {list(dataset.dims)}")


def load_era5_from_s3(year, temp_dir):
    """
    Download and load ERA5 data for a given year from S3.
    
    Downloads to temp directory, loads into xarray, then cleans up.
    """
    global TIME_DIM
    
    logger.info(f"Loading ERA5 data for {year} from S3...")
    
    datasets = {}
    
    # Download and load pressure data
    logger.info("  Downloading pressure data...")
    p_files = []
    for i in range(3):
        s3_key = f"{S3_PREFIX}/era5_p/era5_p_{year}_{i}.nc"
        local_path = os.path.join(temp_dir, f"era5_p_{year}_{i}.nc")
        
        if download_from_s3(s3_key, local_path):
            p_files.append(xr.open_dataset(local_path))
        else:
            raise RuntimeError(f"Failed to download {s3_key}")
    
    if TIME_DIM is None:
        TIME_DIM = detect_time_dimension(p_files[0])
        logger.info(f"  Detected time dimension: '{TIME_DIM}'")
    
    datasets['p'] = xr.concat(p_files, dim=TIME_DIM)
    logger.info(f"  ✅ Pressure: {len(datasets['p'][TIME_DIM])} timesteps")
    
    # Download and load U-wind data
    logger.info("  Downloading U-wind data...")
    u_files = []
    for i in range(3):
        s3_key = f"{S3_PREFIX}/era5_u/era5_u_{year}_{i}.nc"
        local_path = os.path.join(temp_dir, f"era5_u_{year}_{i}.nc")
        
        if download_from_s3(s3_key, local_path):
            u_files.append(xr.open_dataset(local_path))
        else:
            raise RuntimeError(f"Failed to download {s3_key}")
    
    datasets['u'] = xr.concat(u_files, dim=TIME_DIM)
    logger.info(f"  ✅ U-wind: {len(datasets['u'][TIME_DIM])} timesteps")
    
    # Download and load V-wind data
    logger.info("  Downloading V-wind data...")
    v_files = []
    for i in range(3):
        s3_key = f"{S3_PREFIX}/era5_v/era5_v_{year}_{i}.nc"
        local_path = os.path.join(temp_dir, f"era5_v_{year}_{i}.nc")
        
        if download_from_s3(s3_key, local_path):
            v_files.append(xr.open_dataset(local_path))
        else:
            raise RuntimeError(f"Failed to download {s3_key}")
    
    datasets['v'] = xr.concat(v_files, dim=TIME_DIM)
    logger.info(f"  ✅ V-wind: {len(datasets['v'][TIME_DIM])} timesteps")
    
    # Download and load SST data
    logger.info("  Downloading SST data...")
    s3_key = f"{S3_PREFIX}/era5_sst/era5_sst_{year}.nc"
    local_path = os.path.join(temp_dir, f"era5_sst_{year}.nc")
    
    if download_from_s3(s3_key, local_path):
        datasets['sst'] = xr.open_dataset(local_path)
        logger.info(f"  ✅ SST: {len(datasets['sst'][TIME_DIM])} timesteps")
    else:
        raise RuntimeError(f"Failed to download {s3_key}")
    
    return datasets


# ============================================================================
# ENVIRONMENTAL DATA EXTRACTION (NO DEVORTEXING)
# ============================================================================

def extract_environmental_data(lat, lon, timestamp, era5_data):
    """
    Extract environmental fields around a cyclone position.
    
    SIMPLIFIED VERSION: No devortexing (for speed during testing).
    You can add devortexing back later if needed.
    """
    
    # Select nearest time
    u_t = era5_data['u'].sel({TIME_DIM: timestamp}, method='nearest')
    v_t = era5_data['v'].sel({TIME_DIM: timestamp}, method='nearest')
    p_t = era5_data['p'].sel({TIME_DIM: timestamp}, method='nearest')
    sst_t = era5_data['sst'].sel({TIME_DIM: timestamp}, method='nearest')
    
    # Define spatial bounds
    lat_min = lat - WIND_SST_RADIUS
    lat_max = lat + WIND_SST_RADIUS
    lon_min = lon - WIND_SST_RADIUS
    lon_max = lon + WIND_SST_RADIUS
    
    # Extract wind at each pressure level
    wind_fields = {}
    
    for level in PRESSURE_LEVELS:
        try:
            u_box = u_t.sel(
                pressure_level=level,
                latitude=slice(lat_max, lat_min),
                longitude=slice(lon_min, lon_max)
            )
            
            v_box = v_t.sel(
                pressure_level=level,
                latitude=slice(lat_max, lat_min),
                longitude=slice(lon_min, lon_max)
            )
            
            # Just reverse latitude, no devortexing
            wind_fields[f'u_{level}'] = u_box.u.values[::-1, :]
            wind_fields[f'v_{level}'] = v_box.v.values[::-1, :]
            
        except Exception as e:
            logger.warning(f"Could not extract wind at {level} hPa: {e}")
            return None
    
    # Extract SST
    try:
        sst_box = sst_t.sel(
            latitude=slice(lat_max, lat_min),
            longitude=slice(lon_min, lon_max)
        )
        sst_field = sst_box.sst.values[::-1, :]
    except Exception as e:
        logger.warning(f"Could not extract SST: {e}")
        return None
    
    # Extract geopotential
    try:
        lat_min_p = lat + GEOPOTENTIAL_BOUNDS['lat_min_offset']
        lat_max_p = min(lat + GEOPOTENTIAL_BOUNDS['lat_max_offset'], 89)
        lon_min_p = lon + GEOPOTENTIAL_BOUNDS['lon_min_offset']
        lon_max_p = lon + GEOPOTENTIAL_BOUNDS['lon_max_offset']
        
        if lat_max_p >= 90:
            lat_max_p = 89
            lat_min_p = 44
        
        p_box = p_t.sel(
            pressure_level=500,
            latitude=slice(lat_max_p, lat_min_p),
            longitude=slice(lon_min_p, lon_max_p)
        )
        p_field = p_box.z.values[::-1, :]
    except Exception as e:
        logger.warning(f"Could not extract geopotential: {e}")
        return None
    
    return {
        'wind': wind_fields,
        'sst': sst_field,
        'geopotential': p_field,
        'actual_time': timestamp,
        'position': (lat, lon)
    }


def create_training_sample(cyclone_data, start_idx, era5_data):
    """Create one training sample using sliding window."""
    
    n_input = INPUT_HOURS // TIME_INTERVAL
    
    input_traj = cyclone_data.iloc[start_idx:start_idx + n_input].copy()
    
    if len(input_traj) < n_input:
        return None
    
    current_time = input_traj['datetime'].iloc[-1]
    current_lat = input_traj['lat'].iloc[-1]
    current_lon = input_traj['lon'].iloc[-1]
    
    # Extract environmental data at each input timestep
    environmental_data = []
    
    # for idx, row in input_traj.iterrows():
    #     env = extract_environmental_data(
    #         row['lat'], row['lon'], row['datetime'], era5_data
    #     )
    #     if env is None:
    #         return None
    #     environmental_data.append(env)
    for idx, row in input_traj.iterrows():
        try:
            lt = round(row['lat'])
            if lt < 10:
                continue
            ln = round(row['lon'])
            if ln < 110 or ln > 160:
                continue
            dt = row['datetime']
            env = extract_environmental_data(lt, ln, dt, era5_data)
            environmental_data.append(env)
        except Exception as e:
            print(f"  Warning: Could not extract environment at {row['datetime']}: {e}")
            return None
    # Find target positions
    targets = {}
    for fh in FORECAST_HOURS:
        target_time = current_time + timedelta(hours=fh)
        time_diff = abs(cyclone_data['datetime'] - target_time)
        closest_idx = time_diff.idxmin()
        
        if time_diff[closest_idx].total_seconds() / 3600 <= 1.5:
            target_row = cyclone_data.loc[closest_idx]
            targets[f't+{fh}h'] = {
                'lat': target_row['lat'],
                'lon': target_row['lon'],
                'ws': target_row['ws'],
                'p': target_row['p']
            }
        else:
            targets[f't+{fh}h'] = None
    
    return {
        'input_trajectory': input_traj[['lat', 'lon', 'ws', 'p', 'speed', 'direct']].values,
        'environmental_data': environmental_data,
        'current_position': (current_lat, current_lon),
        'current_time': current_time,
        'targets': targets,
        'cyclone_name': input_traj['name'].iloc[0]
    }


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_single_year(year, ibtracs_df):
    """Process all cyclones for a single year."""
    
    logger.info("="*70)
    logger.info(f"  PROCESSING YEAR {year}")
    logger.info("="*70)
    
    # Check if already processed
    if check_processed(year):
        logger.info(f"  ⏭️  Year {year} already processed, skipping...")
        return
    
    # Filter IBTrACS for this year
    df_year = ibtracs_df[ibtracs_df['year'] == year].copy()
    
    if len(df_year) == 0:
        logger.info(f"  ⚠️  No cyclones in {year}, skipping...")
        return
    
    logger.info(f"  Found {df_year['name'].nunique()} cyclones in {year}")
    logger.info(f"  Total observations: {len(df_year)}")
    
    # Load ERA5 data for this year
    with tempfile.TemporaryDirectory() as temp_dir:
        start_time = time.time()
        
        try:
            era5_data = load_era5_from_s3(year, temp_dir)
        except Exception as e:
            logger.error(f"  ❌ Failed to load ERA5 for {year}: {e}")
            return
        
        load_time = time.time() - start_time
        logger.info(f"  Data loading took {load_time:.1f} seconds")
        
        # Process each cyclone
        logger.info(f"  Processing cyclones...")
        
        all_samples = []
        cyclone_names = df_year['name'].unique()
        
        for cyclone_idx, cyclone_name in enumerate(cyclone_names):
            cyclone_track = df_year[df_year['name'] == cyclone_name].copy()
            cyclone_track = cyclone_track.sort_values('datetime').reset_index(drop=True)
            
            n_obs = len(cyclone_track)
            n_input = INPUT_HOURS // TIME_INTERVAL
            max_forecast = max(FORECAST_HOURS) // TIME_INTERVAL
            
            if n_obs < (n_input + max_forecast):
                logger.info(f"    [{cyclone_idx+1}/{len(cyclone_names)}] ⏭️  {cyclone_name} (only {n_obs} obs)")
                continue
            
            logger.info(f"    [{cyclone_idx+1}/{len(cyclone_names)}] 🌀 {cyclone_name} ({n_obs} obs)")
            
            # Create samples
            cyclone_samples = []
            max_start = n_obs - n_input - max_forecast + 1
            
            for start_idx in range(max_start):
                sample = create_training_sample(cyclone_track, start_idx, era5_data)
                if sample is not None:
                    cyclone_samples.append(sample)
            
            logger.info(f"       → {len(cyclone_samples)} samples")
            all_samples.extend(cyclone_samples)
        
        logger.info(f"  ✅ Total samples for {year}: {len(all_samples)}")
        
        # Save to local file
        local_output = OUTPUT_DIR / f"processed_samples_{year}.pkl"
        with open(local_output, 'wb') as f:
            pickle.dump(all_samples, f)
        
        logger.info(f"  💾 Saved locally: {local_output}")
        
        # Upload to S3
        s3_key = f"processed_data/processed_samples_{year}.pkl"
        upload_to_s3(str(local_output), s3_key)
        
        total_time = time.time() - start_time
        logger.info(f"  ⏱️  Year {year} completed in {total_time/60:.1f} minutes")


def process_all_years():
    """Main entry point: process all years."""
    
    logger.info("="*70)
    logger.info("  MULTI-YEAR TC PROCESSING PIPELINE")
    logger.info("="*70)
    logger.info(f"  Years: {START_YEAR}-{END_YEAR}")
    logger.info(f"  S3 Bucket: {S3_BUCKET}")
    logger.info("="*70)
    
    # Load IBTrACS data
    logger.info("Loading IBTrACS trajectory data...")
    df = pd.read_csv(IBTRACS_FILE, header=None,
                     names=['name','date','lat','lon','ws','p','speed','direct'])
    
    df_clean = df[df['name'] != '66666'].copy()
    df_clean['datetime'] = pd.to_datetime(df_clean['date'])
    df_clean['year'] = df_clean['datetime'].dt.year
    
    logger.info(f"  Total observations: {len(df_clean)}")
    logger.info(f"  Years available: {df_clean['year'].min()}-{df_clean['year'].max()}")
    
    # Process each year
    overall_start = time.time()
    years_to_process = range(START_YEAR, END_YEAR + 1)
    
    for year_idx, year in enumerate(years_to_process):
        logger.info(f"\n{'='*70}")
        logger.info(f"  YEAR {year} [{year_idx+1}/{len(years_to_process)}]")
        logger.info(f"{'='*70}\n")
        
        try:
            process_single_year(year, df_clean)
        except Exception as e:
            logger.error(f"  ❌ Failed to process {year}: {e}")
            import traceback
            traceback.print_exc()
            logger.info(f"  Continuing to next year...")
            continue
    
    total_time = time.time() - overall_start
    logger.info("\n" + "="*70)
    logger.info("  ALL YEARS COMPLETED!")
    logger.info("="*70)
    logger.info(f"  Total time: {total_time/3600:.2f} hours")
    logger.info(f"  Average per year: {total_time/len(years_to_process)/60:.1f} minutes")


if __name__ == "__main__":
    process_all_years()
