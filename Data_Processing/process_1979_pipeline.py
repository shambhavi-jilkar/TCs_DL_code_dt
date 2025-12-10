"""
1979 TC Track Forecasting - Complete Data Processing Pipeline
==============================================================

This script processes ONE YEAR (1979) of tropical cyclone data to create
training samples for a Diffusion Transformer model.

Steps:
1. Load IBTrACS cyclone trajectories
2. Load ERA5 environmental data
3. Extract spatial grids around each cyclone position
4. Apply devortexing to remove cyclone's own circulation
5. Create sliding window samples
6. Save in format ready for model training

Author: Adapted from TCs_DL_code repository
Purpose: Test run before scaling to full dataset
"""

import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
from pathlib import Path
import metpy.calc as mpcalc
from metpy.units import units

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths - UPDATE THESE TO MATCH YOUR SETUP
IBTRACS_FILE = "IBTrACS_fore72.txt"
ERA5_DIR = "."  # Folder containing your ERA5 files

OUTPUT_DIR = Path("processed_1979")
OUTPUT_DIR.mkdir(exist_ok=True)

# Spatial grid parameters (from the paper)
WIND_SST_RADIUS = 10  # degrees (±10° → 21×21 grid at 1° resolution)
GEOPOTENTIAL_BOUNDS = {
    'lat_min_offset': -10,
    'lat_max_offset': 35,
    'lon_min_offset': -20,
    'lon_max_offset': 20
}

PRESSURE_LEVELS = [300, 500, 700, 850]  # hPa

# Temporal parameters
INPUT_HOURS = 24  # Past 24 hours as input
FORECAST_HOURS = [6, 12, 24, 48, 72]  # Forecast lead times
TIME_INTERVAL = 3  # Hours between observations

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_era5_data(year=1979):
    """
    Load and combine ERA5 data for a given year.
    Handles the split files (0, 1, 2) for p, u, v.
    """
    print(f"Loading ERA5 data for {year}...")
    
    # Load pressure files and concatenate
    p_files = [
        xr.open_dataset(f"{ERA5_DIR}/era5_p_{year}_0.nc"),
        xr.open_dataset(f"{ERA5_DIR}/era5_p_{year}_1.nc"),
        xr.open_dataset(f"{ERA5_DIR}/era5_p_{year}_2.nc")
    ]
    p_data = xr.concat(p_files, dim='valid_time')
    print(f"  Pressure: {len(p_data.valid_time)} timesteps")
    
    # Load u-wind files and concatenate
    u_files = [
        xr.open_dataset(f"{ERA5_DIR}/era5_u_{year}_0.nc"),
        xr.open_dataset(f"{ERA5_DIR}/era5_u_{year}_1.nc"),
        xr.open_dataset(f"{ERA5_DIR}/era5_u_{year}_2.nc")
    ]
    u_data = xr.concat(u_files, dim='valid_time')
    print(f"  U-wind: {len(u_data.valid_time)} timesteps")
    
    # Load v-wind files and concatenate
    v_files = [
        xr.open_dataset(f"{ERA5_DIR}/era5_v_{year}_0.nc"),
        xr.open_dataset(f"{ERA5_DIR}/era5_v_{year}_1.nc"),
        xr.open_dataset(f"{ERA5_DIR}/era5_v_{year}_2.nc")
    ]
    v_data = xr.concat(v_files, dim='valid_time')
    print(f"  V-wind: {len(v_data.valid_time)} timesteps")
    
    # Load SST (single file)
    sst_data = xr.open_dataset(f"{ERA5_DIR}/era5_sst_{year}.nc")
    print(f"  SST: {len(sst_data.valid_time)} timesteps")
    
    print("✅ ERA5 data loaded successfully!\n")
    
    return {
        'u': u_data,
        'v': v_data,
        'p': p_data,
        'sst': sst_data
    }

def devortex_wind(longitude, latitude, u, v):
    """Simplified: just return the raw wind for now"""
    u_plain = np.asarray(u.magnitude if hasattr(u, 'magnitude') else u)
    v_plain = np.asarray(v.magnitude if hasattr(v, 'magnitude') else v)
    return u_plain, v_plain

# def devortex_wind(longitude, latitude, u, v):
#     """
#     Remove cyclone's own circulation from wind field.
    
#     This is CRITICAL for track forecasting because we need the environmental
#     steering flow, not the cyclone's own winds.
    
#     Physics:
#     - Any wind field can be decomposed into rotational + divergent components
#     - Rotational part (vorticity): captures cyclone's circular motion
#     - Divergent part: captures outflow/inflow
#     - Environmental wind = Total - Rotational - Divergent
    
#     Returns:
#     --------
#     u_env, v_env : Environmental (steering) wind components
#     """
#     # u = u * units('m/s')
#     # v = v * units('m/s')
#     # Parameters for iterative solver
#     if not hasattr(u, 'units'):
#         u = u * units('m/s')
#         v = v * units('m/s')

#     MAX_ITER = 1000
#     EPSILON = 1e-5  # Convergence criterion
#     SOR_INDEX = 0.2  # Successive Over-Relaxation parameter
    
#     M, N = len(latitude), len(longitude)
    
#     # Grid spacing
#     dx, dy = mpcalc.lat_lon_grid_deltas(longitude, latitude)
#     print(" Computing divergent component...")
#     # dxx = np.array(dx)
#     # dyy = np.array(dy)

#     # divh = mpcalc.divergence(u, v, dx=dx, dy=dy)
#     # divh = np.array(divh)
    
#     # ========================================================================
#     # STEP 1: Remove divergent component (irrotational flow)
#     # ========================================================================
    
#     # Calculate divergence
#     divh = mpcalc.divergence(u, v, dx=dx, dy=dy)
#     #divh = np.asarray(divh.magnitude)
#     divh = np.array(divh)
#     print(f"DEBUG: divh shape={divh.shape}, sample values={divh[10,10]}, min={divh.min()}, max={divh.max()}")
#     # dxx=np.asarray(dx.magnitude)
#     # dyy=np.asarray(dy.magnitude)
#     dxx = np.array(dx)
#     dyy = np.array(dy)
#     # print(f"DEBUG: dxx shape={dxx.shape}, sample values={dxx[10,10]}, {dxx[10,11]}")
#     # print(f"DEBUG: dyy shape={dyy.shape}, sample values={dyy[10,10]}, {dyy[11,10]}")
#     # print(f"DEBUG: Checking boundary access...")
#     # try:
#     #     test_val = dxx[18, 18]
#     #     print(f"  dxx[18, 18] = {test_val}")
#     # except IndexError as e:
#     #     print(f"  ERROR accessing dxx[18, 18]: {e}")
#     # Solve Poisson equation: ∇²χ = div
#     chi = np.zeros((M, N))  # Velocity potential
#     #Res = np.ones((M, N)) * (-9999)
#     Res = np.zeros((M, N))
#     # print(f"DEBUG: M={M}, N={N}, i_max={M-2}, j_max={N-2}")
#     # print(f"DEBUG: Will access dxx[{M-2}, {N-2}] but dxx.shape={dxx.shape}")
#     for k in range(MAX_ITER):
#         #print(f"  Iteration {k}...")
#         for i in range(1, M-1):
#             for j in range(1, N-1):
#                 # Finite difference approximation
#                 Res[i,j] = (chi[i+1,j] + chi[i-1,j] - 2*chi[i,j]) / (dxx[i,j-1]*dxx[i,j]) + \
#                            (chi[i,j+1] + chi[i,j-1] - 2*chi[i,j]) / (dyy[i-1,j]*dyy[i,j]) + \
#                            divh[i,j]
#                 # if k == 0 and i == 10 and j == 10:
#                 #     term1 = (chi[i+1,j] + chi[i-1,j] - 2*chi[i,j]) / (dxx[i,j-1]*dxx[i,j])
#                 #     term2 = (chi[i,j+1] + chi[i,j-1] - 2*chi[i,j]) / (dyy[i-1,j]*dyy[i,j])
#                 #     term3 = divh[i,j]
#                 #     denom = 2/(dxx[i,j-1]*dxx[i,j]) + 2/(dyy[i-1,j]*dyy[i,j])
#                 #     print(f"  At i=10,j=10:")
#                 #     print(f"    term1={term1}, term2={term2}, term3={term3}")
#                 #     print(f"    Res[10,10]={Res[i,j]}")
#                 #     print(f"    denominator={denom}")
#                 #     print(f"    update amount={(1+SOR_INDEX)*Res[i,j]/denom}")
                
#                 chi[i,j] = chi[i,j] + (1+SOR_INDEX) * Res[i,j] / \
#                            (2/(dxx[i,j-1]*dxx[i,j]) + 2/(dyy[i-1,j]*dyy[i,j]))
#         # if k % 100 == 0:
#         #     max_res = np.max(np.abs(Res))
#         #     print(f"    Iteration {k}, max residual: {max_res:.2e}")
#         if np.max(np.abs(Res)) < EPSILON:
#             break
    
#     # Calculate divergent wind from velocity potential
#     chi = chi * units.meters * units.meters / units.seconds
#     grad_chi = mpcalc.gradient(chi, deltas=(dy, dx))
#     u_div = np.asarray(grad_chi[1].magnitude)
#     v_div = np.asarray(grad_chi[0].magnitude)
#     print(" Computing rotational component...")
#     # ========================================================================
#     # STEP 2: Remove rotational component (cyclone's vortex)
#     # ========================================================================
    
#     # Calculate vorticity (relative rotation)
#     vort = mpcalc.vorticity(u, v, dx=dx, dy=dy)
#     vort = np.asarray(vort.magnitude)
    
#     # Solve Poisson equation: ∇²ψ = vorticity
#     psi = np.zeros((M, N))  # Stream function
#     #Res = np.ones((M, N)) * (-9999)
#     Res = np.zeros((M, N))
    
#     for k in range(MAX_ITER):
#         #print(f"  Iteration {k}...")
#         for i in range(1, M-1):
#             for j in range(1, N-1):
#                 Res[i,j] = (psi[i+1,j] + psi[i-1,j] - 2*psi[i,j]) / (dxx[i,j-1]*dxx[i,j]) + \
#                            (psi[i,j+1] + psi[i,j-1] - 2*psi[i,j]) / (dyy[i-1,j]*dyy[i,j]) - \
#                            vort[i,j]
                
#                 psi[i,j] = psi[i,j] + (1+SOR_INDEX) * Res[i,j] / \
#                            (2/(dxx[i,j-1]*dxx[i,j]) + 2/(dyy[i-1,j]*dyy[i,j]))
#         # if k % 100 == 0:
#         #     max_res = np.max(np.abs(Res))
#         #     print(f"    Iteration {k}, max residual: {max_res:.2e}")
#         if np.max(np.abs(Res)) < EPSILON:
#             break
    
#     # Calculate rotational wind from stream function
#     psi = psi * units.meters * units.meters / units.seconds
#     grad_psi = mpcalc.gradient(psi, deltas=(dy, dx))
#     u_rot = np.asarray(-grad_psi[0].magnitude)
#     v_rot = np.asarray(grad_psi[1].magnitude)
    
#     # ========================================================================
#     # STEP 3: Get environmental wind (steering flow)
#     # ========================================================================
    
#     u_plain = np.asarray(u.magnitude) 
#     v_plain = np.asarray(v.magnitude) 
    
#     u_env = u_plain - u_div - u_rot
#     v_env = v_plain - v_div - v_rot

#     return u_env, v_env


def extract_environmental_data(lat, lon, timestamp, era5_data):
    """
    Extract environmental fields around a cyclone position.
    
    Parameters:
    -----------
    lat, lon : float
        Cyclone center position
    timestamp : datetime
        Time to extract data
    era5_data : dict
        Dictionary containing u, v, p, sst datasets
        
    Returns:
    --------
    dict with processed environmental fields
    """
    
    # Select nearest time in ERA5 data
    u_t = era5_data['u'].sel(valid_time=timestamp, method='nearest')
    v_t = era5_data['v'].sel(valid_time=timestamp, method='nearest')
    p_t = era5_data['p'].sel(valid_time=timestamp, method='nearest')
    sst_t = era5_data['sst'].sel(valid_time=timestamp, method='nearest')
    
    # Define spatial bounds for wind/SST (21×21 grid)
    lat_min = round(lat - WIND_SST_RADIUS)
    lat_max = round(lat + WIND_SST_RADIUS)
    lon_min = round(lon - WIND_SST_RADIUS)
    lon_max = round(lon + WIND_SST_RADIUS)

    if lat_min < 0:
        return None
    
    # Process wind fields at each pressure level
    wind_fields = {}
    
    for level in PRESSURE_LEVELS:
        # Extract wind at this level
        u_box = u_t.sel(
            pressure_level=level,
            latitude=slice(lat_max, lat_min),  # ERA5 lat is descending
            longitude=slice(lon_min, lon_max)
        )
        print(f'({lat_min}, {lon_min}), ({lat_max}, {lon_max})')
        print(u_box)
        
        v_box = v_t.sel(
            pressure_level=level,
            latitude=slice(lat_max, lat_min),
            longitude=slice(lon_min, lon_max)
        )
        u_values = u_box.u.values[::-1, :] * units('m/s') # Just reverse latitude for consistency
        v_values = v_box.v.values[::-1, :] * units('m/s')
        print(f"    Extracted shape: {u_values.shape}")
        # Apply devortexing
        u_env, v_env = devortex_wind(
            u_box.longitude.values,
            u_box.latitude.values[::-1],  # Reverse back for devortexing
            u_values,
            v_values
        )
        
        # Downsample to match paper's resolution (every 2nd point)
        wind_fields[f'u_{level}'] = u_env
        wind_fields[f'v_{level}'] = v_env
    
    # Extract SST
    sst_box = sst_t.sel(
        latitude=slice(lat_max, lat_min),
        longitude=slice(lon_min, lon_max)
    )
    sst_field = sst_box.sst.values[::-1, :] 
    print(f"    SST shape: {sst_field.shape}")
    # Extract geopotential (larger domain)
    lat_min_p = round(lat + GEOPOTENTIAL_BOUNDS['lat_min_offset'])
    lat_max_p = round(min(lat + GEOPOTENTIAL_BOUNDS['lat_max_offset'], 89))
    lon_min_p = round(lon + GEOPOTENTIAL_BOUNDS['lon_min_offset'])
    lon_max_p = round(lon + GEOPOTENTIAL_BOUNDS['lon_max_offset'])
    if lat_min_p <0:
        return None
    p_box = p_t.sel(
        pressure_level=500,  # Use 500 hPa geopotential
        latitude=slice(lat_max_p, lat_min_p),
        longitude=slice(lon_min_p, lon_max_p)
    )
    p_field = p_box.z.values[::-1, :]   
    print(f"    Geopotential shape: {p_field.shape}")
    
    return {\
        'wind': wind_fields,
        'sst': sst_field,
        'geopotential': p_field,
        'actual_time': u_t.valid_time.values,
        'position': (lat, lon)
    }


def create_training_sample(cyclone_data, start_idx, era5_data):
    """
    Create one training sample using sliding window approach.
    
    Parameters:
    -----------
    cyclone_data : pd.DataFrame
        Trajectory data for one cyclone
    start_idx : int
        Starting index for this sample
    era5_data : dict
        ERA5 datasets
        
    Returns:
    --------
    dict containing input features and target positions
    """
    
    n_input = INPUT_HOURS // TIME_INTERVAL  # 24h / 3h = 8 timesteps
    
    # Extract input trajectory (past 24 hours)
    input_traj = cyclone_data.iloc[start_idx:start_idx + n_input].copy()
    
    if len(input_traj) < n_input:
        return None  # Not enough data
    
    current_time = input_traj['datetime'].iloc[-1]
    current_lat = input_traj['lat'].iloc[-1]
    current_lon = input_traj['lon'].iloc[-1]
    
    # Extract environmental data at each input timestamp
    environmental_data = []
    
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
            print(f"  ⚠️  Warning: Could not extract environment at {row['datetime']}: {e}")
            return None
    
    # Find target positions (future)
    targets = {}
    for fh in FORECAST_HOURS:
        target_time = current_time + timedelta(hours=fh)
        
        # Find closest observation
        time_diff = abs(cyclone_data['datetime'] - target_time)
        closest_idx = time_diff.idxmin()
        
        if time_diff[closest_idx].total_seconds() / 3600 <= 1.5:
            target_row = cyclone_data.loc[closest_idx]
            targets[f't+{fh}h'] = {
                'lat': target_row['lat'],
                'lon': target_row['lon']
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
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_1979_data():
    """
    Main function to process all 1979 cyclones.
    """
    
    print("="*70)
    print("  1979 TROPICAL CYCLONE DATA PROCESSING PIPELINE")
    print("="*70)
    print()
    
    # Step 1: Load IBTrACS data
    print("STEP 1: Loading IBTrACS trajectory data...")
    df = pd.read_csv(IBTRACS_FILE, header=None,
                     names=['name','date','lat','lon','ws','p','speed','direct'])
    
    # Remove separator rows and filter for 1979
    df_clean = df[df['name'] != '66666'].copy()
    df_clean['datetime'] = pd.to_datetime(df_clean['date'])
    df_clean['year'] = df_clean['datetime'].dt.year
    df_1979 = df_clean[df_clean['year'] == 1979].copy()
    
    print(f"  Found {df_1979['name'].nunique()} cyclones in 1979")
    print(f"  Total observations: {len(df_1979)}")
    print()
    
    # Step 2: Load ERA5 data
    print("STEP 2: Loading ERA5 environmental data...")
    era5_data = load_era5_data(year=1979)
    
    # Step 3: Process each cyclone
    print("STEP 3: Processing cyclones and creating training samples...")
    print()
    
    all_samples = []
    
    for cyclone_name in df_1979['name'].unique():
        cyclone_track = df_1979[df_1979['name'] == cyclone_name].copy()
        cyclone_track = cyclone_track.sort_values('datetime').reset_index(drop=True)
        
        n_obs = len(cyclone_track)
        n_input = INPUT_HOURS // TIME_INTERVAL
        max_forecast = max(FORECAST_HOURS) // TIME_INTERVAL
        
        # Check if cyclone is long enough
        if n_obs < (n_input + max_forecast):
            print(f"  ⏭️  Skipping {cyclone_name} (only {n_obs} obs, need {n_input + max_forecast})")
            continue
        
        print(f"  🌀 Processing {cyclone_name} ({n_obs} observations)")
        
        # Create samples using sliding window
        cyclone_samples = []
        for start_idx in range(n_obs - n_input - max_forecast + 1):
            sample = create_training_sample(cyclone_track, start_idx, era5_data)
            if sample is not None:
                cyclone_samples.append(sample)
        
        print(f"     → Created {len(cyclone_samples)} samples")
        all_samples.extend(cyclone_samples)
    
    print()
    print(f"✅ TOTAL SAMPLES CREATED: {len(all_samples)}")
    print()
    
    # Step 4: Save processed data
    print("STEP 4: Saving processed data...")
    
    output_file = OUTPUT_DIR / "processed_samples_1979.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(all_samples, f)
    
    print(f"  💾 Saved to: {output_file}")
    print()
    
    # Print sample statistics
    print("="*70)
    print("  PROCESSING COMPLETE!")
    print("="*70)
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Output file: {output_file}")
    print()
    print("  Next steps:")
    print("  1. Load the processed data")
    print("  2. Create features (19 trajectory features)")
    print("  3. Normalize all variables")
    print("  4. Train your Diffusion Transformer!")
    print("="*70)


if __name__ == "__main__":
    process_1979_data()
