import os
import time
import rasterio
from rasterio.transform import from_origin
import numpy as np
from noise import pnoise2, snoise2
from scipy.spatial import cKDTree
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configuration
input_file = "Data/1.tif"
output_dir = "Generated"

# Noise Parameters
perlin_scale = 0.005; perlin_freq = 0.002
simplex_scale = 0.005; simplex_freq = 0.002
voronoi_scale = 0.05; voronoi_freq = 0.002; voronoi_points = 5000
white_scale = 0.003

# Voronoi noise parameters
tile_size = 512  # Process the image in 512x512 tiles to save memory

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Open the input GeoTIFF file
with rasterio.open(input_file) as src:
    # Read the data, metadata, and transform
    data = src.read(1)
    meta = src.meta
    transform = meta['transform']

    # Handle "No Data" values and calculate range
    current_nodata = src.nodata if src.nodata is not None else -3.4028230607370965e+38
    valid_mask = data != current_nodata
    valid_data = np.where(valid_mask, data, np.nan)
    data_min, data_max = np.nanmin(valid_data), np.nanmax(valid_data)
    data_range = data_max - data_min

    print(f"Original data range: {data_min} to {data_max}")
    print(f"Setting new 'No Data' value to lowest elevation: {data_min}")

    # Set the lowest elevation as the new "No Data" value
    new_nodata = data_min
    data = np.where(valid_mask, data, new_nodata)

    # Write the fixed data
    meta.update({'nodata': new_nodata, 'dtype': rasterio.float32})
    with rasterio.open(os.path.join(output_dir, "fixed.tif"), 'w', **meta) as dst:
        dst.write(data.astype(rasterio.float32), 1)

    # Recalculate data range
    valid_data = np.where(data > new_nodata, data, np.nan)
    data_min, data_max = np.nanmin(valid_data), np.nanmax(valid_data)
    data_range = data_max - data_min
    print(f"Adjusted data range: {data_min} to {data_max}")

    # Calculate the pixel coordinates
    height, width = data.shape
    rows, cols = np.mgrid[0:height, 0:width]
    xs, ys = rasterio.transform.xy(transform, cols, rows, offset='center')
    xs, ys = np.array(xs), np.array(ys)

# Function to apply noise and save output
def apply_noise(noise_data, scale, additive=True):
    valid_mask = data > new_nodata
    scaled_noise = noise_data * scale * data_range
    if additive:
        with np.errstate(over='ignore', under='ignore'):
            noisy_data = np.where(valid_mask, data + scaled_noise, new_nodata)
    else:  # multiplicative
        noise_norm = (noise_data - np.min(noise_data)) / (np.max(noise_data) - np.min(noise_data))
        noise_mult = noise_norm * 0.6 + 0.7  # map to [0.7, 1.3]
        with np.errstate(over='ignore', under='ignore'):
            noisy_data = np.where(valid_mask, data * noise_mult, new_nodata)
    np.clip(noisy_data, data_min, data_max, out=noisy_data)
    return noisy_data.astype(rasterio.float32)

# Function to save output
def save_output(data, name):
    with rasterio.open(os.path.join(output_dir, f"{name}.tif"), 'w', **meta) as dst:
        dst.write(data, 1)

# Dictionary to store timing and statistics
stats = {}

# Generate and apply each type of noise
for noise_type, freq, scale in [
    ("perlin", perlin_freq, perlin_scale),
    ("simplex", simplex_freq, simplex_scale),
    ("white", None, white_scale)
]:
    print(f"Generating {noise_type.capitalize()} Noise...")
    start_time = time.time()
    
    if noise_type == "white":
        noise_data = np.random.normal(0, 1, data.shape).astype(np.float32)
    else:
        noise_func = snoise2 if noise_type == "simplex" else pnoise2
        noise_data = np.frompyfunc(lambda i, j: noise_func(i * freq, j * freq, octaves=6), 2, 1)(xs, ys).astype(np.float32)
    
    end_time = time.time()
    duration = end_time - start_time
    
    save_output(apply_noise(noise_data, scale, additive=True), f"{noise_type}_additive")
    save_output(apply_noise(noise_data, scale, additive=False), f"{noise_type}_multiplicative")
    
    stats[noise_type] = {
        'time': duration,
        'frequency': freq if freq is not None else 'N/A',
        'scale': scale,
        'min': np.min(noise_data),
        'max': np.max(noise_data),
        'mean': np.mean(noise_data),
        'std': np.std(noise_data)
    }

# Generate and apply Voronoi noise
print("Generating Voronoi Noise...")
start_time = time.time()
try:
    # Generate Voronoi points
    x_range = xs.max() - xs.min()
    y_range = ys.max() - ys.min()
    points = np.column_stack([
        np.random.rand(voronoi_points) * x_range + xs.min(),
        np.random.rand(voronoi_points) * y_range + ys.min()
    ]) * voronoi_freq

    # Build a KD-Tree for efficient nearest neighbor queries
    tree = cKDTree(points)

    # Process the image in tiles to save memory
    tile_rows = range(0, height, tile_size)
    tile_cols = range(0, width, tile_size)

    voronoi_data = np.zeros_like(data, dtype=np.float32)

    for i in tqdm(tile_rows, desc="Processing Voronoi Noise"):
        for j in tile_cols:
            # Define the current tile
            row_end = min(i + tile_size, height)
            col_end = min(j + tile_size, width)
            tile_xs = xs[i:row_end, j:col_end] * voronoi_freq
            tile_ys = ys[i:row_end, j:col_end] * voronoi_freq
            tile = np.column_stack([tile_xs.ravel(), tile_ys.ravel()])

            # Calculate distances to nearest Voronoi points
            distances, _ = tree.query(tile)
            voronoi_data[i:row_end, j:col_end] = distances.reshape(tile_xs.shape)

    # Normalize and invert Voronoi noise
    voronoi_data = 1 - (voronoi_data - np.min(voronoi_data)) / (np.max(voronoi_data) - np.min(voronoi_data))
    
    save_output(apply_noise(voronoi_data, voronoi_scale, additive=True), "voronoi_additive")
    save_output(apply_noise(voronoi_data, voronoi_scale, additive=False), "voronoi_multiplicative")

    end_time = time.time()
    duration = end_time - start_time
    
    stats['voronoi'] = {
        'time': duration,
        'frequency': voronoi_freq,
        'scale': voronoi_scale,
        'points': voronoi_points,
        'min': np.min(voronoi_data),
        'max': np.max(voronoi_data),
        'mean': np.mean(voronoi_data),
        'std': np.std(voronoi_data)
    }

except Exception as e:
    print(f"Error in Voronoi noise: {e}")

print("Done!")

# Plotting functions
def plot_time_taken():
    noise_types = list(stats.keys())
    times = [stats[nt]['time'] for nt in noise_types]
    
    plt.figure(figsize=(10, 5))
    plt.bar(noise_types, times)
    plt.title('Time Taken for Each Noise Type')
    plt.ylabel('Time (seconds)')
    plt.grid(axis='y', alpha=0.3)
    for i, v in enumerate(times):
        plt.text(i, v + 0.1, f'{v:.2f}s', ha='center')
    plt.savefig(os.path.join(output_dir, 'time_taken.png'))

def plot_frequency_scale():
    noise_types = [nt for nt in stats.keys() if 'frequency' in stats[nt] and stats[nt]['frequency'] != 'N/A']
    frequencies = [stats[nt]['frequency'] for nt in noise_types]
    scales = [stats[nt]['scale'] for nt in noise_types]
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    ax1.set_xlabel('Noise Type')
    ax1.set_ylabel('Frequency', color='tab:red')
    ax1.bar(noise_types, frequencies, color='tab:red', alpha=0.6, label='Frequency')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Scale', color='tab:blue')
    ax2.bar(noise_types, scales, color='tab:blue', alpha=0.6, label='Scale')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    fig.tight_layout()
    plt.title('Frequency and Scale for Each Noise Type')
    plt.savefig(os.path.join(output_dir, 'frequency_scale.png'))

def plot_noise_stats():
    noise_types = list(stats.keys())
    mins = [stats[nt]['min'] for nt in noise_types]
    maxs = [stats[nt]['max'] for nt in noise_types]
    means = [stats[nt]['mean'] for nt in noise_types]
    stds = [stats[nt]['std'] for nt in noise_types]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.boxplot([mins, maxs, means], labels=['Min', 'Max', 'Mean'])
    ax1.set_title('Min, Max, and Mean Values')
    ax1.grid(axis='y', alpha=0.3)
    
    ax2.bar(noise_types, stds)
    ax2.set_title('Standard Deviation')
    ax2.set_ylabel('Std')
    ax2.grid(axis='y', alpha=0.3)
    
    fig.suptitle('Statistics for Each Noise Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'noise_stats.png'))

# Call the plotting functions
plot_time_taken()
plot_frequency_scale()
plot_noise_stats()
