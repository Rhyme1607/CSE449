import numpy as np
from PIL import Image
import time
import multiprocessing as mp
import os
from numba import cuda
import math

def cpu_gaussian_blur(image_array, kernel):
    height, width, channels = image_array.shape
    k_height, k_width = kernel.shape
    output_array = np.zeros_like(image_array)
    pad_h, pad_w = k_height // 2, k_width // 2
    padded_image = np.pad(image_array, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='edge')
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                region = padded_image[y:y + k_height, x:x + k_width, c]
                output_array[y, x, c] = np.sum(region * kernel)
    return output_array.astype(np.uint8)

def blur_chunk(image_chunk, kernel):
    height, width, channels = image_chunk.shape
    k_height, k_width = kernel.shape
    output_chunk = np.zeros_like(image_chunk)
    pad_h, pad_w = k_height // 2, k_width // 2
    padded_chunk = np.pad(image_chunk, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='edge')
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                region = padded_chunk[y:y + k_height, x:x + k_width, c]
                output_chunk[y, x, c] = np.sum(region * kernel)
    return output_chunk

def cpu_multicore_gaussian_blur(image_array, kernel, num_processes):
    chunks = np.array_split(image_array, num_processes, axis=0)
    process_args = [(chunk, kernel) for chunk in chunks]
    with mp.Pool(processes=num_processes) as pool:
        processed_chunks = pool.starmap(blur_chunk, process_args)
    return np.vstack(processed_chunks).astype(np.uint8)

@cuda.jit
def gpu_gaussian_blur_kernel(input_image, output_image, kernel):
    y, x = cuda.grid(2)
    img_height, img_width, channels = input_image.shape
    k_height, k_width = kernel.shape
    pad_h, pad_w = k_height // 2, k_width // 2
    if y < img_height and x < img_width:
        for c in range(channels):
            pixel_sum = 0.0
            for ky in range(-pad_h, pad_h + 1):
                for kx in range(-pad_w, pad_w + 1):
                    ny, nx = y + ky, x + kx
                    if 0 <= ny < img_height and 0 <= nx < img_width:
                        pixel_sum += input_image[ny, nx, c] * kernel[ky + pad_h, kx + pad_w]
            output_image[y, x, c] = pixel_sum

if __name__ == '__main__':
    resolutions = {
        "SD (480p)": (640, 480), "HD (720p)": (1280, 720),
        "FHD (1080p)": (1920, 1080), "QHD (1440p)": (2560, 1440)
    }
    results = {}
    
    print("Loading base image...")
    original_image = Image.open('test_image.jpg')
    gaussian_kernel_5x5 = (1/256) * np.array([
        [1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]
    ])
    num_cores = os.cpu_count()

    print("\n--- Running all CPU tests ---")
    for name, res in resolutions.items():
        print(f"\n--- Testing CPU for Resolution: {name} {res} ---")
        width, height = res
        resized_image = original_image.resize((width, height))
        input_image_array = np.array(resized_image)
        
        print(f"Running Single-Core CPU test...")
        start_time = time.time()
        cpu_gaussian_blur(input_image_array, gaussian_kernel_5x5)
        cpu_single_duration = time.time() - start_time
        
        print(f"Running Multi-Core CPU test ({num_cores} cores)...")
        start_time = time.time()
        cpu_multicore_gaussian_blur(input_image_array, gaussian_kernel_5x5, num_cores)
        cpu_multi_duration = time.time() - start_time
        
        results[name] = {
            'resolution': f"{width}x{height}", 'pixels': width * height,
            'single_cpu': cpu_single_duration, 'multi_cpu': cpu_multi_duration
        }

    print("\n--- Running all GPU tests ---")
    for name, res in resolutions.items():
        print(f"\n--- Testing GPU for Resolution: {name} {res} ---")
        width, height = res
        resized_image = original_image.resize((width, height))
        input_image_array = np.array(resized_image)
        
        d_input_image = cuda.to_device(input_image_array)
        d_kernel = cuda.to_device(gaussian_kernel_5x5)
        d_output_image = cuda.device_array_like(d_input_image)
        threads_per_block = (16, 16)
        blocks_per_grid_x = math.ceil(width / threads_per_block[0])
        blocks_per_grid_y = math.ceil(height / threads_per_block[1])
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        
        start_time = time.time()
        gpu_gaussian_blur_kernel[blocks_per_grid, threads_per_block](d_input_image, d_output_image, d_kernel)
        cuda.synchronize()
        gpu_duration = time.time() - start_time
        results[name]['gpu'] = gpu_duration

    # --- PART 3: Print Final Summary Table ---
    print("\n\n--- FINAL SCALABILITY ANALYSIS RESULTS ---")
    print("-" * 80)
    print(f"{'Test Name':<15} | {'Resolution':<15} | {'Pixels':<12} | {'Single CPU (s)':<16} | {'Multi CPU (s)':<15} | {'GPU (s)':<10}")
    print("-" * 80)
    for name, data in results.items():
        print(f"{name:<15} | {data['resolution']:<15} | {data['pixels']:<12,} | {data['single_cpu']:<16.4f} | {data['multi_cpu']:<15.4f} | {data['gpu']:<10.4f}")
    print("-" * 80)