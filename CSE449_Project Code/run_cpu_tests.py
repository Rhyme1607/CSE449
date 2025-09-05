import numpy as np
from PIL import Image
import time
import multiprocessing as mp
import os
import json

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

    with open('cpu_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\n\nCPU tests complete. Results saved to cpu_results.json")