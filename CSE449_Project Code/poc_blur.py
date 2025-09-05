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
    pad_h = k_height // 2
    pad_w = k_width // 2

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
    print("Loading image...")
    original_image = Image.open('test_image.jpg')
    print("Resizing image to 1920x1080 for all tests...")
    resized_image = original_image.resize((1920, 1080))
    input_image_array = np.array(resized_image)

    gaussian_kernel_5x5 = (1/256) * np.array([
        [1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]
    ])
    print("-" * 30)
    print("1. Processing Gaussian Blur on Single-Core CPU...")
    start_time = time.time()
    blurred_single_core = cpu_gaussian_blur(input_image_array, gaussian_kernel_5x5)
    cpu_single_duration = time.time() - start_time
    print(f"Single-Core CPU Execution Time: {cpu_single_duration:.4f} seconds")
    Image.fromarray(blurred_single_core).save('blur_cpu_single_core.jpg')

    print("-" * 30)
    num_cores = os.cpu_count()
    print(f"2. Processing Gaussian Blur on Multi-Core CPU ({num_cores} cores)...")
    start_time = time.time()
    blurred_multi_core = cpu_multicore_gaussian_blur(input_image_array, gaussian_kernel_5x5, num_cores)
    cpu_multi_duration = time.time() - start_time
    print(f"Multi-Core CPU Execution Time: {cpu_multi_duration:.4f} seconds")
    Image.fromarray(blurred_multi_core).save('blur_cpu_multi_core.jpg')

    print("-" * 30)
    print("3. Processing Gaussian Blur on GPU...")
    
    d_input_image = cuda.to_device(input_image_array)
    d_kernel = cuda.to_device(gaussian_kernel_5x5)
    d_output_image = cuda.device_array_like(d_input_image)
    
    threads_per_block = (16, 16)

    blocks_per_grid_x = math.ceil(input_image_array.shape[1] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(input_image_array.shape[0] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    start_time = time.time()
    gpu_gaussian_blur_kernel[blocks_per_grid, threads_per_block](d_input_image, d_output_image, d_kernel)
    cuda.synchronize()
    gpu_duration = time.time() - start_time
    print(f"GPU Execution Time: {gpu_duration:.4f} seconds")


    blurred_gpu = d_output_image.copy_to_host()
    Image.fromarray(blurred_gpu).save('blur_gpu.jpg')
    print("-" * 30)

    print("\n--- Proof of Concept Summary ---")
    print(f"Single-Core CPU time: {cpu_single_duration:.4f}s")
    print(f"Multi-Core CPU time:  {cpu_multi_duration:.4f}s")
    print(f"GPU time:             {gpu_duration:.4f}s")
    print("-" * 30)
    print(f"Multi-core was {cpu_single_duration/cpu_multi_duration:.2f}x faster than single-core.")
    print(f"GPU was {cpu_single_duration/gpu_duration:.2f}x faster than single-core.")
    print(f"GPU was {cpu_multi_duration/gpu_duration:.2f}x faster than multi-core.")
