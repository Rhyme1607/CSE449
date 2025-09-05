import numpy as np
from PIL import Image
import time
from numba import cuda
import math
import sys

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
    if len(sys.argv) != 3:
        print("Usage: python run_single_gpu_test.py <width> <height>")
        sys.exit(1)
    width = int(sys.argv[1])
    height = int(sys.argv[2])
    
    print(f"--- Testing GPU for Resolution: {width}x{height} ---")

    original_image = Image.open('test_image.jpg')
    gaussian_kernel_5x5 = (1/256) * np.array([
        [1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]
    ])
    
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
    
    print(f"GPU Execution Time for {width}x{height}: {gpu_duration:.4f} seconds")