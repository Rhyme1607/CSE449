import numpy as np
from PIL import Image
from numba import cuda
import math
import time 

def cpu_grayscale(image_array):
    height, width, _ = image_array.shape
    gray_array = np.empty((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            r, g, b = image_array[y, x]
            gray_value = 0.21 * r + 0.71 * g + 0.07 * b
            gray_array[y, x] = int(gray_value)
    return gray_array

@cuda.jit
def gpu_grayscale_kernel(input_image, output_image):
    x, y = cuda.grid(2)
    if y < output_image.shape[0] and x < output_image.shape[1]:
        r, g, b = input_image[y, x]
        gray_value = 0.21 * r + 0.71 * g + 0.07 * b
        output_image[y, x] = int(gray_value)

print("Loading image...")
original_image = Image.open('test_image.jpg')

print("Resizing image to a manageable size (1920x1080)...")
resized_image = original_image.resize((1920, 1080))
input_image_array = np.array(resized_image)
print("-" * 30)

print("Processing on CPU...")
start_cpu_time = time.time()
processed_cpu_array = cpu_grayscale(input_image_array) 
end_cpu_time = time.time() 
cpu_duration = end_cpu_time - start_cpu_time 
print(f"CPU Execution Time: {cpu_duration:.4f} seconds")

output_cpu_image = Image.fromarray(processed_cpu_array)
output_cpu_image.save('grayscale_cpu.jpg')
print("-" * 30)

print("Processing on GPU...")
d_input_image = cuda.to_device(input_image_array)
height, width, _ = input_image_array.shape
d_output_image = cuda.device_array((height, width), dtype=np.uint8)

threads_per_block = (16, 16)
blocks_per_grid_x = math.ceil(width / threads_per_block[0])
blocks_per_grid_y = math.ceil(height / threads_per_block[1])
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

start_gpu_time = time.time()
gpu_grayscale_kernel[blocks_per_grid, threads_per_block](d_input_image, d_output_image) 
cuda.synchronize() 
end_gpu_time = time.time() 
gpu_duration = end_gpu_time - start_gpu_time 
print(f"GPU Execution Time: {gpu_duration:.4f} seconds")

processed_gpu_array = d_output_image.copy_to_host()
output_gpu_image = Image.fromarray(processed_gpu_array)
output_gpu_image.save('grayscale_gpu.jpg')
print("-" * 30)

speedup = cpu_duration / gpu_duration
print(f"GPU was {speedup:.2f} times faster than the CPU.")