from numba import cuda
print(f"Is the GPU available? --> {cuda.is_available()}")