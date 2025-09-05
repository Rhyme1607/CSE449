import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

pixels = np.array([307200, 921600, 2073600, 3686400])
single_cpu_times = np.array([5.3052, 15.7493, 36.5289, 68.9172])
multi_cpu_times = np.array([1.5501, 2.8213, 5.7986, 9.6404])
gpu_times = np.array([0.2811, 0.2834, 0.2772, 0.2832])

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 7))

plt.plot(pixels, single_cpu_times, marker='o', linestyle='-', label='Single-Core CPU')
plt.plot(pixels, multi_cpu_times, marker='s', linestyle='--', label='Multi-Core CPU (16 Threads)')
plt.plot(pixels, gpu_times, marker='^', linestyle=':', label='GPU (NVIDIA RTX 3070)')

plt.yscale('log')

plt.title('Performance Scalability of Gaussian Blur on Different Architectures', fontsize=16)
plt.xlabel('Number of Pixels in Image', fontsize=12)
plt.ylabel('Execution Time (seconds) - Logarithmic Scale', fontsize=12)
plt.legend(fontsize=11)

def format_pixels(x, pos):
    if x >= 1e6:
        return f'{x*1e-6:.1f}M'
    elif x >= 1e3:
        return f'{x*1e-3:.0f}K'
    return str(int(x))

plt.gca().xaxis.set_major_formatter(FuncFormatter(format_pixels))
plt.xticks(pixels)

plt.savefig('scalability_graph.png', dpi=300, bbox_inches='tight')

print("Successfully created and saved final 'scalability_graph.png'.")