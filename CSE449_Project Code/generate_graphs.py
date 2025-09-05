import matplotlib.pyplot as plt
import numpy as np

cpu_time = 13.1576 
gpu_time = 0.4676  
speedup = cpu_time / gpu_time

methods = ['Single-Core CPU', 'NVIDIA RTX 3070 GPU']
times = [cpu_time, gpu_time]

plt.figure(figsize=(8, 6))
bars = plt.bar(methods, times, color=['#ff6347', '#4682b4'])

plt.ylabel('Execution Time (seconds)')
plt.title('Performance Comparison of Grayscale Filter (1920x1080 Image)')
plt.yscale('log')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}s', va='bottom', ha='center') 

plt.savefig('barchart_times.png', dpi=300, bbox_inches='tight')
print("Saved 'barchart_times.png'")

plt.figure(figsize=(6, 6))
plt.bar(['GPU Speedup'], [speedup], color=['#32cd32'])

plt.ylabel('Speedup Factor (X times faster)')
plt.title('GPU Speedup Over Single-Core CPU')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.text(0, speedup, f'{speedup:.2f}x', va='bottom', ha='center', fontsize=14)

plt.savefig('barchart_speedup.png', dpi=300, bbox_inches='tight')
print("Saved 'barchart_speedup.png'")