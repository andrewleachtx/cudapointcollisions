"""
    ChatGPT 4o wrote this initial code to enable quick visualization of formatless text files from cout statements
    with output redirected to a file based on a shell script (./run.sh) because it was a tedious task otherwise.

    I then was able to parse it and make modifications as necessary :) but I won't say I came up with the idea!
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn")

def parse_simulation_file(file_path):
    data = {}

    with open(file_path, 'r') as file:
        content = file.read()

        # Extract particles and threads per block from filename
        filename = os.path.basename(file_path)
        print(f"Processing file: {filename}")  # Debugging print

        match = re.match(r"(\d+)_particles_(\d+)_threads.*", filename)
        if match:
            data['particles'] = int(match.group(1))
            data['threads'] = int(match.group(2))
        else:
            print(f"Filename pattern didn't match for file: {filename}")
            data['particles'] = None
            data['threads'] = None

        # Extract key metrics from content
        try:
            data['program_time'] = float(re.search(r'Actual program time: ([\d.]+) ms', content).group(1))
        except AttributeError:
            print(f"Could not find program time in file: {filename}")
            data['program_time'] = None

        try:
            data['kernel_time'] = float(
                re.search(r'Overall kernel time before convergence: ([\d.]+) ms', content).group(1))
        except AttributeError:
            print(f"Could not find kernel time in file: {filename}")
            data['kernel_time'] = None

        try:
            data['blocks'] = int(re.search(r'blocks per grid: (\d+)', content).group(1))
        except AttributeError:
            print(f"Could not find blocks per grid in file: {filename}")
            data['blocks'] = None

        try:
            data['kernel_fraction'] = float(re.search(r'Kernel time / total program time: ([\d.]+)', content).group(1))
        except AttributeError:
            print(f"Could not find kernel fraction in file: {filename}")
            data['kernel_fraction'] = None

    return data

def process_all_files(directory):
    results = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                result = parse_simulation_file(file_path)
                results.append(result)

    return pd.DataFrame(results)

def plotKernelTimeVsParticles(df):
    # Sort data by the number of particles
    df = df.sort_values(by='particles')
    unique_threads = sorted(df['threads'].unique())

    markers = ['o', 's', '^', 'D', 'v', 'x', 'p', '*', 'h', '+']

    plt.figure(figsize=(10, 6))

    for i, thread in enumerate(unique_threads):
        # Filter data by thread count
        subset = df[df['threads'] == thread]

        particle_counts = subset['particles'].unique()
        x_values = range(len(particle_counts))
        marker = markers[i % len(markers)]

        plt.plot(x_values, subset['kernel_time'], marker=marker, linestyle='-', linewidth=2,
                 alpha=0.7, label=f'{thread} Threads')

    plt.xlabel('Number of Particles', fontsize=14)
    plt.ylabel('simulateKernel Time (ms)', fontsize=14)
    plt.title('Kernel Time vs Particles by Thread Count', fontsize=16, weight='bold')

    plt.xticks(x_values, particle_counts, rotation=45, fontsize=12)

    plt.yscale('log')

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Add a legend and ensure it is sorted by thread count
    plt.legend(title='Thread Count', fontsize=12, title_fontsize=14)

    plt.tight_layout()

    plt.savefig("figures/kernel_time_vs_particles_by_thread_count.png")
    plt.show()

def plotKernelUsageVsParticles(df):
    # Sort data by the number of particles
    df = df.sort_values(by='particles')
    unique_threads = sorted(df['threads'].unique())

    markers = ['o', 's', '^', 'D', 'v', 'x', 'p', '*', 'h', '+']

    plt.figure(figsize=(10, 6))

    for i, thread in enumerate(unique_threads):
        # Filter data by thread count
        subset = df[df['threads'] == thread]

        particle_counts = subset['particles'].unique()
        x_values = range(len(particle_counts))
        marker = markers[i % len(markers)]

        kernel_usage_percent = subset['kernel_fraction'] * 100
        plt.plot(x_values, kernel_usage_percent, marker=marker, linestyle='-', linewidth=2,
                 alpha=0.7, label=f'{thread} Threads')

    plt.xlabel('Number of Particles', fontsize=14)
    plt.ylabel('simulateKernel Usage (%)', fontsize=14)
    plt.title('Kernel Usage vs Particles by Thread Count', fontsize=16, weight='bold')

    plt.xticks(x_values, particle_counts, rotation=45, fontsize=12)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Add a legend and ensure it is sorted by thread count
    plt.legend(title='Thread Count', fontsize=12, title_fontsize=14)

    plt.tight_layout()

    plt.savefig("figures/kernel_usage_vs_particles_by_thread_count.png")
    plt.show()

def plotLowestKernelTimes(df):
    df = df.sort_values(by='particles')

    lowest_kernel_times = df.loc[df.groupby('particles')['kernel_time'].idxmin()]

    plt.figure(figsize=(10, 6))

    particle_counts = lowest_kernel_times['particles'].unique()
    x_values = range(len(particle_counts))  # Create evenly spaced x values

    plt.plot(x_values, lowest_kernel_times['kernel_time'], marker='o', linestyle='-', linewidth=2, color='blue')
    for i, txt in enumerate(lowest_kernel_times['threads']):
        plt.annotate(f'{txt} Threads', (x_values[i], lowest_kernel_times['kernel_time'].iloc[i]),
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, color='black')

    plt.xlabel('Number of Particles', fontsize=14)
    plt.ylabel('Lowest simulateKernel Time (ms)', fontsize=14)
    plt.title('Lowest Kernel Time vs Number of Particles (with Corresponding Thread Count)', fontsize=16, weight='bold')

    plt.xticks(x_values, particle_counts, rotation=45, fontsize=12)
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    plt.savefig("figures/lowest_kernel_times.png")
    plt.show()

# Main execution
directory = "cout/"  # Specify your directory here
df = process_all_files(directory)

print(df.head())

plotKernelTimeVsParticles(df)
plotKernelUsageVsParticles(df)
plotLowestKernelTimes(df)