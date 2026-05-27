import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from multiprocessing import Pool
from GRNModel import Model
from numpy.random import random
import time


def plot_run(filename: str, ax=None, title: str = None):
    """
    Plot mean distance, minimum distance, and std band over time
    for a single experiment file, with background shading showing
    which environment (target A or B) was active at each timestep.
    """
    data_rows  = []
    target_ids = []
    with open(filename, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        parts = line.split('\t')
        if len(parts) == 6:
            data_rows.append([float(p) for p in parts[:5]])
            target_ids.append(parts[5])
        i += 1

    data = np.array(data_rows)
    if data.ndim == 1:
        data = data[np.newaxis, :]

    t         = data[:, 0]
    min_dist  = data[:, 2]
    mean_dist = data[:, 3]
    std_dist  = data[:, 4]

    # Figures
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#0f0f14')
    else:
        fig = ax.get_figure()

    ax.set_facecolor('#0f0f14')
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a2a38')

    # We find contiguous spans where the target is the same and shade each
    # span. Target A gets a slightly lighter grey, B a slightly darker one.
    COLOR_A = '#2a2a2a'
    COLOR_B = '#181818'

    if len(target_ids) == len(t):
        span_start = 0
        for k in range(1, len(t)):
            if target_ids[k] != target_ids[k - 1] or k == len(t) - 1:
                span_end = k if target_ids[k] != target_ids[k - 1] else k + 1
                color = COLOR_A if target_ids[span_start] == 'A' else COLOR_B
                ax.axvspan(t[span_start], t[span_end - 1],
                           facecolor=color, alpha=1.0, zorder=0)
                span_start = k

        # Add legend patches for the environments
        from matplotlib.patches import Patch
        env_handles = [
            Patch(facecolor=COLOR_A, edgecolor='#3a3a3a', label='Target A'),
            Patch(facecolor=COLOR_B, edgecolor='#282828', label='Target B'),
        ]
    else:
        env_handles = []

    # Standard deviation
    ax.fill_between(t,
                    mean_dist - std_dist,
                    mean_dist + std_dist,
                    color='#ffa028', alpha=0.15, linewidth=0,
                    label='mean ± std', zorder=2)

    # Plotting values
    ax.plot(t, mean_dist, color='#ffa028', linewidth=1.8,
            label='mean distance', zorder=3)
    ax.plot(t, min_dist,  color='#00d2b4', linewidth=1.8,
            label='min distance',  zorder=3)

    # Styling
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, 20)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.grid(which='major', color='#2a2a38', linewidth=0.8, zorder=1)
    ax.grid(which='minor', color='#1e1e28', linewidth=0.4, zorder=1)
    ax.tick_params(colors='#6e7a8a', which='both')
    ax.set_xlabel('Timestep',        color='#6e7a8a', fontsize=11)
    ax.set_ylabel('Hamming Distance', color='#6e7a8a', fontsize=11)

    plot_title = title or os.path.splitext(os.path.basename(filename))[0]
    ax.set_title(plot_title, color='#d2dce8', fontsize=13, pad=10)

    # Combine environment and data-series legend entries
    data_handles, data_labels = ax.get_legend_handles_labels()
    ax.legend(handles=env_handles + data_handles,
              facecolor='#16161f', edgecolor='#2a2a38',
              labelcolor='#d2dce8', fontsize=10)

    if standalone:
        fig.tight_layout()
        plt.show()
        return fig, ax

    return fig, ax


def plot_runs(filenames: list[str], titles: list[str] = None):
    """
    Plot multiple experiment files side by side, one subplot each.
    """
    n = len(filenames)
    fig, axes = plt.subplots(1, n, figsize=(10 * n, 5),
                             sharey=True, squeeze=False)
    fig.patch.set_facecolor('#0f0f14')

    for k, (fname, ax) in enumerate(zip(filenames, axes[0])):
        title = (titles[k] if titles and k < len(titles)
                 else os.path.splitext(os.path.basename(fname))[0])
        plot_run(fname, ax=ax, title=title)

    fig.tight_layout()
    plt.show()
    return fig, axes


def run_experiment(i):
    max_time_steps = 600000
    lamb = 3e-4
    experiment_run = 9
    buffer = []

    model = Model()
    start = time.time()
    with open(f"Data/Data-{experiment_run}.{i}.txt", 'w') as file:
        for t in range(max_time_steps):
            if t %10000==0:
                end = time.time()
                print(f"Run {i}: {t}/{max_time_steps}: {(end - start):.3f} seconds")
                start = time.time()
            if random() < lamb:
                model.SwitchTarget()
            model.ExecuteStep()

            buffer.append(f"{t}\t{model.TotalPopulation}\t"
                          f"{model.MinimalDistance}\t"
                          f"{model.MeanDistance:.3f}\t"
                          f"{model.STDDistance:.3f}\t"
                          f"{model.TargetID}\n")
            if len(buffer) >= 1000:
                file.writelines(buffer)
                buffer.clear()

            if model.TotalPopulation == 0:
                print(f"Population {experiment_run}.{i} has died.")
                break
        file.writelines(buffer)


if __name__ == '__main__':
    amount_of_runs = 5

    """"
    Run model on higher mutation rate,
    same death count and lower fitness power.
    """

    start = time.time()
    # with Pool() as pool:
    #     pool.map(run_experiment, range(amount_of_runs))
    end = time.time()
    print(f"Total runtime {end - start} seconds")

    experiment_run = 8
    files = [f"Data/Data-{experiment_run}.{i}.txt" for i in range(amount_of_runs)]
    plot_run(files[1])