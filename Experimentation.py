from enum import unique

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from Model import Model
from numpy.random import random
from matplotlib.patches import Patch
import time
from multiprocessing import Pool

# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT CONFIGURATION — adjust everything here
# ══════════════════════════════════════════════════════════════════════════════

EXPERIMENT_ID   = 1
AMOUNT_OF_RUNS  = 5           # number of parallel runs
MAX_TIMESTEPS   = 5_000_000   # steps per run
LAMBDA          = 3e-6        # probability of target switch per step
WRITE_INTERVAL  = 1_000       # how often to flush buffer to disk
LOG_INTERVAL    = 10_000      # how often to print progress

# Model parameters — passed to every run
MODEL_PARAMS = dict(
    fitness_power   = 2.5,
    min_fitness = 0.0,
    max_fitness = 0.9,
    mutation_factor = 75,
    mean_resource   = 2.0,
    sd_recourse     = 2.0,
    regen_rate      = 0.05,
    division_thres  = 15,
    division_timesteps = 5,
)

# Output directory
DATA_DIR = "Data"

# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_run(filename: str, axes=None, title: str = None, plot_population: bool = True, plot_ignore: int = 100):
    """
    Plot a single experiment file with stacked subplots.
      - Top:    Hamming distance statistics (min, mean ± std)
      - Bottom: Population statistics (total population, mean/max org size) [Optional]

    Pass axes=(ax_top, ax_bottom) to embed in a larger figure,
    or leave as None to create a standalone figure.
    """
    # ── parse file ────────────────────────────────────────────────────────────
    data_rows  = []
    target_ids = []

    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 9 and float(parts[0]) % plot_ignore == 0:
                data_rows.append([float(p) for p in parts[:8]])
                target_ids.append(parts[8])

    if not data_rows:
        print(f"Warning: no data found in {filename}")
        return None, None

    data = np.array(data_rows)

    t          = data[:, 0]
    population = data[:, 1]
    min_dist   = data[:, 2]
    mean_dist  = data[:, 3]
    std_dist   = data[:, 4]
    mean_org   = data[:, 5]
    max_org    = data[:, 6]
    std_org    = data[:, 7]

    # ── figure setup ──────────────────────────────────────────────────────────
    standalone = axes is None
    if standalone:
        if plot_population:
            fig, (ax_top, ax_bot) = plt.subplots(
                2, 1, figsize=(12, 8), sharex=True,
                gridspec_kw={'height_ratios': [1.2, 1]}
            )
        else:
            fig, ax_top = plt.subplots(1, 1, figsize=(12, 5))
            ax_bot = None
        fig.patch.set_facecolor('#0f0f14')
    else:
        # Extract both axes if a sequence was passed, so we can manage visibility
        if isinstance(axes, (tuple, list, np.ndarray)) and len(axes) >= 2:
            ax_top, ax_bot = axes[0], axes[1]
        else:
            ax_top = axes
            ax_bot = None
        fig = ax_top.get_figure()

    # Determine which axes are actively being drawn to
    if plot_population and ax_bot is not None:
        active_axes = [ax_top, ax_bot]
    else:
        active_axes = [ax_top]
        if ax_bot is not None:
            ax_bot.set_visible(False)  # <── Hides the blank white plot completely

    for ax in active_axes:
        ax.set_facecolor('#0f0f14')
        for spine in ax.spines.values():
            spine.set_edgecolor('#2a2a38')

    # ── environment shading (shared between active panels) ────────────────────
    COLOR_A = '#2a2a2a'
    COLOR_B = '#181818'

    env_handles = []
    if len(target_ids) == len(t):
        span_start = 0
        for k in range(1, len(t)):
            changed  = target_ids[k] != target_ids[k - 1]
            last     = k == len(t) - 1
            if changed or last:
                span_end = k if changed else k + 1
                color = COLOR_A if target_ids[span_start] == 'A' else COLOR_B
                for ax in active_axes:
                    ax.axvspan(t[span_start], t[span_end - 1],
                               facecolor=color, alpha=1.0, zorder=0)
                span_start = k

        env_handles = [
            Patch(facecolor=COLOR_A, edgecolor='#3a3a3a', label='Target A'),
            Patch(facecolor=COLOR_B, edgecolor='#282828', label='Target B'),
        ]

    # ── top panel: Hamming distance ───────────────────────────────────────────
    ax_top.fill_between(t,
                        mean_dist - std_dist,
                        mean_dist + std_dist,
                        color='#ffa028', alpha=0.15, linewidth=0,
                        label='mean ± std', zorder=2)
    ax_top.plot(t, mean_dist, color='#ffa028', linewidth=1.8,
                label='mean distance', zorder=3)
    ax_top.plot(t, min_dist,  color='#00d2b4', linewidth=1.8,
                label='min distance',  zorder=3)

    ax_top.set_xlim(t[0], t[-1])
    ax_top.set_ylim(0, 20)
    ax_top.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax_top.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax_top.grid(which='major', color='#2a2a38', linewidth=0.8, zorder=1)
    ax_top.grid(which='minor', color='#1e1e28', linewidth=0.4, zorder=1)
    ax_top.tick_params(colors='#6e7a8a', which='both')
    ax_top.set_ylabel('Hamming Distance', color='#6e7a8a', fontsize=11)

    # If population is hidden, the top panel becomes the bottom-most panel
    if not plot_population:
        ax_top.set_xlabel('Timestep', color='#6e7a8a', fontsize=11)

    plot_title = title or os.path.splitext(os.path.basename(filename))[0]
    ax_top.set_title(plot_title, color='#d2dce8', fontsize=13, pad=10)

    dist_handles, dist_labels = ax_top.get_legend_handles_labels()
    ax_top.legend(handles=env_handles + dist_handles,
                  facecolor='#16161f', edgecolor='#2a2a38',
                  labelcolor='#d2dce8', fontsize=10)

    # ── bottom panel: population statistics ───────────────────────────────────
    if plot_population and ax_bot is not None:
        ax_bot.fill_between(t,
                            mean_org - std_org,
                            mean_org + std_org,
                            color='#32dc64', alpha=0.15, linewidth=0,
                            label='mean org size ± std', zorder=2)
        ax_bot.plot(t, mean_org,   color='#32dc64', linewidth=1.8,
                    label='mean org size',    zorder=3)
        ax_bot.plot(t, max_org,    color='#f03264', linewidth=1.4,
                    linestyle='--', label='max org size', zorder=3)

        ax_bot.set_ylim(bottom=0)
        ax_bot.yaxis.set_major_locator(ticker.AutoLocator())
        ax_bot.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax_bot.grid(which='major', color='#2a2a38', linewidth=0.8, zorder=1)
        ax_bot.grid(which='minor', color='#1e1e28', linewidth=0.4, zorder=1)
        ax_bot.tick_params(colors='#6e7a8a', which='both')
        ax_bot.set_xlabel('Timestep',          color='#6e7a8a', fontsize=11)
        ax_bot.set_ylabel('Count / Org Size',  color='#6e7a8a', fontsize=11)

        pop_handles, _ = ax_bot.get_legend_handles_labels()
        ax_bot.legend(handles=pop_handles,
                      facecolor='#16161f', edgecolor='#2a2a38',
                      labelcolor='#d2dce8', fontsize=10)

    if standalone:
        fig.tight_layout()
        plt.show()

    return fig, (ax_top, ax_bot)


def plot_runs(filenames: list, titles: list = None):
    """
    Plot multiple experiment files side by side.
    Each column has two stacked subplots (distance on top, population below).
    """
    n = len(filenames)
    fig, axes = plt.subplots(
        2, n, figsize=(12 * n, 8), sharex='col',
        gridspec_kw={'height_ratios': [1.2, 1]}
    )
    fig.patch.set_facecolor('#0f0f14')

    # Normalise axes shape for n==1
    if n == 1:
        axes = axes.reshape(2, 1)

    for k, fname in enumerate(filenames):
        title = (titles[k] if titles and k < len(titles)
                 else os.path.splitext(os.path.basename(fname))[0])
        plot_run(fname, axes=(axes[0, k], axes[1, k]), title=title)

    fig.tight_layout()
    plt.show()
    return fig, axes


# ══════════════════════════════════════════════════════════════════════════════
#  SIMULATION RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(args):
    """Worker function. Receives (run_index, params_dict)."""
    i, params = args
    buffer = []

    os.makedirs(DATA_DIR, exist_ok=True)

    # Build the folder name from params, not MODEL_PARAMS
    folder = (f"fp{params['fitness_power']}_"
              f"mi{params['min_fitness']}_"
              f"ma{params['max_fitness']}_"
              f"mu{params['mutation_factor']}_"
              f"xs{params['x_size']}_"
              f"ys{params['y_size']}_"
              f"mr{params['mean_resource']}_"
              f"sd{params['sd_recourse']}_"
              f"rr{params['regen_rate']}_"
              f"th{params['division_thres']}_"
              f"ti{params['division_timesteps']}")

    out_path = f"{DATA_DIR}/{folder}/{EXPERIMENT_ID}.{i}.txt"
    os.makedirs(f"{DATA_DIR}/{folder}", exist_ok=True)

    model = Model(**params)
    start = time.time()

    with open(out_path, 'w') as file:
        for t in range(MAX_TIMESTEPS):
            if t % LOG_INTERVAL == 0:
                elapsed = time.time() - start
                print(f"Run {i}: {t}/{MAX_TIMESTEPS}  ({elapsed:.1f}s)")
                start = time.time()

            if random() < LAMBDA:
                model.SwitchTarget()

            model.ExecuteStep()

            buffer.append(
                f"{t}\t"
                f"{model.TotalPopulation}\t"
                f"{model.MinimalDistance}\t"
                f"{model.MeanDistance:.3f}\t"
                f"{model.STDDistance:.3f}\t"
                f"{model.MeanOrgSize:.3f}\t"
                f"{model.MaxOrgSize}\t"
                f"{model.STDOrgSize:.3f}\t"
                f"{model.TargetID}\n"
            )

            if len(buffer) >= WRITE_INTERVAL:
                file.writelines(buffer)
                buffer.clear()

            if model.TotalPopulation == 0:
                print(f"Run {i}: population extinct at t={t}.")
                break
        file.writelines(buffer)

    os.makedirs(f"{DATA_DIR}/{folder}/cells", exist_ok=True)
    cells = model.Cells
    for i,c in enumerate(cells):
        c.UniCellular = True
        c.ToJSON(f"{DATA_DIR}/{folder}/cells/{i}.{c.ID}")

    print(f"Run {i} complete → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT — only touch things below this line
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    EXPERIMENT_ID = 1

    # Model parameters — passed to every run
    MODEL_PARAMS = dict(
        fitness_power=3,
        min_fitness=0.001,
        max_fitness=0.2,
        mutation_factor=25,
        x_size=50,
        y_size=50,
        mean_resource=1.0,
        sd_recourse=5.0,
        regen_rate=0.05,
        division_thres=15,
        division_timesteps=5
    )

    folder = (f"fp{MODEL_PARAMS["fitness_power"]}_"
              f"mi{MODEL_PARAMS["min_fitness"]}_"
              f"ma{MODEL_PARAMS["max_fitness"]}_"
              f"mu{MODEL_PARAMS["mutation_factor"]}_"
              f"xs{MODEL_PARAMS["x_size"]}_"
              f"ys{MODEL_PARAMS["y_size"]}_"
              f"mr{MODEL_PARAMS["mean_resource"]}_"
              f"sd{MODEL_PARAMS["sd_recourse"]}_"
              f"rr{MODEL_PARAMS["regen_rate"]}_"
              f"th{MODEL_PARAMS["division_thres"]}_"
              f"ti{MODEL_PARAMS["division_timesteps"]}")

    #── run ───────────────────────────────────────────────────────────────────
    args = [(i, MODEL_PARAMS) for i in range(AMOUNT_OF_RUNS)]

    start = time.time()
    with Pool() as pool:
        pool.map(run_experiment, args)
    print(f"All runs finished in {time.time() - start:.1f}s")

    # ── plot ──────────────────────────────────────────────────────────────────
    files = [f"{DATA_DIR}/{folder}/{EXPERIMENT_ID}.{i}.txt" for i in range(AMOUNT_OF_RUNS)]

    # Single run
    # plot_run(files[4])

    # All runs side by side (comment out if you only want one)
    plot_runs(files)