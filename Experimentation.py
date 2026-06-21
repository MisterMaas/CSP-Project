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
MAX_TIMESTEPS   = 500_000       # steps per run
LAMBDA          = 0      # probability of target switch per step
WRITE_INTERVAL  = 1_000       # how often to flush buffer to disk
LOG_INTERVAL    = 10_000      # how often to print progress

# Model parameters — passed to every run
MODEL_PARAMS = dict(
    fitness_power   = 100,
    min_fitness = 0.0,
    max_fitness = 1,
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

# Set font family globally to Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

def plot_run(filename: str, axes=None, title: str = None, plot_population: bool = True, plot_ignore: int = 100):
    """
    Plot a single experiment file with stacked subplots.
      - Top:    Hamming distance statistics (min, mean ± std)
      - Bottom: Population statistics (total population, mean org size ± std) [Optional]

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

    # ── B&W palette ───────────────────────────────────────────────────────────
    COLOR_BG      = 'white'          # figure / axes background
    COLOR_SPINE   = '#bbbbbb'        # axis border lines
    COLOR_GRID_MJ = '#cccccc'        # major gridlines
    COLOR_GRID_MN = '#e0e0e0'        # minor gridlines
    COLOR_TICK    = '#444444'        # tick labels
    COLOR_LABEL   = '#222222'        # axis labels & title
    COLOR_LEGEND_BG  = 'white'
    COLOR_LEGEND_EDG = '#bbbbbb'
    COLOR_LEGEND_TXT = '#222222'

    # Line / fill colours (all neutral)
    COLOR_MEAN_DIST  = '#111111'     # mean Hamming distance  – solid black
    COLOR_MIN_DIST   = '#555555'     # min  Hamming distance  – dark gray
    COLOR_FILL_DIST  = '#aaaaaa'     # std band               – mid gray
    COLOR_MEAN_ORG   = '#222222'     # mean org size          – near-black
    COLOR_FILL_ORG   = '#bbbbbb'     # std band               – light gray

    # Environment shading
    COLOR_ENV_A = '#ffffff'          # Target A → white
    COLOR_ENV_B = '#e8e8e8'          # Target B → light gray

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
        fig.patch.set_facecolor(COLOR_BG)
    else:
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
            ax_bot.set_visible(False)

    for ax in active_axes:
        ax.set_facecolor(COLOR_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(COLOR_SPINE)

    # ── environment shading ───────────────────────────────────────────────────
    env_handles = []
    if len(target_ids) == len(t):
        span_start = 0
        for k in range(1, len(t)):
            changed = target_ids[k] != target_ids[k - 1]
            last    = k == len(t) - 1
            if changed or last:
                span_end = k if changed else k + 1
                color = COLOR_ENV_A if target_ids[span_start] == 'A' else COLOR_ENV_B
                for ax in active_axes:
                    ax.axvspan(t[span_start], t[span_end - 1],
                               facecolor=color, alpha=1.0, zorder=0)
                span_start = k

        # fontname removed from Patch (handled globally now)
        env_handles = [
            Patch(facecolor=COLOR_ENV_A, edgecolor=COLOR_SPINE, label='Target A'),
            Patch(facecolor=COLOR_ENV_B, edgecolor=COLOR_SPINE, label='Target B'),
        ]

    # ── top panel: Hamming distance ───────────────────────────────────────────
    # fontname removed from fill_between and plot
    ax_top.fill_between(t,
                        mean_dist - std_dist,
                        mean_dist + std_dist,
                        color=COLOR_FILL_DIST, alpha=0.4, linewidth=0,
                        label='Mean ± STD', zorder=2)
    ax_top.plot(t, mean_dist, color=COLOR_MEAN_DIST, linewidth=1.8,
                label='Mean Hamming Distance', zorder=3)

    ax_top.set_xlim(t[0], t[-1])
    ax_top.yaxis.set_major_locator(ticker.AutoLocator())
    ax_top.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_top.grid(which='major', color=COLOR_GRID_MJ, linewidth=0.8, zorder=1)
    ax_top.grid(which='minor', color=COLOR_GRID_MN, linewidth=0.4, zorder=1)
    ax_top.tick_params(colors=COLOR_TICK, which='both')
    ax_top.set_ylabel('Hamming Distance', color=COLOR_LABEL, fontsize=11)

    if not plot_population:
        ax_top.set_xlabel('Timestep', color=COLOR_LABEL, fontsize=11)

    plot_title = title or os.path.splitext(os.path.basename(filename))[0]
    ax_top.set_title(plot_title, color=COLOR_LABEL, fontsize=13, pad=10)

    dist_handles, _ = ax_top.get_legend_handles_labels()
    ax_top.legend(handles=env_handles + dist_handles,
                  facecolor=COLOR_LEGEND_BG, edgecolor=COLOR_LEGEND_EDG,
                  labelcolor=COLOR_LEGEND_TXT, fontsize=10)

    # ── bottom panel: population statistics (mean ± std only) ─────────────────
    if plot_population and ax_bot is not None:
        # fontname removed from fill_between and plot
        ax_bot.fill_between(t,
                            mean_org - std_org,
                            mean_org + std_org,
                            color=COLOR_FILL_ORG, alpha=0.4, linewidth=0,
                            label='Mean Org Size ± std', zorder=2)
        ax_bot.plot(t, mean_org, color=COLOR_MEAN_ORG, linewidth=1.8,
                    label='Mean Org Size', zorder=3)

        ax_bot.set_ylim(bottom=0)
        ax_bot.yaxis.set_major_locator(ticker.AutoLocator())
        ax_bot.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax_bot.grid(which='major', color=COLOR_GRID_MJ, linewidth=0.8, zorder=1)
        ax_bot.grid(which='minor', color=COLOR_GRID_MN, linewidth=0.4, zorder=1)
        ax_bot.tick_params(colors=COLOR_TICK, which='both')
        ax_bot.set_xlabel('Timestep', color=COLOR_LABEL, fontsize=11)
        ax_bot.set_ylabel('Organism Size', color=COLOR_LABEL, fontsize=11)

        pop_handles, _ = ax_bot.get_legend_handles_labels()
        ax_bot.legend(handles=pop_handles,
                      facecolor=COLOR_LEGEND_BG, edgecolor=COLOR_LEGEND_EDG,
                      labelcolor=COLOR_LEGEND_TXT, fontsize=10)

    if standalone:
        fig.tight_layout()
        plt.show()

    return fig, (ax_top, ax_bot)


def plot_runs(filenames: list, titles: list = None):
    """
    Plot multiple experiment files side by side.
    Each column has two stacked subplots (distance on top, population below).
    Two unified legends are displayed on the far right.
    """
    n = len(filenames)
    # Extra width added to figsize (e.g., + 2.5) to accommodate the right-side legends
    fig, axes = plt.subplots(
        2, n, figsize=(5 * n + 2.5, 7), sharex='col',
        gridspec_kw={'height_ratios': [1.2, 1]}
    )

    # Normalise axes shape for n==1
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for k, fname in enumerate(filenames):
        title = (titles[k] if titles and k < len(titles)
                 else os.path.splitext(os.path.basename(fname))[0])

        title = "Run "+ str(int(title.split(".")[1]) + 1)
        # Plot each run
        plot_run(fname, axes=(axes[0, k], axes[1, k]), title=title)

        # Remove individual subplots' internal legends
        if axes[0, k].get_legend():
            axes[0, k].get_legend().remove()
        if axes[1, k].get_legend():
            axes[1, k].get_legend().remove()

    # ── Gather distinct handles from the first column ────────────────────────
    top_handles, top_labels = axes[0, 0].get_legend_handles_labels()
    bot_handles, bot_labels = axes[1, 0].get_legend_handles_labels()

    # ── Render Top Right Legend (Fitness & Environment) ──────────────────────
    leg_top = fig.legend(
        handles=top_handles,
        labels=top_labels,
        loc='upper left',
        bbox_to_anchor=(0.84, 0.88),  # Positioned to the right of the top row
        facecolor='white',
        edgecolor='#bbbbbb',
        fontsize=10,
        title="Hamming Distance & Environment",
        title_fontsize=11
    )
    leg_top._legend_box.align = "left"

    # ── Render Bottom Right Legend (Organism Size) ───────────────────────────
    leg_bot = fig.legend(
        handles=bot_handles,
        labels=bot_labels,
        loc='upper left',
        bbox_to_anchor=(0.84, 0.45),  # Positioned to the right of the bottom row
        facecolor='white',
        edgecolor='#bbbbbb',
        fontsize=10,
        title="Organism Size",
        title_fontsize=11
    )
    leg_bot._legend_box.align = "left"

    # Use rect parameter to leave the right 16% of the canvas open for legends
    fig.tight_layout(rect=[0.05, 0, 0.83, 1])
    plt.show()

    return fig, axes

# ══════════════════════════════════════════════════════════════════════════════
#  SIMULATION RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def plot_runs(filenames: list, titles: list = None):
    """
    Plot multiple experiment files side by side.
    Each column has two stacked subplots (distance on top, population below).
    Y-axis labels are only shown on the leftmost subplots.
    Two unified legends are displayed on the far right.
    """
    n = len(filenames)
    # Extra width added to figsize to accommodate the right-side legends
    fig, axes = plt.subplots(
        2, n, figsize=(5 * n + 2.5, 7), sharex='col',
        gridspec_kw={'height_ratios': [1.2, 1]}
    )

    # Normalise axes shape for n==1
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for k, fname in enumerate(filenames):
        title = (titles[k] if titles and k < len(titles)
                 else os.path.splitext(os.path.basename(fname))[0])

        title = "Run " + str(int(title.split(".")[1]) + 1)

        # Plot each run
        plot_run(fname, axes=(axes[0, k], axes[1, k]), title=title)

        # Hide y-axis labels for all columns except the first one
        if k > 0:
            axes[0, k].set_ylabel('')
            axes[1, k].set_ylabel('')

        # Remove individual subplots' internal legends
        if axes[0, k].get_legend():
            axes[0, k].get_legend().remove()
        if axes[1, k].get_legend():
            axes[1, k].get_legend().remove()

    # ── Gather distinct handles from the first column ────────────────────────
    top_handles, top_labels = axes[0, 0].get_legend_handles_labels()
    bot_handles, bot_labels = axes[1, 0].get_legend_handles_labels()

    # ── Render Top Right Legend (Fitness & Environment) ──────────────────────
    leg_top = fig.legend(
        handles=top_handles,
        labels=top_labels,
        loc='upper left',
        bbox_to_anchor=(0.84, 0.88),  # Positioned to the right of the top row
        facecolor='white',
        edgecolor='#bbbbbb',
        fontsize=10,
        title="Fitness",
        title_fontsize=11
    )
    leg_top._legend_box.align = "left"

    # ── Render Bottom Right Legend (Organism Size) ───────────────────────────
    leg_bot = fig.legend(
        handles=bot_handles,
        labels=bot_labels,
        loc='upper left',
        bbox_to_anchor=(0.84, 0.45),  # Positioned to the right of the bottom row
        facecolor='white',
        edgecolor='#bbbbbb',
        fontsize=10,
        title="Organism Size",
        title_fontsize=11
    )
    leg_bot._legend_box.align = "left"

    # Compress subplots to the left 83% of the window so they don't hit the legends
    fig.tight_layout(rect=[0.05, 0, 0.83, 1])
    plt.show()

    return fig, axes


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT — only touch things below this line
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    EXPERIMENT_ID = 1

    # Model parameters — passed to every run
    MODEL_PARAMS = dict(
        fitness_power=1,
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

    # #── run ───────────────────────────────────────────────────────────────────
    # args = [(i, MODEL_PARAMS) for i in range(AMOUNT_OF_RUNS)]
    #
    # start = time.time()
    # with Pool() as pool:
    #     pool.map(run_experiment, args)
    # print(f"All runs finished in {time.time() - start:.1f}s")

    # ── plot ──────────────────────────────────────────────────────────────────
    files = [(f"Data/{folder}/{EXPERIMENT_ID}.{i}.txt") for i in range(AMOUNT_OF_RUNS)]

    # Single run
    # plot_run(files[4])

    # All runs side by side (comment out if you only want one)
    plot_runs(files)