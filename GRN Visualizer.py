import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

""""
Simple code to visualize the graphs of the model
Made with help from Claude
"""

def visualize_grn(grn: np.ndarray, title: str = "Gene Regulatory Network",
                  layout: str = "circular", figsize: tuple = (10, 8),
                  save_path: str = None):
    """
    Visualizes a GRN matrix as a directed graph.

    Parameters
    ----------
    grn        : np.ndarray  — n×n GRN matrix (0 = no edge, 1 = activating, -1 = inhibiting)
    title      : str         — plot title
    layout     : str         — "circular" | "spring" | "kamada_kawai" | "shell"
    figsize    : tuple       — matplotlib figure size
    save_path  : str | None  — if given, saves the figure to this path instead of showing it
    """
    n = grn.shape[0]
    gene_labels = {i: f"G{i+1}" for i in range(n)}

    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    activating_edges = []
    inhibiting_edges = []

    for i in range(n):
        for j in range(n):
            if grn[i, j] == 1:
                G.add_edge(i, j, weight=1)
                activating_edges.append((i, j))
            elif grn[i, j] == -1:
                G.add_edge(i, j, weight=-1)
                inhibiting_edges.append((i, j))

    layouts = {
        "circular":     nx.circular_layout,
        "spring":       lambda G: nx.spring_layout(G, seed=42, k=2.5 / np.sqrt(n)),
        "kamada_kawai": nx.kamada_kawai_layout,
        "shell":        nx.shell_layout,
    }
    pos = layouts.get(layout, nx.circular_layout)(G)

    # ── node colours: blue = has outgoing edges, grey = isolated ──────────
    node_colors = []
    for node in G.nodes():
        out_deg = G.out_degree(node)
        node_colors.append("#378ADD" if out_deg > 0 else "#888780")

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#F8F8F8")
    ax.set_facecolor("#F8F8F8")

    node_size = max(600, int(4000 / n))
    font_size = max(7, int(14 - n * 0.3))

    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors,
                           node_size=node_size,
                           linewidths=1.5,
                           edgecolors="#185FA5")

    nx.draw_networkx_labels(G, pos, ax=ax,
                            labels=gene_labels,
                            font_size=font_size,
                            font_color="white",
                            font_weight="bold")

    arc_rad = 0.15 if n <= 10 else 0.1

    # activating edges — green solid
    nx.draw_networkx_edges(G, pos, ax=ax,
                           edgelist=activating_edges,
                           edge_color="#1D9E75",
                           arrows=True,
                           arrowstyle="-|>",
                           arrowsize=15,
                           width=1.8,
                           connectionstyle=f"arc3,rad={arc_rad}",
                           min_source_margin=18,
                           min_target_margin=18)

    # inhibiting edges — coral dashed
    nx.draw_networkx_edges(G, pos, ax=ax,
                           edgelist=inhibiting_edges,
                           edge_color="#D85A30",
                           style="dashed",
                           arrows=True,
                           arrowstyle="-|>",
                           arrowsize=15,
                           width=1.8,
                           connectionstyle=f"arc3,rad={arc_rad}",
                           min_source_margin=18,
                           min_target_margin=18)

    # ── stats annotation ──────────────────────────────────────────────────
    total_edges = len(activating_edges) + len(inhibiting_edges)
    avg_out = total_edges / n if n > 0 else 0
    stats = (f"Genes: {n}   |   Edges: {total_edges}   |   "
             f"Activating: {len(activating_edges)}   |   "
             f"Inhibiting: {len(inhibiting_edges)}   |   "
             f"Avg out-degree: {avg_out:.1f}")
    fig.text(0.5, 0.02, stats, ha="center", fontsize=9, color="#5F5E5A")

    # ── legend ────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color="#1D9E75", label="Activating (+1)"),
        mpatches.Patch(color="#D85A30", label="Inhibiting (−1)"),
        mpatches.Patch(color="#378ADD", label="Gene (active regulator)"),
        mpatches.Patch(color="#888780", label="Gene (no outgoing edges)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=9, framealpha=0.85, edgecolor="#D3D1C7")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=16, color="#2C2C2A")
    ax.axis("off")
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


from Cell import Cell
from GRNModel import Model

model = Model()
cell = Cell(model)
visualize_grn(cell.GRN)

cell.ExecutePropagation()
visualize_grn(cell.GRN)

cell.Mutate()
visualize_grn(cell.GRN)