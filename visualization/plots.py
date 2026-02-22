# """
# plots.py
# All matplotlib/seaborn figures for the paper and demo.

# Figures produced:
# 1. adaptation_curves.png    — Rolling mean reward per agent per world
# 2. hypothesis_trajectory.png — UMAP of LLM hypothesis embeddings over time
# 3. strategy_heatmap.png     — Strategy diversity across worlds (agent vs world)
# 4. boden_taxonomy.png       — Distribution of Boden types per agent
# 5. linguistic_surprise.png  — Embedding distance over revision cycles
# """

# import json
# import numpy as np
# import pandas as pd
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import seaborn as sns
# from pathlib import Path

# RESULTS_DIR = Path(__file__).parent.parent / "results"
# RESULTS_DIR.mkdir(exist_ok=True)

# # Consistent color palette per agent
# AGENT_COLORS = {
#     "v1_dqn":           "#E74C3C",   # red
#     "v2_rule_follower":  "#3498DB",  # blue
#     "v3_scientist":      "#2ECC71",  # green
#     "v4_novelty_seeker": "#9B59B6",  # purple
# }

# AGENT_LABELS = {
#     "v1_dqn":           "V1: Silent Instinct (DQN)",
#     "v2_rule_follower":  "V2: Rule Follower",
#     "v3_scientist":      "V3: Scientist",
#     "v4_novelty_seeker": "V4: Novelty Seeker",
# }

# WORLD_LABELS = {
#     "baseline": "A: Baseline",
#     "inverted": "B: Inverted Gravity",
#     "viscous":  "C: Viscous Void",
#     "delayed":  "D: Drunken Cart",
# }

# BODEN_COLORS = {
#     "combinational":   "#F39C12",
#     "exploratory":     "#27AE60",
#     "transformational": "#8E44AD",
# }


# def _rolling_mean(series, window=10):
#     return pd.Series(series).rolling(window=window, min_periods=1).mean()


# # ── Figure 1: Adaptation Curves ────────────────────────────────────────────

# def plot_adaptation_curves(episodes_df: pd.DataFrame, world: str,
#                            save: bool = True, ax=None):
#     """
#     Rolling mean reward over episodes for all agents in a given world.
#     This is your main result figure.
#     """
#     own_fig = ax is None
#     if own_fig:
#         fig, ax = plt.subplots(figsize=(10, 6))

#     world_df = episodes_df[episodes_df["world"] == world]

#     for agent in world_df["agent"].unique():
#         agent_df = world_df[world_df["agent"] == agent].sort_values("episode_num")
#         rewards = agent_df["total_reward"].values
#         smoothed = _rolling_mean(rewards)
#         color = AGENT_COLORS.get(agent, "gray")
#         label = AGENT_LABELS.get(agent, agent)
#         ax.plot(smoothed, color=color, label=label, linewidth=2)
#         ax.fill_between(range(len(rewards)),
#                         smoothed - pd.Series(rewards).rolling(10, min_periods=1).std().fillna(0),
#                         smoothed + pd.Series(rewards).rolling(10, min_periods=1).std().fillna(0),
#                         color=color, alpha=0.15)

#     ax.axhline(y=400, color="black", linestyle="--", alpha=0.4, label="Solved threshold (400)")
#     ax.set_xlabel("Episode", fontsize=12)
#     ax.set_ylabel("Reward (rolling mean, window=10)", fontsize=12)
#     ax.set_title(f"Adaptation Curves — {WORLD_LABELS.get(world, world)}", fontsize=14, fontweight="bold")
#     ax.legend(fontsize=10)
#     ax.grid(True, alpha=0.3)
#     sns.despine(ax=ax)

#     if own_fig and save:
#         path = RESULTS_DIR / f"adaptation_curves_{world}.png"
#         fig.savefig(path, dpi=150, bbox_inches="tight")
#         plt.close(fig)
#         print(f"Saved: {path}")
#         return str(path)

#     return ax


# def plot_all_adaptation_curves(episodes_df: pd.DataFrame, save: bool = True):
#     """2×2 grid of adaptation curves for all four worlds."""
#     worlds = ["baseline", "inverted", "viscous", "delayed"]
#     fig, axes = plt.subplots(2, 2, figsize=(16, 10))
#     axes = axes.flatten()

#     for i, world in enumerate(worlds):
#         plot_adaptation_curves(episodes_df, world, save=False, ax=axes[i])

#     fig.suptitle("Agent Adaptation Across Physics Worlds", fontsize=16, fontweight="bold", y=1.02)
#     plt.tight_layout()

#     if save:
#         path = RESULTS_DIR / "adaptation_curves_all.png"
#         fig.savefig(path, dpi=150, bbox_inches="tight")
#         plt.close(fig)
#         print(f"Saved: {path}")
#         return str(path)
#     return fig


# # ── Figure 2: Hypothesis Trajectory (UMAP) ─────────────────────────────────

# def plot_hypothesis_trajectory(hypotheses_df: pd.DataFrame,
#                                 agent: str = "v3_scientist",
#                                 world: str = "inverted",
#                                 save: bool = True):
#     """
#     UMAP projection of hypothesis embeddings over time.
#     Shows how the LLM's 'theory' moves through concept space across revisions.
#     This is your signature visualization.
#     """
#     import umap

#     df = hypotheses_df[
#         (hypotheses_df["agent"] == agent) &
#         (hypotheses_df["world"] == world) &
#         (hypotheses_df["embedding"].notna())
#     ].copy()

#     if len(df) < 2:
#         print(f"Not enough hypotheses for UMAP ({len(df)}). Need at least 2.")
#         return None

#     # Parse embeddings
#     embeddings = np.array([json.loads(e) for e in df["embedding"].values])

#     # Adjust n_neighbors based on number of samples
#     # n_neighbors = max(1, min(3, len(df) - 1))
#     if len(df) < 3:
#         print(f"Skipping UMAP for {agent}/{world} — need at least 3 hypotheses (have {len(df)})")
#         return None
#     n_neighbors = max(2, min(5, len(df) - 1))
#     reducer = umap.UMAP(n_components=2, random_state=42, min_dist=0.1, n_neighbors=n_neighbors)
#     coords = reducer.fit_transform(embeddings)

#     fig, ax = plt.subplots(figsize=(10, 8))

#     # Color by episode number (time gradient)
#     episodes = df["episode_num"].values
#     norm = plt.Normalize(episodes.min(), episodes.max())
#     cmap = cm.viridis

#     # Draw trajectory lines
#     for i in range(len(coords) - 1):
#         ax.plot(coords[i:i+2, 0], coords[i:i+2, 1],
#                 color="gray", alpha=0.4, linewidth=1, zorder=1)

#     # Draw points colored by Boden type
#     boden_types = df["boden_type"].fillna("unknown").values
#     for i, (x, y) in enumerate(coords):
#         btype = boden_types[i]
#         color = BODEN_COLORS.get(btype, "#95A5A6")
#         ax.scatter(x, y, c=color, s=120, zorder=2, edgecolors="white", linewidth=1.5)

#         # Label each point with revision number
#         ax.annotate(f"H{i+1}", (x, y), textcoords="offset points",
#                     xytext=(6, 6), fontsize=8, color="gray")

#     # Legend for Boden types
#     from matplotlib.patches import Patch
#     legend_elements = [
#         Patch(facecolor=BODEN_COLORS["combinational"], label="Combinational"),
#         Patch(facecolor=BODEN_COLORS["exploratory"], label="Exploratory"),
#         Patch(facecolor=BODEN_COLORS["transformational"], label="Transformational"),
#         Patch(facecolor="#95A5A6", label="Unknown"),
#     ]
#     ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

#     ax.set_xlabel("UMAP Dimension 1", fontsize=12)
#     ax.set_ylabel("UMAP Dimension 2", fontsize=12)
#     ax.set_title(
#         f"Hypothesis Trajectory in Concept Space\n"
#         f"{AGENT_LABELS.get(agent, agent)} | {WORLD_LABELS.get(world, world)}",
#         fontsize=13, fontweight="bold"
#     )
#     ax.grid(True, alpha=0.2)
#     sns.despine(ax=ax)

#     if save:
#         path = RESULTS_DIR / f"hypothesis_trajectory_{agent}_{world}.png"
#         fig.savefig(path, dpi=150, bbox_inches="tight")
#         plt.close(fig)
#         print(f"Saved: {path}")
#         return str(path)
#     return fig


# # ── Figure 3: Strategy Heatmap ─────────────────────────────────────────────

# def plot_strategy_heatmap(novelty_df: pd.DataFrame, save: bool = True):
#     """
#     Heatmap: mean linguistic surprise per (agent, world) cell.
#     Shows which worlds forced the most hypothesis revision.
#     """
#     pivot = novelty_df.groupby(["agent", "world"])["linguistic_surprise"].mean().unstack(fill_value=0)

#     # Rename for display
#     pivot.index = [AGENT_LABELS.get(a, a) for a in pivot.index]
#     pivot.columns = [WORLD_LABELS.get(w, w) for w in pivot.columns]

#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
#                 linewidths=0.5, ax=ax, cbar_kws={"label": "Mean Linguistic Surprise"})
#     ax.set_title("Mean Linguistic Surprise per Agent per World\n(higher = more theory revision forced)",
#                  fontsize=13, fontweight="bold")
#     ax.set_xlabel("World", fontsize=11)
#     ax.set_ylabel("Agent", fontsize=11)
#     plt.xticks(rotation=30, ha="right")

#     if save:
#         path = RESULTS_DIR / "strategy_heatmap.png"
#         fig.savefig(path, dpi=150, bbox_inches="tight")
#         plt.close(fig)
#         print(f"Saved: {path}")
#         return str(path)
#     return fig


# # ── Figure 4: Boden Taxonomy Distribution ──────────────────────────────────

# def plot_boden_distribution(hypotheses_df: pd.DataFrame, save: bool = True):
#     """
#     Stacked bar chart showing proportion of Boden types per agent.
#     This is your theoretical contribution visualization.
#     """
#     df = hypotheses_df[hypotheses_df["boden_type"].notna()].copy()
#     if df.empty:
#         print("No Boden classifications yet.")
#         return None

#     counts = df.groupby(["agent", "boden_type"]).size().unstack(fill_value=0)
#     counts = counts.div(counts.sum(axis=1), axis=0)  # Normalize to proportions

#     # Rename for display
#     counts.index = [AGENT_LABELS.get(a, a) for a in counts.index]

#     fig, ax = plt.subplots(figsize=(10, 6))
#     colors = [BODEN_COLORS.get(c, "#95A5A6") for c in counts.columns]
#     counts.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor="white", linewidth=0.5)

#     ax.set_xlabel("Agent", fontsize=12)
#     ax.set_ylabel("Proportion of Strategies", fontsize=12)
#     ax.set_title("Boden Novelty Type Distribution per Agent", fontsize=14, fontweight="bold")
#     ax.legend(title="Boden Type", loc="upper right")
#     ax.set_ylim(0, 1)
#     plt.xticks(rotation=15, ha="right")
#     sns.despine(ax=ax)

#     if save:
#         path = RESULTS_DIR / "boden_taxonomy.png"
#         fig.savefig(path, dpi=150, bbox_inches="tight")
#         plt.close(fig)
#         print(f"Saved: {path}")
#         return str(path)
#     return fig


# # ── Figure 5: Linguistic Surprise Over Time ─────────────────────────────────

# def plot_linguistic_surprise(novelty_df: pd.DataFrame,
#                               agent: str = "v3_scientist",
#                               save: bool = True):
#     """
#     Line plot of embedding distance over revision cycles.
#     Shows when the LLM made large conceptual jumps.
#     """
#     df = novelty_df[novelty_df["agent"] == agent].sort_values("episode_num")
#     if df.empty:
#         print("No novelty data.")
#         return None

#     fig, ax = plt.subplots(figsize=(12, 5))

#     worlds = df["world"].unique()
#     for world in worlds:
#         wdf = df[df["world"] == world]
#         ax.plot(wdf["episode_num"], wdf["linguistic_surprise"],
#                 label=WORLD_LABELS.get(world, world), linewidth=2, marker="o", markersize=4)

#     ax.set_xlabel("Episode", fontsize=12)
#     ax.set_ylabel("Linguistic Surprise (embedding distance from prev hypothesis)", fontsize=11)
#     ax.set_title(f"Hypothesis Revision Intensity — {AGENT_LABELS.get(agent, agent)}",
#                  fontsize=13, fontweight="bold")
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#     sns.despine(ax=ax)

#     if save:
#         path = RESULTS_DIR / f"linguistic_surprise_{agent}.png"
#         fig.savefig(path, dpi=150, bbox_inches="tight")
#         plt.close(fig)
#         print(f"Saved: {path}")
#         return str(path)
#     return fig


# def generate_all_plots(run_id: str = None):
#     """Generate all plots from current database state."""
#     from measurement.database import get_episodes_df, get_hypotheses_df, get_novelty_df

#     print("Generating all plots...")
#     episodes_df = get_episodes_df(run_id)
#     hypotheses_df = get_hypotheses_df(run_id)
#     novelty_df = get_novelty_df(run_id)

#     paths = []

#     if not episodes_df.empty:
#         p = plot_all_adaptation_curves(episodes_df)
#         if p: paths.append(p)

#     if not hypotheses_df.empty:
#         for agent in hypotheses_df["agent"].unique():
#             for world in hypotheses_df["world"].unique():
#                 p = plot_hypothesis_trajectory(hypotheses_df, agent=agent, world=world)
#                 if p: paths.append(p)

#         p = plot_boden_distribution(hypotheses_df)
#         if p: paths.append(p)

#     if not novelty_df.empty:
#         p = plot_strategy_heatmap(novelty_df)
#         if p: paths.append(p)

#         for agent in novelty_df["agent"].unique():
#             p = plot_linguistic_surprise(novelty_df, agent=agent)
#             if p: paths.append(p)

#     print(f"\nGenerated {len(paths)} plots in {RESULTS_DIR}")
#     return paths


# if __name__ == "__main__":
#     generate_all_plots()

"""
plots.py — Fixed version
All matplotlib/seaborn figures for the paper and demo.

Fixes applied:
- Boden taxonomy now shows breakdown per WORLD (not just per agent)
- UMAP guard: skips worlds with < 3 hypotheses instead of crashing
- generate_all_plots now generates hypothesis chain table for ALL worlds
- Key comparison generated for EVERY world that has both V1 and V3
- Phase performance bars generated per world
- Summary box in comparison plot uses final mean reward (not episodes)
- Filters out "Unable to parse" failed LLM calls from all plots
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import seaborn as sns
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

AGENT_COLORS = {
    "v1_dqn":            "#E74C3C",
    "v2_rule_follower":  "#3498DB",
    "v3_scientist":      "#2ECC71",
    "v4_novelty_seeker": "#9B59B6",
}
AGENT_LABELS = {
    "v1_dqn":            "V1: Silent Instinct (DQN)",
    "v2_rule_follower":  "V2: Rule Follower",
    "v3_scientist":      "V3: Scientist",
    "v4_novelty_seeker": "V4: Novelty Seeker",
}
WORLD_LABELS = {
    "baseline": "A: Baseline",
    "inverted": "B: Inverted Gravity",
    "viscous":  "C: Viscous Void",
    "delayed":  "D: Drunken Cart",
}
BODEN_COLORS = {
    "combinational":    "#F39C12",
    "exploratory":      "#27AE60",
    "transformational": "#8E44AD",
    "unknown":          "#95A5A6",
}


def _rolling_mean(series, window=10):
    return pd.Series(series).rolling(window=window, min_periods=1).mean()


def _clean_hypotheses(df):
    """Filter out failed LLM parse attempts."""
    return df[~df["hypothesis_text"].str.startswith("Unable to parse", na=False)].copy()


# ── Figure 1: Adaptation Curves ────────────────────────────────────────────

def plot_adaptation_curves(episodes_df, world, save=True, ax=None):
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 6))

    world_df = episodes_df[episodes_df["world"] == world]
    for agent in sorted(world_df["agent"].unique()):
        adf = world_df[world_df["agent"] == agent].sort_values("episode_num")
        rewards = adf["total_reward"].values
        smoothed = _rolling_mean(rewards)
        color = AGENT_COLORS.get(agent, "gray")
        label = AGENT_LABELS.get(agent, agent)
        ax.plot(smoothed, color=color, label=label, linewidth=2)
        ax.fill_between(range(len(rewards)),
                        smoothed - pd.Series(rewards).rolling(10, min_periods=1).std().fillna(0),
                        smoothed + pd.Series(rewards).rolling(10, min_periods=1).std().fillna(0),
                        color=color, alpha=0.12)

    ax.axhline(y=400, color="black", linestyle="--", alpha=0.4, label="Solved threshold (400)")
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Reward (rolling mean, w=10)", fontsize=11)
    ax.set_title(f"Adaptation Curves — {WORLD_LABELS.get(world, world)}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    sns.despine(ax=ax)

    if own_fig and save:
        path = RESULTS_DIR / f"adaptation_curves_{world}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
        return str(path)
    return ax


def plot_all_adaptation_curves(episodes_df, save=True):
    worlds = ["baseline", "inverted", "viscous", "delayed"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    for i, world in enumerate(worlds):
        plot_adaptation_curves(episodes_df, world, save=False, ax=axes[i])
    fig.suptitle("Agent Adaptation Across Physics Worlds", fontsize=16, fontweight="bold")
    plt.tight_layout()
    if save:
        path = RESULTS_DIR / "adaptation_curves_all.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
        return str(path)
    return fig


# ── Figure 2: Hypothesis Trajectory (UMAP) ─────────────────────────────────

def plot_hypothesis_trajectory(hypotheses_df, agent="v3_scientist", world="inverted", save=True):
    import umap as umap_lib

    df = _clean_hypotheses(hypotheses_df[
        (hypotheses_df["agent"] == agent) &
        (hypotheses_df["world"] == world) &
        (hypotheses_df["embedding"].notna())
    ])

    if len(df) < 3:
        print(f"  Skipping UMAP for {agent}/{world} — need ≥3 hypotheses (have {len(df)})")
        return None

    embeddings = np.array([json.loads(e) for e in df["embedding"].values])
    n_neighbors = max(2, min(5, len(df) - 1))
    # reducer = umap_lib.UMAP(n_components=2, random_state=42, min_dist=0.1, n_neighbors=n_neighbors)
    reducer = umap_lib.UMAP(n_components=2, random_state=42, min_dist=0.1, n_neighbors=n_neighbors, init="pca")
    coords = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(len(coords) - 1):
        ax.plot(coords[i:i+2, 0], coords[i:i+2, 1],
                color="gray", alpha=0.4, linewidth=1, zorder=1)

    boden_types = df["boden_type"].fillna("unknown").values
    for i, (x, y) in enumerate(coords):
        btype = boden_types[i]
        color = BODEN_COLORS.get(btype, "#95A5A6")
        ax.scatter(x, y, c=color, s=130, zorder=2, edgecolors="white", linewidth=1.5)
        ax.annotate(f"H{i+1}", (x, y), textcoords="offset points",
                    xytext=(6, 6), fontsize=8, color="gray")

    legend_elements = [mpatches.Patch(facecolor=v, label=k.capitalize())
                       for k, v in BODEN_COLORS.items()]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)
    ax.set_xlabel("UMAP Dimension 1", fontsize=12)
    ax.set_ylabel("UMAP Dimension 2", fontsize=12)
    ax.set_title(
        f"Hypothesis Trajectory in Concept Space\n"
        f"{AGENT_LABELS.get(agent, agent)} | {WORLD_LABELS.get(world, world)}",
        fontsize=13, fontweight="bold"
    )
    ax.grid(True, alpha=0.2)
    sns.despine(ax=ax)

    if save:
        path = RESULTS_DIR / f"hypothesis_trajectory_{agent}_{world}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
        return str(path)
    return fig


# ── Figure 3: Strategy Heatmap ─────────────────────────────────────────────

def plot_strategy_heatmap(novelty_df, save=True):
    pivot = novelty_df.groupby(["agent", "world"])["linguistic_surprise"].mean().unstack(fill_value=0)
    pivot.index = [AGENT_LABELS.get(a, a) for a in pivot.index]
    pivot.columns = [WORLD_LABELS.get(w, w) for w in pivot.columns]

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 3), 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Mean Linguistic Surprise"})
    ax.set_title("Mean Linguistic Surprise per Agent × World\n(higher = more theory revision forced)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("World", fontsize=11)
    ax.set_ylabel("Agent", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save:
        path = RESULTS_DIR / "strategy_heatmap.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
        return str(path)
    return fig


# ── Figure 4a: Boden per Agent ─────────────────────────────────────────────

def plot_boden_distribution(hypotheses_df, save=True):
    """Stacked bar: Boden type proportions per agent (all worlds combined)."""
    df = _clean_hypotheses(hypotheses_df[hypotheses_df["boden_type"].notna()])
    if df.empty:
        print("No Boden data.")
        return None

    counts = df.groupby(["agent", "boden_type"]).size().unstack(fill_value=0)
    counts = counts.div(counts.sum(axis=1), axis=0)
    counts.index = [AGENT_LABELS.get(a, a) for a in counts.index]

    fig, ax = plt.subplots(figsize=(max(8, len(counts) * 3), 6))
    colors = [BODEN_COLORS.get(c, "#95A5A6") for c in counts.columns]
    counts.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Agent", fontsize=12)
    ax.set_ylabel("Proportion of Strategies", fontsize=12)
    ax.set_title("Boden Novelty Type Distribution per Agent\n(all worlds combined)",
                 fontsize=14, fontweight="bold")
    ax.legend(title="Boden Type", loc="upper right")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=15, ha="right")
    sns.despine(ax=ax)
    plt.tight_layout()

    if save:
        path = RESULTS_DIR / "boden_taxonomy.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
        return str(path)
    return fig


# ── Figure 4b: Boden per World ─────────────────────────────────────────────

def plot_boden_by_world(hypotheses_df, save=True):
    """
    NEW: Grouped bar showing Boden type breakdown per world.
    Key question: Does world difficulty/alienness drive different creativity types?
    """
    df = _clean_hypotheses(hypotheses_df[hypotheses_df["boden_type"].notna()])
    if df.empty:
        print("No Boden data for world breakdown.")
        return None

    counts = df.groupby(["world", "boden_type"]).size().unstack(fill_value=0)
    counts = counts.div(counts.sum(axis=1), axis=0)
    counts.index = [WORLD_LABELS.get(w, w) for w in counts.index]

    for btype in ["combinational", "exploratory", "transformational"]:
        if btype not in counts.columns:
            counts[btype] = 0.0
    counts = counts[["combinational", "exploratory", "transformational"]]

    fig, ax = plt.subplots(figsize=(max(8, len(counts) * 2.5), 6))
    colors = [BODEN_COLORS["combinational"], BODEN_COLORS["exploratory"], BODEN_COLORS["transformational"]]
    counts.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Physics World", fontsize=12)
    ax.set_ylabel("Proportion of Strategies", fontsize=12)
    ax.set_title("Boden Novelty Type by Physics World\n(Does world difficulty drive different creativity types?)",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Boden Type", loc="upper right")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    sns.despine(ax=ax)
    plt.tight_layout()

    if save:
        path = RESULTS_DIR / "boden_by_world.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
        return str(path)
    return fig


# ── Figure 5: Linguistic Surprise Over Time ─────────────────────────────────

def plot_linguistic_surprise(novelty_df, agent="v3_scientist", save=True):
    df = novelty_df[novelty_df["agent"] == agent].sort_values("episode_num")
    if df.empty:
        print("No novelty data.")
        return None

    world_colors = {"inverted": "#2980B9", "viscous": "#E67E22",
                    "delayed": "#8E44AD", "baseline": "#27AE60"}

    fig, ax = plt.subplots(figsize=(12, 5))
    for world in df["world"].unique():
        wdf = df[df["world"] == world].sort_values("episode_num")
        color = world_colors.get(world, "gray")
        ax.plot(wdf["episode_num"], wdf["linguistic_surprise"],
                label=WORLD_LABELS.get(world, world), linewidth=2,
                marker="o", markersize=5, color=color)
        ax.fill_between(wdf["episode_num"], 0, wdf["linguistic_surprise"],
                        color=color, alpha=0.08)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Linguistic Surprise\n(embedding distance from prev hypothesis)", fontsize=11)
    ax.set_title(f"Hypothesis Revision Intensity — {AGENT_LABELS.get(agent, agent)}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    sns.despine(ax=ax)
    plt.tight_layout()

    if save:
        path = RESULTS_DIR / f"linguistic_surprise_{agent}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
        return str(path)
    return fig


# ── Figure 6: Hypothesis Chain Table (per world) ───────────────────────────

def plot_hypothesis_chain_table(hypotheses_df, agent="v3_scientist", world="inverted", save=True):
    df = _clean_hypotheses(hypotheses_df[
        (hypotheses_df["agent"] == agent) &
        (hypotheses_df["world"] == world)
    ]).sort_values("episode_num")

    seen, rows = set(), []
    for _, row in df.iterrows():
        key = str(row["hypothesis_text"])[:80]
        if key not in seen:
            seen.add(key)
            rows.append(row)

    if not rows:
        print(f"No hypotheses for {agent}/{world}")
        return None

    unique_h = pd.DataFrame(rows).reset_index(drop=True)
    n = len(unique_h)

    fig, ax = plt.subplots(figsize=(16, max(3, n * 0.9)))
    ax.axis("off")

    for i, row in unique_h.iterrows():
        btype = str(row.get("boden_type") or "unknown").lower()
        dist = float(row.get("embedding_distance") or 0)
        hyp_text = str(row["hypothesis_text"])
        if len(hyp_text) > 130:
            hyp_text = hyp_text[:127] + "..."

        color = BODEN_COLORS.get(btype, "#95A5A6")
        y = 1 - (i + 0.5) / n

        ax.add_patch(mpatches.FancyBboxPatch(
            (0, y - 0.038), 0.11, 0.072,
            boxstyle="round,pad=0.01", color=color,
            transform=ax.transAxes, clip_on=False
        ))
        ax.text(0.055, y, btype.upper(),
                ha="center", va="center", fontsize=7.5,
                color="white", fontweight="bold", transform=ax.transAxes)
        ax.text(0.13, y, f"H{i+1}: {hyp_text}",
                ha="left", va="center", fontsize=9, transform=ax.transAxes)
        ax.text(0.97, y, f"Δ={dist:.3f}",
                ha="right", va="center", fontsize=9,
                color="#666666", transform=ax.transAxes)
        if i < n - 1:
            ax.axhline(y=y - 0.5/n, xmin=0.12, xmax=0.96,
                       color="#e0e0e0", linewidth=0.5)

    world_label = WORLD_LABELS.get(world, world)
    ax.set_title(
        f"Hypothesis Chain — V3 Scientist | {world_label}\n"
        f"Δ = linguistic surprise (embedding distance from previous hypothesis)",
        fontsize=13, fontweight="bold", pad=20
    )
    plt.tight_layout()

    if save:
        path = RESULTS_DIR / f"hypothesis_chain_{world}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
        return str(path)
    return fig


# ── Figure 7: Phase Performance Bars (per world) ───────────────────────────

def plot_phase_performance_bars(episodes_df, hypotheses_df, agent="v3_scientist",
                                 world="inverted", save=True):
    h_df = _clean_hypotheses(hypotheses_df[
        (hypotheses_df["agent"] == agent) &
        (hypotheses_df["world"] == world)
    ]).sort_values("episode_num")

    seen, unique_rows = set(), []
    for _, row in h_df.iterrows():
        key = str(row["hypothesis_text"])[:80]
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)
    if not unique_rows:
        return None

    h_unique = pd.DataFrame(unique_rows).reset_index(drop=True)
    ep_df = episodes_df[
        (episodes_df["agent"] == agent) &
        (episodes_df["world"] == world)
    ].sort_values("episode_num")

    phase_starts = h_unique["episode_num"].tolist() + [ep_df["episode_num"].max() + 1]
    phase_means, phase_labels = [], []
    for i in range(len(phase_starts) - 1):
        phase_eps = ep_df[
            (ep_df["episode_num"] >= phase_starts[i]) &
            (ep_df["episode_num"] < phase_starts[i + 1])
        ]
        if not phase_eps.empty:
            phase_means.append(phase_eps["total_reward"].mean())
            phase_labels.append(f"Phase {i+1}\n(H{i+1})")

    if not phase_means:
        return None

    fig, ax = plt.subplots(figsize=(max(8, len(phase_means) * 1.4), 6))
    max_abs = max(abs(v) for v in phase_means) if phase_means else 1
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(phase_means)))
    bars = ax.bar(phase_labels, phase_means, color=colors, edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, phase_means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_abs * 0.02,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(y=400, color="black", linestyle="--", alpha=0.5, label="Solved threshold (400)")
    ax.set_ylabel("Mean Episode Reward per Phase", fontsize=12)
    world_label = WORLD_LABELS.get(world, world)
    ax.set_title(f"V3 Scientist: Performance per Hypothesis Revision\n{world_label}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25, axis="y")
    sns.despine(ax=ax)
    plt.tight_layout()

    if save:
        path = RESULTS_DIR / f"phase_performance_{world}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
        return str(path)
    return fig


# ── Figure 8: Key Comparison V1 vs V3 per world ────────────────────────────

def plot_key_comparison(episodes_df, hypotheses_df, novelty_df, world="inverted", save=True):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                             gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    world_df = episodes_df[episodes_df["world"] == world]

    final_rewards = {}
    for agent in ["v1_dqn", "v3_scientist"]:
        adf = world_df[world_df["agent"] == agent].sort_values("episode_num").reset_index(drop=True)
        if adf.empty:
            continue
        rewards = adf["total_reward"].values
        smoothed = pd.Series(rewards).rolling(window=15, min_periods=1).mean()
        color = AGENT_COLORS[agent]
        label = AGENT_LABELS[agent]
        ax.plot(smoothed.values, color=color, linewidth=2.5, label=label, zorder=3)
        ax.fill_between(range(len(rewards)),
                        smoothed - pd.Series(rewards).rolling(15, min_periods=1).std().fillna(0),
                        smoothed + pd.Series(rewards).rolling(15, min_periods=1).std().fillna(0),
                        color=color, alpha=0.12, zorder=2)
        tail = smoothed.iloc[-min(50, len(smoothed)):]
        final_rewards[agent] = float(tail.mean())

    # V3 revision markers (on V3 episode axis)
    h_df = _clean_hypotheses(hypotheses_df[
        (hypotheses_df["agent"] == "v3_scientist") &
        (hypotheses_df["world"] == world)
    ]).sort_values("episode_num")

    seen, rev_eps = set(), []
    for _, row in h_df.iterrows():
        key = str(row["hypothesis_text"])[:80]
        if key not in seen:
            seen.add(key)
            rev_eps.append(int(row["episode_num"]))

    v3_ep_series = world_df[world_df["agent"] == "v3_scientist"].sort_values("episode_num")["episode_num"].values
    for i, ep in enumerate(rev_eps[1:], 1):
        pos = int(np.searchsorted(v3_ep_series, ep))
        ax.axvline(x=pos, color="#2ECC71", linestyle="--", linewidth=1, alpha=0.5)
        ymax = ax.get_ylim()[1]
        ax.text(pos + max(len(v3_ep_series) * 0.005, 3), ymax * 0.88,
                f"Rev {i}", fontsize=8, color="#27AE60", alpha=0.9)

    ax.axhline(y=400, color="black", linestyle="--", linewidth=1.2, alpha=0.5,
               label="Solved threshold (400)")

    v1f = final_rewards.get("v1_dqn", 0)
    v3f = final_rewards.get("v3_scientist", 0)
    if v1f > 0 and v3f > 0:
        ratio = v3f / v1f
        summary = f"V1 final mean: {v1f:.0f}\nV3 final mean: {v3f:.0f}\nV3/V1 ratio: {ratio:.1f}x"
    else:
        summary = f"V1 final mean: {v1f:.0f}\nV3 final mean: {v3f:.0f}"

    ax.text(0.98, 0.05, summary, transform=ax.transAxes, fontsize=9,
            va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

    world_label = WORLD_LABELS.get(world, world)
    ax.set_ylabel("Reward (rolling mean, window=15)", fontsize=12)
    ax.set_title(f"V1 vs V3: {world_label}\nDashed green lines = LLM hypothesis revisions",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.2)
    sns.despine(ax=ax)

    ax2 = axes[1]
    n_df = novelty_df[
        (novelty_df["agent"] == "v3_scientist") &
        (novelty_df["world"] == world)
    ].sort_values("episode_num")
    if not n_df.empty:
        ax2.plot(n_df["episode_num"], n_df["linguistic_surprise"],
                 color="#2ECC71", linewidth=2, marker="o", markersize=4)
        ax2.fill_between(n_df["episode_num"], 0, n_df["linguistic_surprise"],
                         color="#2ECC71", alpha=0.15)
    ax2.set_ylabel("Linguistic\nSurprise", fontsize=10)
    ax2.set_xlabel("Episode (V3 axis)", fontsize=11)
    ax2.grid(True, alpha=0.2)
    sns.despine(ax=ax2)

    plt.tight_layout()
    if save:
        path = RESULTS_DIR / f"key_comparison_{world}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
        return str(path)
    return fig

# ── Figure 9: All-Agent Comparison per world ───────────────────────────────

def plot_all_agents_comparison(episodes_df, world="inverted", save=True):
    """
    All 4 agents on one plot per world.
    This is your main result figure once V2 and V4 data exists.
    V1 = no LLM baseline
    V2 = one-shot LLM (no revision)
    V3 = iterative revision
    V4 = iterative revision + novelty pressure
    Gap between lines shows the value of each design choice.
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    world_df = episodes_df[episodes_df["world"] == world]

    agent_order = ["v1_dqn", "v2_rule_follower", "v3_scientist", "v4_novelty_seeker"]
    final_rewards = {}

    for agent in agent_order:
        adf = world_df[world_df["agent"] == agent].sort_values("episode_num").reset_index(drop=True)
        if adf.empty:
            continue
        rewards = adf["total_reward"].values
        smoothed = pd.Series(rewards).rolling(window=15, min_periods=1).mean()
        color = AGENT_COLORS.get(agent, "gray")
        label = AGENT_LABELS.get(agent, agent)
        ax.plot(smoothed.values, color=color, linewidth=2.5, label=label, zorder=3)
        ax.fill_between(
            range(len(rewards)),
            smoothed - pd.Series(rewards).rolling(15, min_periods=1).std().fillna(0),
            smoothed + pd.Series(rewards).rolling(15, min_periods=1).std().fillna(0),
            color=color, alpha=0.10, zorder=2
        )
        final_rewards[agent] = float(smoothed.iloc[-min(50, len(smoothed)):].mean())

    ax.axhline(y=400, color="black", linestyle="--", linewidth=1.2,
               alpha=0.5, label="Solved threshold (400)")

    # Summary box
    lines = [f"{AGENT_LABELS.get(a, a)}: {v:.0f}" 
             for a, v in final_rewards.items()]
    ax.text(0.98, 0.05, "\n".join(lines), transform=ax.transAxes,
            fontsize=9, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

    world_label = WORLD_LABELS.get(world, world)
    ax.set_ylabel("Reward (rolling mean, window=15)", fontsize=12)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_title(f"All Agents: {world_label}\nV1=no LLM | V2=one-shot | V3=revision | V4=novelty pressure",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.2)
    sns.despine(ax=ax)
    plt.tight_layout()

    if save:
        path = RESULTS_DIR / f"all_agents_{world}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
        return str(path)
    return fig


# ── Master generate ────────────────────────────────────────────────────────

def generate_all_plots(run_id=None):
    from measurement.database import get_episodes_df, get_hypotheses_df, get_novelty_df

    print("Generating all plots...")
    episodes_df   = get_episodes_df(run_id)
    hypotheses_df = get_hypotheses_df(run_id)
    novelty_df    = get_novelty_df(run_id)

    paths = []

    # 1. Adaptation curves grid
    if not episodes_df.empty:
        p = plot_all_adaptation_curves(episodes_df)
        if p: paths.append(p)

    # 2. Per-world plots for LLM agents (V3 and V4 both have hypotheses)
    LLM_AGENTS = ["v3_scientist", "v4_novelty_seeker"]
    if not hypotheses_df.empty:
        for agent in LLM_AGENTS:
            agent_worlds = hypotheses_df[hypotheses_df["agent"] == agent]["world"].unique()
            for world in agent_worlds:
                p = plot_hypothesis_trajectory(hypotheses_df, agent=agent, world=world)
                if p: paths.append(p)
                p = plot_hypothesis_chain_table(hypotheses_df, agent=agent, world=world)
                if p: paths.append(p)
                if not episodes_df.empty:
                    p = plot_phase_performance_bars(episodes_df, hypotheses_df,
                                                    agent=agent, world=world)
                    if p: paths.append(p)

        p = plot_boden_distribution(hypotheses_df)
        if p: paths.append(p)
        p = plot_boden_by_world(hypotheses_df)
        if p: paths.append(p)

    # 3. Heatmap + linguistic surprise
    if not novelty_df.empty:
        p = plot_strategy_heatmap(novelty_df)
        if p: paths.append(p)
        for agent in novelty_df["agent"].unique():
            p = plot_linguistic_surprise(novelty_df, agent=agent)
            if p: paths.append(p)

    # 4. Key comparisons for every world with BOTH agents
    if not episodes_df.empty and not hypotheses_df.empty and not novelty_df.empty:
        v1_worlds = set(episodes_df[episodes_df["agent"] == "v1_dqn"]["world"].unique())
        v3_worlds = set(episodes_df[episodes_df["agent"] == "v3_scientist"]["world"].unique())
        for world in sorted(v1_worlds & v3_worlds):
            p = plot_key_comparison(episodes_df, hypotheses_df, novelty_df, world=world)
            if p: paths.append(p)
    
    # 5. All-agents comparison for every world that has data
    all_worlds = episodes_df["world"].unique()
    for world in sorted(all_worlds):
        p = plot_all_agents_comparison(episodes_df, world=world)
        if p: paths.append(p)

    print(f"\nGenerated {len(paths)} plots in {RESULTS_DIR}")
    return paths


if __name__ == "__main__":
    generate_all_plots()