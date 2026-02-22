"""
make_comparison_plot.py
Run this standalone to generate the key comparison figures.
Place in your project root and run: python make_comparison_plot.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from measurement.database import get_episodes_df, get_hypotheses_df, get_novelty_df, init_db

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

AGENT_COLORS = {
    "v1_dqn":          "#E74C3C",
    "v3_scientist":    "#2ECC71",
}
AGENT_LABELS = {
    "v1_dqn":          "V1: Silent Instinct (DQN only)",
    "v3_scientist":    "V3: Scientist (LLM + RL)",
}


def plot_focused_comparison(episodes_df, hypotheses_df, world="inverted"):
    """
    The key comparison figure: V1 vs V3 on inverted gravity.
    Shows phase boundaries, episode counts to solve, and performance gap clearly.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                             gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    world_df = episodes_df[episodes_df["world"] == world].copy()

    solve_episodes = {}

    for agent in ["v1_dqn", "v3_scientist"]:
        adf = world_df[world_df["agent"] == agent].sort_values("episode_num").reset_index(drop=True)
        if adf.empty:
            continue
        rewards = adf["total_reward"].values
        smoothed = pd.Series(rewards).rolling(window=15, min_periods=1).mean()
        color = AGENT_COLORS[agent]
        label = AGENT_LABELS[agent]

        ax.plot(smoothed.values, color=color, linewidth=2.5, label=label, zorder=3)
        ax.fill_between(
            range(len(rewards)),
            smoothed - pd.Series(rewards).rolling(15, min_periods=1).std().fillna(0),
            smoothed + pd.Series(rewards).rolling(15, min_periods=1).std().fillna(0),
            color=color, alpha=0.12, zorder=2
        )

        # Find episode where rolling mean first exceeds 400
        solved_at = next((i for i, v in enumerate(smoothed) if v > 400), None)
        if solved_at:
            solve_episodes[agent] = solved_at
            ax.axvline(x=solved_at, color=color, linestyle=":", linewidth=1.5, alpha=0.7)
            ax.annotate(f"Solved\n(ep {solved_at})",
                        xy=(solved_at, 400),
                        xytext=(solved_at + len(rewards)*0.03, 450),
                        fontsize=9, color=color,
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.2))

    # Add V3 phase revision boundaries
    h_df = hypotheses_df[
        (hypotheses_df["agent"] == "v3_scientist") &
        (hypotheses_df["world"] == world)
    ].sort_values("episode_num")

    # Filter out failed LLM calls (those that couldn't parse responses)
    h_df = h_df[~h_df["hypothesis_text"].astype(str).str.startswith("Unable to parse")]

    revision_episodes = h_df["episode_num"].values[1:]  # skip H1 (initial)
    for i, ep in enumerate(revision_episodes):
        ax.axvline(x=ep, color="#2ECC71", linestyle="--", linewidth=1, alpha=0.45)
        ax.text(ep + 10, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 200,
                f"Rev {i+1}", fontsize=8, color="#27AE60", alpha=0.8)

    ax.axhline(y=400, color="black", linestyle="--", linewidth=1.2,
               alpha=0.5, label="Solved threshold (400)", zorder=1)
    ax.set_ylabel("Reward (rolling mean, window=15)", fontsize=12)
    ax.set_title("V1 vs V3: Inverted Gravity World\nDashed green lines = LLM hypothesis revisions",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.25)
    sns.despine(ax=ax)

    # Performance summary text box
    summary_lines = []
    
    # Calculate final mean rewards for each agent (last 50 episodes)
    final_rewards = {}
    for agent in ["v1_dqn", "v3_scientist"]:
        agent_df = world_df[world_df["agent"] == agent].sort_values("episode_num")
        if not agent_df.empty:
            final_50 = agent_df.tail(50)["total_reward"].mean()
            final_rewards[agent] = final_50
    
    if len(final_rewards) == 2:
        r_v1 = final_rewards["v1_dqn"]
        r_v3 = final_rewards["v3_scientist"]
        
        # Calculate stability (std dev of final 50 episodes)
        v1_stability = world_df[world_df["agent"] == "v1_dqn"].tail(50)["total_reward"].std()
        v3_stability = world_df[world_df["agent"] == "v3_scientist"].tail(50)["total_reward"].std()
        
        v1_status = "unstable" if v1_stability > 100 else "stable"
        v3_status = "stable" if v3_stability < 100 else "unstable"
        
        summary_lines.append(f"V1 final mean reward: {r_v1:.0f} ({v1_status})")
        summary_lines.append(f"V3 final mean reward: {r_v3:.0f} ({v3_status})")
    else:
        for agent, ep in solve_episodes.items():
            summary_lines.append(f"{AGENT_LABELS[agent]}: solved at episode {ep}")
    
    if summary_lines:
        ax.text(0.98, 0.05, "\n".join(summary_lines),
                transform=ax.transAxes, fontsize=9,
                verticalalignment="bottom", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    # ── Bottom panel: Linguistic surprise over episodes ────────────────────
    ax2 = axes[1]
    novelty_df = get_novelty_df()
    n_df = novelty_df[
        (novelty_df["agent"] == "v3_scientist") &
        (novelty_df["world"] == world)
    ].sort_values("episode_num")

    if not n_df.empty:
        ax2.plot(n_df["episode_num"], n_df["linguistic_surprise"],
                 color="#2ECC71", linewidth=2, marker="o", markersize=5,
                 label="Linguistic Surprise (V3)")
        ax2.fill_between(n_df["episode_num"], 0, n_df["linguistic_surprise"],
                         color="#2ECC71", alpha=0.15)
        ax2.set_ylabel("Linguistic\nSurprise", fontsize=10)
        ax2.set_xlabel("Episode", fontsize=12)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.25)
        sns.despine(ax=ax2)

    plt.tight_layout()
    path = RESULTS_DIR / "key_comparison_inverted.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return str(path)


def plot_phase_performance_bars(episodes_df, hypotheses_df, world="inverted"):
    """
    Bar chart showing mean reward per V3 phase.
    Makes the 1.4 → 16.9 → 45.1 → 302.6 → 1134.2 progression clearly visible.
    """
    h_df = hypotheses_df[
        (hypotheses_df["agent"] == "v3_scientist") &
        (hypotheses_df["world"] == world)
    ].sort_values("episode_num").reset_index(drop=True)

    # Filter out failed LLM calls (those that couldn't parse responses)
    h_df = h_df[~h_df["hypothesis_text"].astype(str).str.startswith("Unable to parse")].reset_index(drop=True)

    ep_df = episodes_df[
        (episodes_df["agent"] == "v3_scientist") &
        (episodes_df["world"] == world)
    ].sort_values("episode_num")

    if h_df.empty or ep_df.empty:
        print("No data for phase performance bars.")
        return

    # Define phase boundaries from hypothesis episode numbers
    phase_starts = h_df["episode_num"].tolist() + [ep_df["episode_num"].max() + 1]
    phase_means = []
    phase_labels = []

    for i in range(len(phase_starts) - 1):
        start = phase_starts[i]
        end = phase_starts[i + 1]
        phase_eps = ep_df[
            (ep_df["episode_num"] >= start) &
            (ep_df["episode_num"] < end)
        ]
        if not phase_eps.empty:
            phase_means.append(phase_eps["total_reward"].mean())
            phase_labels.append(f"Phase {i+1}\n(H{i+1})")

    if not phase_means:
        print("Could not compute phase means.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(phase_means)))
    bars = ax.bar(phase_labels, phase_means, color=colors, edgecolor="white",
                  linewidth=1.5, zorder=3)

    # Annotate each bar with the value
    for bar, val in zip(bars, phase_means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(phase_means) * 0.02,
                f"{val:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.axhline(y=400, color="black", linestyle="--", alpha=0.5, label="Solved threshold")
    ax.set_ylabel("Mean Episode Reward per Phase", fontsize=12)
    ax.set_title("V3 Scientist: Performance Improvement per Hypothesis Revision\nInverted Gravity World",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25, axis="y")
    sns.despine(ax=ax)

    plt.tight_layout()
    path = RESULTS_DIR / "phase_performance_bars.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return str(path)


def plot_hypothesis_summary_table(hypotheses_df, world="inverted"):
    """
    Visual table of all hypotheses with Boden type and linguistic surprise.
    Good for presentations — shows the actual reasoning chain.
    """
    h_df = hypotheses_df[
        (hypotheses_df["agent"] == "v3_scientist") &
        (hypotheses_df["world"] == world)
    ].sort_values("episode_num").reset_index(drop=True)

    if h_df.empty:
        return

    # Filter out failed LLM calls (those that couldn't parse responses)
    h_df = h_df[~h_df["hypothesis_text"].astype(str).str.startswith("Unable to parse")].reset_index(drop=True)

    if h_df.empty:
        return

    # Keep only the revision hypotheses (one per phase), not all logged ones
    # We identify them by looking for the biggest distance jumps
    # Filter to unique hypotheses by looking at hypothesis_text changes
    seen = set()
    rows = []
    for _, row in h_df.iterrows():
        key = row["hypothesis_text"][:80]
        if key not in seen:
            seen.add(key)
            rows.append(row)
    unique_h = pd.DataFrame(rows).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(16, max(4, len(unique_h) * 1.2)))
    ax.axis("off")

    BODEN_COLORS_MAP = {
        "combinational": "#F39C12",
        "exploratory": "#27AE60",
        "transformational": "#8E44AD",
    }

    for i, row in unique_h.iterrows():
        btype = str(row.get("boden_type") or "unknown")
        dist = float(row.get("embedding_distance") or 0)
        hyp_text = str(row["hypothesis_text"])
        if len(hyp_text) > 120:
            hyp_text = hyp_text[:117] + "..."

        color = BODEN_COLORS_MAP.get(btype, "#95A5A6")
        y = 1 - (i + 0.5) / len(unique_h)

        # Colored Boden badge
        ax.add_patch(mpatches.FancyBboxPatch(
            (0, y - 0.04), 0.12, 0.08,
            boxstyle="round,pad=0.01", color=color, transform=ax.transAxes
        ))
        ax.text(0.06, y, btype.upper(),
                ha="center", va="center", fontsize=8, color="white",
                fontweight="bold", transform=ax.transAxes)

        # Hypothesis text
        ax.text(0.14, y, f"H{i+1}: {hyp_text}",
                ha="left", va="center", fontsize=9,
                transform=ax.transAxes)

        # Linguistic surprise score
        ax.text(0.97, y, f"Δ={dist:.3f}",
                ha="right", va="center", fontsize=9, color="gray",
                transform=ax.transAxes)

    ax.set_title(
        "Hypothesis Chain — V3 Scientist | Inverted Gravity\n"
        "Δ = linguistic surprise (embedding distance from previous hypothesis)",
        fontsize=13, fontweight="bold", pad=20
    )
    plt.tight_layout()
    path = RESULTS_DIR / "hypothesis_chain_table.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return str(path)


if __name__ == "__main__":
    init_db()
    episodes_df = get_episodes_df()
    hypotheses_df = get_hypotheses_df()

    if episodes_df.empty:
        print("No data yet. Run experiments first.")
    else:
        print("Generating comparison figures...")
        plot_focused_comparison(episodes_df, hypotheses_df, world="inverted")
        plot_phase_performance_bars(episodes_df, hypotheses_df, world="inverted")
        plot_hypothesis_summary_table(hypotheses_df, world="inverted")
        print(f"\nDone. Check the results/ folder.")
