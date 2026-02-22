"""
dashboard.py
Streamlit live demo — the workshop-facing interface.

Run with: streamlit run visualization/dashboard.py

Features:
- Select world + agent and watch training live
- See LLM hypothesis stream in real time
- Live reward curve updating per episode
- Hypothesis embedding distance gauge
- Full experiment results browser
"""

import sys
import time
import threading
import queue
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environments.cartpole_variants import BaseCartPole, get_all_worlds
from measurement.database import init_db, get_episodes_df, get_hypotheses_df, get_novelty_df


# ── Page Config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Novelty Benchmark",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS Styling ────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .hypothesis-box {
        background: #1a1a2e;
        border-left: 4px solid #00d4aa;
        border-radius: 4px;
        padding: 12px 16px;
        font-family: 'Courier New', monospace;
        font-size: 13px;
        color: #e0e0e0;
        margin: 8px 0;
        white-space: pre-wrap;
    }
    .boden-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
        margin: 2px;
    }
    .boden-combinational   { background: #F39C12; color: white; }
    .boden-exploratory     { background: #27AE60; color: white; }
    .boden-transformational { background: #8E44AD; color: white; }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧪 Novelty Benchmark")
    st.caption("Adaptive Hypothesis Revision in LLM+RL Agents")
    st.divider()

    page = st.radio("Navigate", [
        "🏠 Overview",
        "🔬 Live Experiment",
        "📊 Results Browser",
        "🗺️ Hypothesis Map",
        "📖 About"
    ])


# ── Helper: World descriptions ─────────────────────────────────────────────

WORLD_INFO = {
    "baseline": {
        "icon": "🌍",
        "title": "World A: Baseline",
        "description": "Standard CartPole. Gravity = 9.8 m/s². This is the 'known universe' — LLMs have seen this solved thousands of times.",
        "challenge": "Low",
        "color": "#27AE60"
    },
    "inverted": {
        "icon": "🙃",
        "title": "World B: Inverted Gravity",
        "description": "Gravity = -9.8 m/s². The pole 'falls' upward. Standard balancing strategies fail completely.",
        "challenge": "High",
        "color": "#E74C3C"
    },
    "viscous": {
        "icon": "🌊",
        "title": "World C: Viscous Void",
        "description": "Gravity = 0, high viscosity. The pole resists movement and stays where placed. A completely alien control problem.",
        "challenge": "High",
        "color": "#3498DB"
    },
    "delayed": {
        "icon": "⏰",
        "title": "World D: Drunken Cart",
        "description": "Actions delayed by 3 frames. Cause and effect are decoupled. Intuitive strategies work backwards.",
        "challenge": "Medium-High",
        "color": "#9B59B6"
    }
}


# ── Pages ──────────────────────────────────────────────────────────────────

def page_overview():
    st.title("Novelty Benchmark: Can LLMs Innovate?")
    st.markdown("""
    This framework tests whether language-mediated reasoning produces **genuinely different adaptation strategies**
    compared to pure reinforcement learning — using Boden's creativity taxonomy as our measurement instrument.
    """)

    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("#### 🌍 Four Worlds")
        st.markdown("Physics-deformed CartPole environments where standard strategies fail")
    with col2:
        st.markdown("#### 🤖 Four Agents")
        st.markdown("From pure DQN to iterative LLM+RL feedback loops")
    with col3:
        st.markdown("#### 📐 Boden Taxonomy")
        st.markdown("Combinational → Exploratory → Transformational novelty")
    with col4:
        st.markdown("#### 🗺️ Hypothesis Maps")
        st.markdown("UMAP visualization of how theories evolve through concept space")

    st.divider()
    st.subheader("The Four Physics Worlds")

    cols = st.columns(4)
    for i, (world_id, info) in enumerate(WORLD_INFO.items()):
        with cols[i]:
            st.markdown(f"### {info['icon']} {info['title'].split(':')[1].strip()}")
            st.markdown(info["description"])
            st.markdown(f"**Challenge:** {info['challenge']}")

    st.divider()
    st.subheader("The Four Agents")

    agents = [
        ("V1", "Silent Instinct", "Pure DQN", "Raw pattern matching. No reasoning.", "#E74C3C"),
        ("V2", "Rule Follower", "LLM → Reward → RL", "One-shot LLM guess at reward function.", "#3498DB"),
        ("V3", "Scientist", "RL ↔ LLM loop", "Iterative hypothesis revision from failure.", "#2ECC71"),
        ("V4", "Novelty Seeker", "V3 + novelty pressure", "Forces exploration of non-textbook strategies.", "#9B59B6"),
    ]

    cols = st.columns(4)
    for i, (version, name, stack, description, color) in enumerate(agents):
        with cols[i]:
            st.markdown(f"### {version}: {name}")
            st.markdown(f"**Stack:** {stack}")
            st.markdown(description)


def page_live_experiment():
    st.title("🔬 Live Experiment")
    st.markdown("Run an agent in a physics world and watch it reason in real time.")

    col_config, col_live = st.columns([1, 2])

    with col_config:
        st.subheader("Configuration")
        world = st.selectbox("Physics World", get_all_worlds(),
                             format_func=lambda w: WORLD_INFO[w]["title"])
        agent_choice = st.selectbox("Agent", ["v3_scientist", "v1_dqn"],
                                    format_func=lambda a: {"v3_scientist": "V3: Scientist",
                                                           "v1_dqn": "V1: Silent Instinct (baseline)"}[a])
        n_revisions = st.slider("Max Hypothesis Revisions (V3 only)", 1, 8, 3)
        timesteps = st.slider("Timesteps per Phase", 2000, 15000, 5000, step=1000)

        world_info = WORLD_INFO[world]
        st.markdown(f"""
        **{world_info['icon']} {world_info['title']}**
        {world_info['description']}
        Challenge: **{world_info['challenge']}**
        """)

        run_btn = st.button("▶ Run Experiment", type="primary", use_container_width=True)

    with col_live:
        st.subheader("Live Output")

        if run_btn:
            reward_placeholder = st.empty()
            hypothesis_placeholder = st.empty()
            status_placeholder = st.empty()
            log_placeholder = st.empty()

            status_placeholder.info("🔄 Starting experiment...")

            if agent_choice == "v1_dqn":
                from agents.v1_dqn import run as run_v1
                status_placeholder.info("🤖 V1 DQN training (no language reasoning)...")
                result = run_v1(world=world, total_timesteps=timesteps)
                status_placeholder.success(f"✅ Complete! Final mean reward: {result['final_mean_reward']:.1f}")

            elif agent_choice == "v3_scientist":
                from agents.v3_scientist import run as run_v3, ScientistAgent
                from environments.cartpole_variants import make_world
                from groq import Groq
                import os
                from dotenv import load_dotenv
                load_dotenv()

                status_placeholder.info("🔬 V3 Scientist: collecting telemetry...")

                # We run this synchronously and update UI between phases
                import uuid
                from measurement.database import init_db, log_experiment
                init_db()
                run_id = f"v3_{world}_{uuid.uuid4().hex[:8]}"
                log_experiment(run_id, "v3_scientist", world, {})

                agent = ScientistAgent(world=world, run_id=run_id)
                base_env = make_world(world)
                response = agent.form_initial_hypothesis(base_env)
                base_env.close()

                hypothesis_placeholder.markdown(f"""
                <div class="hypothesis-box">
                <b>🧠 Initial Hypothesis (Phase 1)</b><br><br>
                {agent.current_hypothesis}
                </div>
                """, unsafe_allow_html=True)

                from stable_baselines3 import DQN
                from agents.v3_scientist import CustomRewardWrapper
                from measurement.database import log_episode

                all_rewards = []

                for phase in range(n_revisions + 1):
                    status_placeholder.info(f"⚙️ Phase {phase+1}/{n_revisions+1}: Training RL agent...")

                    env = make_world(world)
                    wrapped = CustomRewardWrapper(env, agent.current_reward_fn_code)

                    phase_rewards = []
                    last_traj = []
                    current_traj = []

                    from stable_baselines3.common.callbacks import BaseCallback

                    class LiveCallback(BaseCallback):
                        def __init__(self):
                            super().__init__()
                            self._ep_r = 0
                            self._ep_n = len(all_rewards)

                        def _on_step(self):
                            obs = self.locals["new_obs"][0]
                            act = self.locals["actions"][0]
                            self._ep_r += self.locals["rewards"][0]
                            current_traj.append((obs.copy(), int(act)))

                            if self.locals["dones"][0]:
                                phase_rewards.append(self._ep_r)
                                all_rewards.append(self._ep_r)
                                log_episode(run_id, "v3_scientist", world, self._ep_n,
                                            self._ep_r, len(current_traj), True)
                                last_traj.clear()
                                last_traj.extend(current_traj)
                                current_traj.clear()
                                self._ep_r = 0
                                self._ep_n += 1

                                # Update chart every 5 episodes
                                if len(all_rewards) % 5 == 0:
                                    fig, ax = plt.subplots(figsize=(8, 3))
                                    rolling = pd.Series(all_rewards).rolling(10, min_periods=1).mean()
                                    ax.plot(rolling, color="#2ECC71", linewidth=2)
                                    ax.axhline(400, color="black", linestyle="--", alpha=0.4)
                                    ax.set_xlabel("Episode")
                                    ax.set_ylabel("Reward (rolling mean)")
                                    ax.set_title(f"V3 Scientist — {WORLD_INFO[world]['title']}")
                                    plt.tight_layout()
                                    reward_placeholder.pyplot(fig)
                                    plt.close(fig)

                            return True

                    cb = LiveCallback()
                    model = DQN("MlpPolicy", wrapped, learning_rate=1e-3, buffer_size=10_000,
                                learning_starts=500, batch_size=64, gamma=0.99,
                                exploration_fraction=0.3, exploration_final_eps=0.05,
                                verbose=0, seed=42)
                    model.learn(total_timesteps=timesteps, callback=cb)
                    wrapped.close()

                    # Score hypothesis
                    scores = agent.score_and_log_hypothesis(len(all_rewards), last_traj)
                    btype = scores.get("boden_type", "unknown")

                    mean_r = np.mean(phase_rewards) if phase_rewards else 0

                    # Update hypothesis display
                    badge_class = f"boden-{btype}" if btype in ["combinational", "exploratory", "transformational"] else ""
                    hypothesis_placeholder.markdown(f"""
                    <div class="hypothesis-box">
                    <b>🧠 Phase {phase+1} Hypothesis</b>
                    &nbsp;<span class="boden-badge {badge_class}">{btype.upper()}</span><br><br>
                    {agent.current_hypothesis}<br><br>
                    <small>📊 Mean reward this phase: {mean_r:.1f} | 
                    💡 Linguistic surprise: {scores['embedding_distance']:.4f}</small>
                    </div>
                    """, unsafe_allow_html=True)

                    if mean_r > 400:
                        status_placeholder.success(f"✅ SOLVED in phase {phase+1}! Mean reward: {mean_r:.1f}")
                        break

                    if phase < n_revisions:
                        status_placeholder.info(f"🔄 Revising hypothesis (mean reward was {mean_r:.1f})...")
                        env2 = make_world(world)
                        agent.revise_hypothesis(env2, phase_rewards, last_traj)
                        env2.close()

                        hypothesis_placeholder.markdown(f"""
                        <div class="hypothesis-box">
                        <b>🔄 Revised Hypothesis (Phase {phase+2})</b><br><br>
                        {agent.current_hypothesis}
                        </div>
                        """, unsafe_allow_html=True)

                if all_rewards:
                    final_mean = np.mean(all_rewards[-50:])
                    status_placeholder.success(
                        f"✅ Experiment complete! Final mean reward: {final_mean:.1f} | "
                        f"Revisions: {agent.revision_count} | "
                        f"Total episodes: {len(all_rewards)}"
                    )
        else:
            st.info("Configure your experiment on the left and press **Run Experiment** to begin.")
            st.markdown("""
            **What you'll see:**
            - 📈 Live reward curve updating every 5 episodes  
            - 🧠 LLM hypothesis streaming after each phase
            - 🏷️ Boden novelty type badge for each strategy
            - 💡 Linguistic surprise score (how much the theory changed)
            """)


def page_results_browser():
    st.title("📊 Results Browser")

    init_db()
    episodes_df = get_episodes_df()
    hypotheses_df = get_hypotheses_df()
    novelty_df = get_novelty_df()

    if episodes_df.empty:
        st.info("No experiment results yet. Run an experiment first!")
        return

    # Summary stats
    st.subheader("Experiment Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Episodes", len(episodes_df))
    with col2:
        st.metric("Unique Runs", episodes_df["run_id"].nunique())
    with col3:
        st.metric("Hypotheses Generated", len(hypotheses_df))
    with col4:
        agents_run = episodes_df["agent"].nunique()
        st.metric("Agents Tested", agents_run)

    st.divider()

    # Adaptation curves
    st.subheader("Adaptation Curves")
    world_sel = st.selectbox("Select World", get_all_worlds(),
                             format_func=lambda w: WORLD_INFO[w]["title"])

    from visualization.plots import plot_adaptation_curves
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_adaptation_curves(episodes_df, world_sel, save=False, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # Hypothesis table
    if not hypotheses_df.empty:
        st.subheader("Hypothesis Log")
        display_cols = ["agent", "world", "episode_num", "hypothesis_text",
                        "boden_type", "embedding_distance"]
        available = [c for c in display_cols if c in hypotheses_df.columns]
        st.dataframe(
            hypotheses_df[available].rename(columns={
                "hypothesis_text": "hypothesis",
                "embedding_distance": "surprise"
            }),
            use_container_width=True,
            height=300
        )

    st.divider()

    # Boden distribution
    if not hypotheses_df.empty and "boden_type" in hypotheses_df.columns:
        st.subheader("Boden Taxonomy Distribution")
        from visualization.plots import plot_boden_distribution
        fig = plot_boden_distribution(hypotheses_df, save=False)
        if fig:
            st.pyplot(fig)
            plt.close(fig)


def page_hypothesis_map():
    st.title("🗺️ Hypothesis Trajectory Map")
    st.markdown("UMAP projection showing how the LLM's theory evolves through concept space.")

    init_db()
    hypotheses_df = get_hypotheses_df()

    if hypotheses_df.empty or "embedding" not in hypotheses_df.columns:
        st.info("No hypothesis embeddings yet. Run a V3 or V4 experiment first.")
        return

    col1, col2 = st.columns(2)
    with col1:
        agents = hypotheses_df["agent"].unique().tolist()
        agent_sel = st.selectbox("Agent", agents)
    with col2:
        worlds = hypotheses_df["world"].unique().tolist()
        world_sel = st.selectbox("World", worlds, format_func=lambda w: WORLD_INFO.get(w, {}).get("title", w))

    from visualization.plots import plot_hypothesis_trajectory
    fig = plot_hypothesis_trajectory(hypotheses_df, agent=agent_sel, world=world_sel, save=False)
    if fig:
        st.pyplot(fig)
        plt.close(fig)

        # Show hypothesis timeline below
        df_filtered = hypotheses_df[
            (hypotheses_df["agent"] == agent_sel) &
            (hypotheses_df["world"] == world_sel)
        ].sort_values("episode_num")

        st.subheader("Hypothesis Timeline")
        for i, row in enumerate(df_filtered.itertuples()):
            btype = getattr(row, "boden_type", "unknown") or "unknown"
            dist = getattr(row, "embedding_distance", 0) or 0
            badge_class = f"boden-{btype}" if btype in ["combinational", "exploratory", "transformational"] else ""
            st.markdown(f"""
            <div class="hypothesis-box">
            <b>H{i+1}</b> (Episode {row.episode_num})
            &nbsp;<span class="boden-badge {badge_class}">{btype.upper()}</span>
            &nbsp;<small>Surprise: {dist:.4f}</small><br><br>
            {row.hypothesis_text}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Not enough hypotheses for UMAP (need at least 3).")


def page_about():
    st.title("📖 About This Project")
    st.markdown("""
    ## Novelty Benchmark

    A framework for measuring and visualizing **adaptive hypothesis revision** in LLM+RL agents,
    with explicit mapping to Margaret Boden's novelty taxonomy.

    ### Research Question
    Does language-mediated reasoning produce measurably different adaptation trajectories
    compared to pure reinforcement learning in novel physics environments?

    ### Boden's Three Types of Creativity
    - **Combinational**: Combines existing known strategies in a new way
    - **Exploratory**: Explores the edges of an existing conceptual space
    - **Transformational**: Genuinely reframes the problem or invents a new framework

    ### The Key Contribution
    Rather than claiming to prove or disprove novelty in LLMs, this project provides:
    1. A **replicable evaluation protocol** for distinguishing retrieval from hypothesis revision
    2. A **measurement instrument** (Boden-labeled hypothesis embeddings) others can build on
    3. **Behavioral fingerprinting** to detect when different reasoning produces structurally similar outcomes

    ### Stack
    Gymnasium · Stable Baselines3 · Groq (Llama 3) · sentence-transformers · UMAP · Streamlit

    ---
    *Research Fellowship Project · 2024*
    """)


# ── Router ─────────────────────────────────────────────────────────────────

if page == "🏠 Overview":
    page_overview()
elif page == "🔬 Live Experiment":
    page_live_experiment()
elif page == "📊 Results Browser":
    page_results_browser()
elif page == "🗺️ Hypothesis Map":
    page_hypothesis_map()
elif page == "📖 About":
    page_about()
