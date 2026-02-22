"""
run_experiment.py
Master experiment runner. Runs all agent/world combinations and generates plots.

Usage:
    python run_experiment.py                          # Full experiment
    python run_experiment.py --agent v3 --world inverted --timesteps 50000
    python run_experiment.py --agent v2 --world inverted --timesteps 50000
    python run_experiment.py --agent v4 --world inverted --timesteps 50000 --revisions 4
    python run_experiment.py --plots-only             # Just regenerate plots
"""

import argparse
import uuid
import time
from pathlib import Path

from measurement.database import init_db
from visualization.plots import generate_all_plots


def run_full_experiment(timesteps_per_agent: int = 30_000, n_revisions: int = 4):
    """Run all four agents across all four worlds."""
    from environments.cartpole_variants import get_all_worlds
    from agents.v1_dqn import run as run_v1_dqn
    from agents.v2_rule_follower import run as run_v2_rule_follower
    from agents.v3_scientist import run as run_v3_scientist
    from agents.v4_novelty_seeker import run as run_v4_novelty_seeker

    worlds = get_all_worlds()
    batch_id = uuid.uuid4().hex[:8]
    results = []

    print(f"\n{'='*60}")
    print(f"  NOVELTY BENCHMARK — Full Experiment Run")
    print(f"  Batch ID: {batch_id}")
    print(f"  Worlds: {worlds}")
    print(f"  Timesteps/agent: {timesteps_per_agent}")
    print(f"{'='*60}\n")

    # V1: DQN baseline — no LLM
    print("\n── V1: Silent Instinct (DQN) ─────────────────────────────")
    for world in worlds:
        run_id = f"v1_{world}_{batch_id}"
        result = run_v1_dqn(world=world, total_timesteps=timesteps_per_agent, run_id=run_id)
        results.append(result)
        print(f"  ✓ {world}: mean reward = {result['final_mean_reward']:.1f}")

    # V2: One-shot LLM reward, no revision
    print("\n── V2: Rule Follower (one-shot LLM) ──────────────────────")
    for world in worlds:
        run_id = f"v2_{world}_{batch_id}"
        result = run_v2_rule_follower(
            world=world, total_timesteps=timesteps_per_agent, run_id=run_id
        )
        results.append(result)
        print(f"  ✓ {world}: mean reward = {result['final_mean_reward']:.1f}")

    # V3: Scientist — LLM + iterative revision
    print("\n── V3: Scientist (RL ↔ LLM) ─────────────────────────────")
    for world in worlds:
        run_id = f"v3_{world}_{batch_id}"
        result = run_v3_scientist(
            world=world,
            n_revisions=n_revisions,
            timesteps_per_phase=timesteps_per_agent // (n_revisions + 1),
            run_id=run_id
        )
        results.append(result)
        print(f"  ✓ {world}: mean reward = {result['final_mean_reward']:.1f}, "
              f"revisions = {result['n_revisions']}")

    # V4: Novelty Seeker — V3 + novelty pressure in prompt
    print("\n── V4: Novelty Seeker (RL ↔ LLM + novelty pressure) ─────")
    for world in worlds:
        run_id = f"v4_{world}_{batch_id}"
        result = run_v4_novelty_seeker(
            world=world,
            n_revisions=n_revisions,
            timesteps_per_phase=timesteps_per_agent // (n_revisions + 1),
            run_id=run_id
        )
        results.append(result)
        print(f"  ✓ {world}: mean reward = {result['final_mean_reward']:.1f}, "
              f"revisions = {result['n_revisions']}")

    print(f"\n{'='*60}")
    print(f"  Experiment complete. Generating plots...")
    print(f"{'='*60}\n")

    plot_paths = generate_all_plots()

    print(f"\n✅ Done! Results in: {Path('results').absolute()}")
    print(f"   Plots generated: {len(plot_paths)}")
    print(f"\nTo view the interactive dashboard:")
    print(f"   streamlit run visualization/dashboard.py")

    return results


def run_single(agent: str, world: str, timesteps: int = 50_000, n_revisions: int = 4):
    """Single agent/world run."""
    if agent == "v1":
        from agents.v1_dqn import run
        return run(world=world, total_timesteps=timesteps)
    elif agent == "v2":
        from agents.v2_rule_follower import run
        return run(world=world, total_timesteps=timesteps)
    elif agent == "v3":
        from agents.v3_scientist import run
        return run(world=world, n_revisions=n_revisions,
                   timesteps_per_phase=timesteps // (n_revisions + 1))
    elif agent == "v4":
        from agents.v4_novelty_seeker import run
        return run(world=world, n_revisions=n_revisions,
                   timesteps_per_phase=timesteps // (n_revisions + 1))
    else:
        raise ValueError(f"Unknown agent: {agent}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Novelty Benchmark Experiment Runner")
    parser.add_argument("--agent", choices=["v1", "v2", "v3", "v4", "all"], default="all")
    parser.add_argument("--world", choices=["baseline", "inverted", "viscous", "delayed", "all"],
                        default="all")
    parser.add_argument("--timesteps", type=int, default=30_000)
    parser.add_argument("--revisions", type=int, default=4)
    parser.add_argument("--plots-only", action="store_true")

    args = parser.parse_args()

    init_db()

    if args.plots_only:
        generate_all_plots()
    elif args.agent == "all" and args.world == "all":
        run_full_experiment(timesteps_per_agent=args.timesteps, n_revisions=args.revisions)
    else:
        agent = args.agent if args.agent != "all" else "v3"
        world = args.world if args.world != "all" else "inverted"
        result = run_single(agent=agent, world=world,
                            timesteps=args.timesteps, n_revisions=args.revisions)
        print(f"\nResult: {result}")
        generate_all_plots()
