"""
v1_dqn.py
V1: The Silent Instinct
Pure DQN agent using Stable Baselines3. No language, no reasoning.
This is your baseline — the dumbest possible agent.

What it tests: Raw pattern matching without any hypothesis formation.
Expected behavior: Solves baseline quickly, struggles badly in broken worlds.
"""

import time
import uuid
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from environments.cartpole_variants import make_world
from measurement.database import init_db, log_experiment, log_episode


class EpisodeLogger(BaseCallback):
    """SB3 callback that logs each episode to our database."""

    def __init__(self, run_id: str, agent: str, world: str, verbose=0):
        super().__init__(verbose)
        self.run_id = run_id
        self.agent = agent
        self.world = world
        self._episode_reward = 0.0
        self._episode_steps = 0
        self._episode_num = 0

    def _on_step(self) -> bool:
        self._episode_reward += self.locals["rewards"][0]
        self._episode_steps += 1

        if self.locals["dones"][0]:
            log_episode(
                run_id=self.run_id,
                agent=self.agent,
                world=self.world,
                episode_num=self._episode_num,
                total_reward=self._episode_reward,
                steps=self._episode_steps,
                terminated=bool(self.locals["infos"][0].get("TimeLimit.truncated", False) is False
                                and self.locals["dones"][0])
            )
            self._episode_reward = 0.0
            self._episode_steps = 0
            self._episode_num += 1

        return True  # Continue training


def run(world: str = "baseline", total_timesteps: int = 50_000,
        run_id: str = None, verbose: bool = True) -> dict:
    """
    Train V1 on a given world. Returns summary stats.

    Args:
        world: One of baseline/inverted/viscous/delayed
        total_timesteps: Training budget
        run_id: Unique experiment ID (auto-generated if None)
        verbose: Print progress

    Returns:
        dict with run_id, final_mean_reward, episodes_to_solve
    """
    init_db()
    run_id = run_id or f"v1_{world}_{uuid.uuid4().hex[:8]}"
    agent_name = "v1_dqn"

    log_experiment(run_id, agent_name, world, config={
        "total_timesteps": total_timesteps,
        "algorithm": "DQN",
        "policy": "MlpPolicy"
    })

    env = make_world(world)
    callback = EpisodeLogger(run_id, agent_name, world)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=10_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        verbose=1 if verbose else 0,
        seed=42
    )

    if verbose:
        print(f"\n[V1 DQN] World: {world} | Timesteps: {total_timesteps}")
        start = time.time()

    model.fit = model.learn  # alias for clarity
    model.learn(total_timesteps=total_timesteps, callback=callback)

    if verbose:
        elapsed = time.time() - start
        print(f"[V1 DQN] Training complete in {elapsed:.1f}s")

    # Evaluate final performance
    eval_rewards = []
    for _ in range(10):
        obs, _ = env.reset()
        ep_reward = 0
        for _ in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break
        eval_rewards.append(ep_reward)

    env.close()

    final_mean = np.mean(eval_rewards)
    if verbose:
        print(f"[V1 DQN] Final eval reward (10 eps): {final_mean:.1f} ± {np.std(eval_rewards):.1f}")

    return {
        "run_id": run_id,
        "agent": agent_name,
        "world": world,
        "final_mean_reward": final_mean,
        "final_std_reward": float(np.std(eval_rewards))
    }


if __name__ == "__main__":
    # Test on all worlds
    for world in ["baseline", "inverted", "viscous", "delayed"]:
        result = run(world=world, total_timesteps=20_000)
        print(f"\nResult: {result}")
