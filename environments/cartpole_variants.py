"""
cartpole_variants.py
Four physics-deformed CartPole environments.
Each world is a wrapper around Gymnasium's CartPole-v1 that overrides
specific physics parameters to break the agent's prior assumptions.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control.cartpole import CartPoleEnv


class BaseCartPole(gym.Env):
    """
    Base wrapper that exposes physics parameters for modification.
    All variant worlds inherit from this.
    """
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    WORLD_DESCRIPTIONS = {
        "baseline": "Standard CartPole. Gravity=9.8. Agent has full training data coverage.",
        "inverted": "Gravity is reversed (-9.8). Pole 'falls' upward. Standard strategies fail completely.",
        "viscous": "Zero gravity with high viscosity. Pole stays where placed but resists movement.",
        "delayed":  "Actions are delayed by 3 frames. Cause and effect are decoupled."
    }

    def __init__(self, world: str = "baseline", render_mode=None):
        self.world = world
        self.render_mode = render_mode

        # Build the underlying CartPole env
        self._env = CartPoleEnv(render_mode=render_mode)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

        # Apply physics overrides
        self._configure_physics()

        # Action delay buffer for world D
        self._action_buffer = []
        self._delay_frames = 3

    def _configure_physics(self):
        """Override CartPole physics based on world type."""
        if self.world == "baseline":
            self._env.gravity = 9.8
            self._env.masscart = 1.0
            self._env.masspole = 0.1
            self._env.force_mag = 10.0
            self.viscosity = 0.0

        elif self.world == "inverted":
            # Gravity reversed: pole wants to fall upward
            self._env.gravity = -9.8
            self._env.masscart = 1.0
            self._env.masspole = 0.1
            self._env.force_mag = 10.0
            self.viscosity = 0.0

        elif self.world == "viscous":
            # Zero gravity, high viscosity: pole resists movement
            self._env.gravity = 0.0
            self._env.masscart = 2.0
            self._env.masspole = 0.3
            self._env.force_mag = 10.0
            self.viscosity = 0.85  # velocity damping coefficient

        elif self.world == "delayed":
            self._env.gravity = 9.8
            self._env.masscart = 1.0
            self._env.masspole = 0.1
            self._env.force_mag = 10.0
            self.viscosity = 0.0

        else:
            raise ValueError(f"Unknown world: {self.world}. Choose from: baseline, inverted, viscous, delayed")

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        self._action_buffer = []
        info["world"] = self.world
        info["physics"] = self._get_physics_summary()
        return obs, info

    def step(self, action):
        # Handle action delay for world D
        if self.world == "delayed":
            self._action_buffer.append(action)
            if len(self._action_buffer) >= self._delay_frames:
                actual_action = self._action_buffer.pop(0)
            else:
                actual_action = 0  # No-op while buffer fills

            obs, reward, terminated, truncated, info = self._env.step(actual_action)
        else:
            obs, reward, terminated, truncated, info = self._env.step(action)

        # Apply viscosity damping (world C)
        if self.viscosity > 0.0:
            # Damp velocities in observation: [cart_pos, cart_vel, pole_angle, pole_angular_vel]
            obs = np.array(obs, dtype=np.float32)
            obs[1] *= (1.0 - self.viscosity)  # cart velocity
            obs[3] *= (1.0 - self.viscosity)  # pole angular velocity

        info["world"] = self.world
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    def _get_physics_summary(self) -> dict:
        return {
            "gravity": self._env.gravity,
            "masscart": self._env.masscart,
            "masspole": self._env.masspole,
            "force_mag": self._env.force_mag,
            "viscosity": getattr(self, "viscosity", 0.0),
            "action_delay": self._delay_frames if self.world == "delayed" else 0
        }

    def get_telemetry_string(self, obs, action, reward, step_num) -> str:
        """
        Format a single step into a compact string for LLM consumption.
        Keeps token usage low while preserving all relevant signal.
        """
        action_str = "PUSH_RIGHT" if action == 1 else "PUSH_LEFT"
        return (
            f"Step {step_num:04d} | Action: {action_str} | "
            f"Cart pos: {obs[0]:.3f}, Cart vel: {obs[1]:.3f}, "
            f"Pole angle: {obs[2]:.4f} rad ({np.degrees(obs[2]):.2f}°), "
            f"Pole vel: {obs[3]:.3f} | Reward: {reward:.1f}"
        )


def make_world(world: str, render_mode=None) -> BaseCartPole:
    """Factory function. Use this instead of instantiating directly."""
    return BaseCartPole(world=world, render_mode=render_mode)


def get_all_worlds():
    return ["baseline", "inverted", "viscous", "delayed"]


if __name__ == "__main__":
    # Quick sanity check
    for world in get_all_worlds():
        env = make_world(world)
        obs, info = env.reset(seed=42)
        print(f"\nWorld: {world}")
        print(f"Physics: {info['physics']}")
        print(f"Description: {BaseCartPole.WORLD_DESCRIPTIONS[world]}")
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(env.get_telemetry_string(obs, action, reward, step))
            if terminated or truncated:
                break
        env.close()
    print("\nAll worlds OK.")
