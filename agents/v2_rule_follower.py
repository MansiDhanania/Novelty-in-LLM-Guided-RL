"""
v2_rule_follower.py
V2: The Rule Follower
LLM generates ONE reward function at the start, then steps back forever.
No revision. No feedback loop. No adaptation.

Scientific purpose: Isolates whether the REVISION LOOP in V3 matters,
or whether any single LLM prompt is enough.

Expected behavior:
- Better than V1 on novel worlds (LLM has physics priors)
- Worse than V3 on novel worlds (no adaptation)
- Gap between V2 and V3 = value of iterative revision
"""

import time
import uuid
import textwrap
import os
import numpy as np
from groq import Groq
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from environments.cartpole_variants import make_world
from measurement.database import (
    init_db, log_experiment, log_episode, log_hypothesis, log_novelty_score
)
from measurement.novelty_scorer import score_hypothesis, fingerprint_trajectory

load_dotenv()

AGENT_NAME = "v2_rule_follower"

# ── Same observation format as V3 so reward functions are comparable ────────

OBSERVATION_FORMAT = """
CRITICAL: obs is a numpy array (NOT a dict). Access it as:
  obs[0] = cart position (meters from center, negative=left)
  obs[1] = cart velocity (m/s)
  obs[2] = pole angle (radians, 0=upright, positive=clockwise)
  obs[3] = pole angular velocity (rad/s)
Do NOT use obs['CartPos'] or any string keys.

WRITE MULTILINE CODE, NOT SINGLE-LINE.
DO NOT use semicolons.

Example:
  def custom_reward(obs, action, base_reward, terminated):
    cart_pos, cart_vel, pole_angle, pole_vel = obs
    if terminated:
      return -10
    survival_bonus = 0.1
    angle_penalty = abs(pole_angle) * 0.5
    return survival_bonus - angle_penalty

Use abs(), max(), min(). MUST return a single float.
"""

ONE_SHOT_PROMPT = """You control a cart-pole system in a NOVEL physics environment.
Observations: [cart_pos, cart_vel, pole_angle, pole_vel]

Telemetry from 10 random episodes:
{telemetry}

Based on this telemetry, form a physics hypothesis about how this world works.
Then write ONE reward function that you believe will solve this environment.

This is your ONLY chance — you will NOT be able to revise it later.
Make it as robust and general as possible.

REWARD DESIGN TIPS:
- In difficult physics, sparse penalties hurt learning. Reward progress toward goals.
- Try: reward if pole angle is DECREASING, or if cart moves toward center.
- Less effective: only penalizing termination with -10.
- Use abs(pole_angle), max(), min() to design smooth rewards.

Use \\n for newlines. Use abs() for absolute value.

RESPOND WITH ONLY THIS JSON (no markdown, no extra text):
{{"hypothesis": "Your one-shot theory about the physics", "reward_function": "def custom_reward(obs, action, base_reward, terminated):\\n    cart_pos, cart_vel, pole_angle, pole_vel = obs\\n    if terminated: return -5\\n    return 0.5 - abs(pole_angle)"}}"""


class HypothesisResponse(BaseModel):
    hypothesis: str = Field(default="Unable to determine")
    reward_function: str = Field(
        default="def custom_reward(obs, action, base_reward, terminated):\n    return base_reward"
    )


# ── Same CustomRewardWrapper as V3 (copy for self-containment) ──────────────

class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_fn_code: str):
        super().__init__(env)
        self.reward_fn = self._compile(reward_fn_code)
        self._error_count = 0
        self._warned = False

    def _compile(self, code: str):
        namespace = {}
        try:
            code = code.replace("\\n", "\n")
            code = textwrap.dedent(code)
            safe_builtins = {
                'abs': abs, 'max': max, 'min': min, 'len': len,
                'float': float, 'int': int, 'bool': bool, 'sum': sum,
                '__name__': '__main__'
            }
            exec(code, {"np": np, "__builtins__": safe_builtins}, namespace)
            fn = namespace.get("custom_reward")
            if fn is None:
                raise ValueError("No custom_reward function found")
            return fn
        except Exception as e:
            print(f"  [V2 Reward Compile Error] {e}. Using base reward.")
            return None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._error_count = 0
        self._warned = False
        return obs, info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        if self.reward_fn is not None:
            try:
                r = float(self.reward_fn(obs, action, base_reward, terminated))
                reward = r if np.isfinite(r) else base_reward
            except Exception as e:
                reward = base_reward
                self._error_count += 1
                if not self._warned and self._error_count >= 3:
                    print(f"  [V2 WARNING] Reward fn crashing: {str(e)[:60]}")
                    self._warned = True
        else:
            reward = base_reward
        return obs, reward, terminated, truncated, info


# ── V2 Agent ────────────────────────────────────────────────────────────────

class RuleFollowerAgent:

    def __init__(self, world: str, run_id: str,
                 groq_model: str = "llama-3.1-8b-instant"):
        self.world = world
        self.run_id = run_id
        self.groq_model = groq_model
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        self.hypothesis = None
        self.reward_fn_code = None
        self.embedding = None

    def _call_llm(self, prompt: str) -> dict:
        """Call Groq, parse JSON, validate with Pydantic."""
        import json, re
        try:
            resp = self.client.chat.completions.create(
                model=self.groq_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800
            )
            raw = resp.choices[0].message.content.strip()

            # Extract JSON
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                raw = json_match.group()
            parsed = json.loads(raw)
            validated = HypothesisResponse(**parsed)
            return validated.model_dump()
        except Exception as e:
            print(f"  [V2 LLM Error] {e}")
            return {
                "hypothesis": "Unable to parse response",
                "reward_function": "def custom_reward(obs, action, base_reward, terminated):\n    return base_reward"
            }

    def form_one_shot_hypothesis(self, env) -> dict:
        """
        Collect telemetry from 10 random episodes, then call LLM ONCE.
        This is the only LLM call V2 will ever make.
        """
        print(f"\n[V2] Collecting telemetry from {self.world}...")
        telemetry_lines = []
        for ep in range(10):
            obs, _ = env.reset()
            ep_rewards, ep_obs = [], []
            for _ in range(100):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_rewards.append(reward)
                ep_obs.append(obs.tolist())
                if terminated or truncated:
                    break
            mean_obs = np.mean(ep_obs, axis=0).tolist()
            telemetry_lines.append(
                f"  Ep {ep+1}: reward={sum(ep_rewards):.1f}, "
                f"steps={len(ep_rewards)}, "
                f"mean_obs=[{', '.join(f'{v:.3f}' for v in mean_obs)}]"
            )

        telemetry_str = "\n".join(telemetry_lines)
        prompt = ONE_SHOT_PROMPT.format(telemetry=telemetry_str)

        print("[V2] Calling LLM (one-shot, no revisions)...")
        response = self._call_llm(prompt)
        self.hypothesis = response["hypothesis"]
        self.reward_fn_code = response["reward_function"]

        print(f"  [V2] Hypothesis: {self.hypothesis[:100]}...")
        return response

    def score_and_log(self, episode_num: int, trajectory: list):
        """Score and log the single hypothesis."""
        scores = score_hypothesis(
            self.hypothesis,
            previous_embedding=None,  # no previous — one-shot
            run_boden=True
        )
        behavioral_hash = fingerprint_trajectory(trajectory)

        log_hypothesis(
            run_id=self.run_id,
            agent=AGENT_NAME,
            world=self.world,
            episode_num=episode_num,
            hypothesis_text=self.hypothesis,
            reward_fn_code=self.reward_fn_code,
            boden_type=scores["boden_type"],
            boden_justification=scores["boden_justification"],
            embedding=scores["embedding"],
            embedding_distance=0.0  # first and only hypothesis, distance = 0
        )
        log_novelty_score(
            run_id=self.run_id,
            agent=AGENT_NAME,
            world=self.world,
            episode_num=episode_num,
            linguistic_surprise=0.0,  # no revision = no surprise
            strategy_label=scores.get("strategy_label", "Unknown"),
            behavioral_hash=behavioral_hash
        )
        self.embedding = scores["embedding"]
        return scores


# ── Main run function ────────────────────────────────────────────────────────

def run(world: str = "inverted",
        total_timesteps: int = 50_000,
        run_id: str = None,
        verbose: bool = True) -> dict:
    """
    Train V2 on a world.
    - One LLM call to generate reward function
    - DQN trains for full timestep budget on that fixed reward
    - No revision, no feedback loop

    Args:
        world: Physics world
        total_timesteps: Training budget (same as V1/V3 for fair comparison)
        run_id: Unique ID
        verbose: Print progress
    """
    init_db()
    run_id = run_id or f"v2_{world}_{uuid.uuid4().hex[:8]}"

    log_experiment(run_id, AGENT_NAME, world, config={
        "total_timesteps": total_timesteps,
        "algorithm": "DQN",
        "llm_calls": 1,
        "revision_loop": False
    })

    agent = RuleFollowerAgent(world=world, run_id=run_id)

    # Collect telemetry and form one-shot hypothesis
    base_env = make_world(world)
    agent.form_one_shot_hypothesis(base_env)
    base_env.close()

    # Set up environment with the one-shot reward function
    base_env = make_world(world)
    wrapped_env = CustomRewardWrapper(base_env, agent.reward_fn_code)

    all_rewards = []
    last_trajectory = []
    episode_num = 0

    class EpisodeCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self._ep_reward = 0.0
            self._ep_steps = 0
            self._ep_num = 0
            self._current_traj = []

        def _on_step(self):
            obs = self.locals["new_obs"][0]
            action = self.locals["actions"][0]
            reward = self.locals["rewards"][0]
            done = self.locals["dones"][0]

            self._current_traj.append((obs.copy(), int(action)))
            self._ep_reward += reward
            self._ep_steps += 1

            if done:
                all_rewards.append(self._ep_reward)
                nonlocal last_trajectory
                last_trajectory = self._current_traj.copy()

                log_episode(
                    run_id=run_id,
                    agent=AGENT_NAME,
                    world=world,
                    episode_num=self._ep_num,
                    total_reward=self._ep_reward,
                    steps=self._ep_steps,
                    terminated=True
                )
                self._current_traj.clear()
                self._ep_reward = 0.0
                self._ep_steps = 0
                self._ep_num += 1
            return True

    cb = EpisodeCallback()

    model = DQN(
        "MlpPolicy", wrapped_env,
        learning_rate=1e-3, buffer_size=10_000,
        learning_starts=500, batch_size=64,
        gamma=0.99, exploration_fraction=0.3,
        exploration_final_eps=0.05,
        verbose=1 if verbose else 0,
        seed=42
    )

    if verbose:
        print(f"\n[V2] Training on {world} for {total_timesteps} timesteps...")
        start = time.time()

    model.learn(total_timesteps=total_timesteps, callback=cb)

    if verbose:
        print(f"[V2] Training complete in {time.time()-start:.1f}s")

    wrapped_env.close()

    # Log the single hypothesis at the end
    agent.score_and_log(episode_num=0, trajectory=last_trajectory)

    final_mean = float(np.mean(all_rewards[-50:])) if all_rewards else 0.0

    if verbose:
        print(f"[V2] Final mean reward (last 50 eps): {final_mean:.1f}")

    return {
        "run_id": run_id,
        "agent": AGENT_NAME,
        "world": world,
        "total_episodes": len(all_rewards),
        "final_mean_reward": final_mean,
        "n_revisions": 0,
        "hypothesis": agent.hypothesis
    }


if __name__ == "__main__":
    for world in ["baseline", "inverted", "viscous", "delayed"]:
        result = run(world=world, total_timesteps=50_000)
        print(f"\nResult: {result}")