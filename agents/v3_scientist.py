"""
v3_scientist.py
V3: The Scientist
The core of your project. An RL ↔ LLM feedback loop where the LLM:
1. Observes episode telemetry
2. Forms a hypothesis about the physics
3. Writes a custom reward function
4. Watches the RL agent train on it
5. Revises its theory after failure

What it tests: Iterative hypothesis revision — the closest thing to
scientific reasoning in current AI.
"""

import time
import uuid
import textwrap
import traceback
import numpy as np
from groq import Groq
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import os
from dotenv import load_dotenv
import gymnasium as gym
from pydantic import BaseModel, ValidationError, Field
from typing import Dict, Any, Optional

from environments.cartpole_variants import make_world, BaseCartPole
from measurement.database import (
    init_db, log_experiment, log_episode, log_hypothesis, log_novelty_score
)
from measurement.novelty_scorer import score_hypothesis, fingerprint_trajectory

load_dotenv()


# Pydantic models for LLM responses
class HypothesisResponse(BaseModel):
    hypothesis: str = Field(default="Unable to determine", description="Physics hypothesis")
    reward_function: str = Field(default="def custom_reward(obs, action, base_reward, terminated):\n    return base_reward")



# ── LLM Prompts ────────────────────────────────────────────────────────────

OBSERVATION_FORMAT = """
CRITICAL: obs is a numpy array (NOT a dict). Access it as:
  obs[0] = cart position (meters from center, negative=left)
  obs[1] = cart velocity (m/s)
  obs[2] = pole angle (radians, 0=upright, positive=clockwise)
  obs[3] = pole angular velocity (rad/s)
Do NOT use obs['CartPos'] or any string keys - this will crash and return 0 reward.

WRITE MULTILINE CODE, NOT SINGLE-LINE. Use newlines and proper indentation.
DO NOT use semicolons. DO NOT put the whole function on one line.

Reward function example:
  def custom_reward(obs, action, base_reward, terminated):
    cart_pos, cart_vel, pole_angle, pole_vel = obs
    if terminated:
      return -10
    survival_bonus = 0.1
    angle_penalty = abs(pole_angle) * 0.5
    return survival_bonus - angle_penalty
  
Use built-in functions: abs(), max(), min(), float(), int(). MUST return a single float.
"""

INITIAL_HYPOTHESIS_PROMPT = """You control a cart-pole system. Observations: [cart_pos, cart_vel, pole_angle, pole_vel].

Telemetry:
{telemetry}

Form a brief physics hypothesis (1-2 sentences). Generate ONE simple reward function.

REWARD DESIGN TIPS:
- In difficult physics, sparse penalties hurt learning. Reward progress toward goals.
- Try: reward if pole angle is DECREASING, or if cart moves toward center.
- Less effective: only penalizing termination with -10.
- Use abs(pole_angle), max(), min() to design smooth rewards.

Use \\n for newlines. Use abs() for absolute value.

RESPOND WITH ONLY THIS JSON (no markdown, no extra text):
{{"hypothesis": "Your theory", "reward_function": "def custom_reward(obs, action, base_reward, terminated):\\n    cart_pos, cart_vel, pole_angle, pole_vel = obs\\n    if terminated: return -5\\n    return 0.5 - abs(pole_angle)"}}"""

REVISION_PROMPT = """Your theory failed. Mean reward: {mean_reward:.1f}, Best: {best_reward:.1f}

Previous theory: {previous_hypothesis}
Failure analysis: {diagnosis}

Revise your hypothesis in 1-2 sentences. Generate a NEW reward function.

REWARD DESIGN TIPS:
- Reward PROGRESS not just survival. Did the pole angle improve? Did cart position get better?
- Try lower termination penalty (-5 instead of -10) so learning isn't overwhelmed by negative rewards.
- Use: 0.5 - abs(pole_angle) to reward staying upright without huge penalties.
- Experiment with cart_vel, pole_vel terms to stabilize the system.

Use \\n for newlines. Use abs(), max(), min() for calculations.

RESPOND WITH ONLY THIS JSON:
{{"hypothesis": "Your revised theory", "reward_function": "def custom_reward(obs, action, base_reward, terminated):\\n    cart_pos, cart_vel, pole_angle, pole_vel = obs\\n    if terminated: return -5\\n    stability = 0.5 - abs(pole_angle)\\n    return stability"}}"""


def _diagnose_failure(mean_reward: float, best_reward: float) -> str:
    """Generate a plain-English failure diagnosis for the revision prompt."""
    if mean_reward < 15:
        return "completely ineffective — the pole is falling almost immediately"
    elif mean_reward < 50:
        return "poor — the agent is making progress but failing early"
    elif mean_reward < 150:
        return "mediocre — the agent survives short bursts but can't sustain balance"
    elif best_reward > 400:
        return "inconsistent — sometimes excellent but not reliably stable"
    else:
        return "suboptimal — better than random but not solving the task"


# ── Custom Reward Wrapper ───────────────────────────────────────────────────

class CustomRewardWrapper(gym.Wrapper):
    """
    Wraps an environment and applies the LLM-generated reward function.
    Falls back to base reward if the custom function crashes.
    """

    def __init__(self, env: BaseCartPole, reward_fn_code: str):
        super().__init__(env)
        self.reward_fn = self._compile_reward_fn(reward_fn_code)
        self._last_obs = None
        self._error_count = 0
        self._error_warning_printed = False

    def _compile_reward_fn(self, code: str):
        """Safely exec the LLM-generated code and extract the function."""
        namespace = {}
        try:
            # Clean up the code string - handle escaped newlines from JSON
            code = code.replace("\\n", "\n")  # Convert escaped newlines to actual newlines
            code = textwrap.dedent(code)
            
            # Provide essential builtins (abs, max, min, etc) but restrict others
            safe_builtins = {
                'abs': abs, 'max': max, 'min': min, 'len': len, 'range': range,
                'float': float, 'int': int, 'bool': bool, 'sum': sum, '__name__': '__main__'
            }
            
            exec(code, {"np": np, "__builtins__": safe_builtins}, namespace)
            fn = namespace.get("custom_reward")
            if fn is None:
                raise ValueError("No custom_reward function found in generated code")
            return fn
        except Exception as e:
            print(f"  [Reward Compile Error] {e}. Using base reward.")
            return None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        self._error_count = 0
        self._error_warning_printed = False
        return obs, info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        if self.reward_fn is not None:
            try:
                custom = self.reward_fn(obs, action, base_reward, terminated)
                reward = float(custom)
                # Safety clamp: reward must be finite
                if not np.isfinite(reward):
                    reward = base_reward
                    self._error_count += 1
            except Exception as e:
                reward = base_reward
                self._error_count += 1
                # Warn once per episode if function keeps failing
                if not self._error_warning_printed and self._error_count >= 3:
                    print(f"  [WARNING] Reward function crashing repeatedly: {str(e)[:60]}. Using base reward.")
                    self._error_warning_printed = True
        else:
            reward = base_reward

        self._last_obs = obs
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()


# ── Main Agent ─────────────────────────────────────────────────────────────

class ScientistAgent:

    def __init__(self, world: str, run_id: str, groq_model: str = "llama-3.1-8b-instant"):
        self.world = world
        self.run_id = run_id
        self.groq_model = groq_model
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        self.current_hypothesis = None
        self.current_reward_fn_code = None
        self.previous_embedding = None
        self.hypothesis_history = []
        self.revision_count = 0

    def _call_llm(self, prompt: str) -> dict:
        """Call Groq and parse JSON response with Pydantic validation."""
        import json
        import re
        response = self.client.chat.completions.create(
            model=self.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=4000  # Increased from 2000 to prevent cutoff
        )
        raw = response.choices[0].message.content.strip()
        
        # DEBUG: Print first 300 chars of response
        print(f"  [DEBUG] LLM response (first 300 chars): {raw[:300]}")

        parsed = None
        
        # Strategy 1: Strip markdown code fences
        if "```" in raw:
            parts = raw.split("```")
            for i, part in enumerate(parts):
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part and part.startswith('{'):
                    try:
                        parsed = json.loads(part)
                        if parsed:
                            break
                    except json.JSONDecodeError:
                        continue
        
        # Strategy 2: Direct JSON parsing
        if not parsed:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Try to complete incomplete JSON
        if not parsed:
            # The response might be cut off - try to complete it
            json_str = raw
            # Count braces to see if we're missing a closing brace
            if json_str.count('{') > json_str.count('}'):
                json_str += '}' * (json_str.count('{') - json_str.count('}'))
            # Try to find unclosed strings and close them
            if json_str.count('"') % 2 == 1:
                json_str += '"'
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # Strategy 4: Extract JSON object using regex (find { to })
        if not parsed:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                try:
                    json_str = match.group()
                    # Try to complete it if necessary
                    if json_str.count('{') > json_str.count('}'):
                        json_str += '}' * (json_str.count('{') - json_str.count('}'))
                    if json_str.count('"') % 2 == 1:
                        json_str += '"'
                    parsed = json.loads(json_str)
                except json.JSONDecodeError:
                    pass
        
        # Validate with Pydantic if we got something
        if parsed:
            try:
                validated = HypothesisResponse(**parsed)
                print(f"  [SUCCESS] Parsed hypothesis: {validated.hypothesis[:80]}...")
                return validated.model_dump()
            except ValidationError as e:
                print(f"  [WARNING] Pydantic validation failed. Trying partial extraction...")
                # Try to extract just the hypothesis and reward function
                if "hypothesis" in parsed or "reward_function" in parsed:
                    return {
                        "hypothesis": parsed.get("hypothesis", "Unable to parse response"),
                        "reward_function": parsed.get("reward_function", "def custom_reward(obs, action, base_reward, terminated):\n    return base_reward")
                    }
        
        # Fallback response
        print(f"  [ERROR] Could not parse LLM response after all strategies. Using fallback.")
        return {
            "hypothesis": "Unable to parse response",
            "reward_function": "def custom_reward(obs, action, base_reward, terminated):\n    return base_reward"
        }

    def _collect_telemetry(self, env: BaseCartPole, n_steps: int = 30) -> tuple:
        """
        Run random policy and collect telemetry string + trajectory.
        Returns (telemetry_string, trajectory_list)
        """
        obs, _ = env.reset(seed=42)
        telemetry_lines = []
        trajectory = []

        for i in range(n_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            telemetry_lines.append(env.get_telemetry_string(obs, action, reward, i))
            trajectory.append((obs.copy(), action))
            if terminated or truncated:
                obs, _ = env.reset()

        return "\n".join(telemetry_lines), trajectory

    def form_initial_hypothesis(self, env: BaseCartPole) -> dict:
        """Observe random play and form first hypothesis."""
        print(f"\n  [V3] Collecting telemetry for initial hypothesis...")
        telemetry, trajectory = self._collect_telemetry(env, n_steps=15)  # Reduced from 40

        prompt = INITIAL_HYPOTHESIS_PROMPT.format(
            n_steps=15,
            telemetry=telemetry
        )

        response = self._call_llm(prompt)
        self.current_hypothesis = response.get("hypothesis", "Unknown physics.")
        self.current_reward_fn_code = response.get("reward_function", "")

        print(f"  [V3] Hypothesis: {self.current_hypothesis[:100]}...")
        return response

    def revise_hypothesis(self, env: BaseCartPole, episode_rewards: list,
                          last_trajectory: list) -> dict:
        """Revise theory based on observed failure."""
        self.revision_count += 1

        # Guard against empty episode list (can happen if DQN spent all timesteps warming up)
        if not episode_rewards:
            print(f"  [V3] Revision {self.revision_count}: No episodes completed — skipping LLM call.")
            return {}

        mean_r = np.mean(episode_rewards)
        best_r = max(episode_rewards)
        worst_r = min(episode_rewards)
        success_rate = sum(1 for r in episode_rewards if r > 100) / len(episode_rewards)

        # Collect fresh failure telemetry
        telemetry_lines = []
        for i, (obs, action) in enumerate(last_trajectory[-20:]):
            reward = 1.0  # approximate
            env_ref = env  # just for format
            line = f"Step {i:04d} | Action: {'PUSH_RIGHT' if action==1 else 'PUSH_LEFT'} | "
            line += f"Cart pos: {obs[0]:.3f}, Cart vel: {obs[1]:.3f}, "
            line += f"Pole angle: {obs[2]:.4f} rad, Pole vel: {obs[3]:.3f}"
            telemetry_lines.append(line)

        prompt = REVISION_PROMPT.format(
            previous_hypothesis=self.current_hypothesis,
            previous_reward_fn=self.current_reward_fn_code,
            n_episodes=len(episode_rewards),
            mean_reward=mean_r,
            best_reward=best_r,
            worst_reward=worst_r,
            success_rate=success_rate,
            telemetry="\n".join(telemetry_lines),
            diagnosis=_diagnose_failure(mean_r, best_r)
        )

        response = self._call_llm(prompt)
        self.current_hypothesis = response.get("hypothesis", self.current_hypothesis)
        self.current_reward_fn_code = response.get("reward_function", self.current_reward_fn_code)

        print(f"  [V3] Revision {self.revision_count}: {self.current_hypothesis[:100]}...")
        return response

    def score_and_log_hypothesis(self, episode_num: int, trajectory: list):
        """Score the current hypothesis and log everything to DB."""
        scores = score_hypothesis(
            self.current_hypothesis,
            previous_embedding=self.previous_embedding,
            run_boden=True
        )

        behavioral_hash = fingerprint_trajectory(trajectory)

        log_hypothesis(
            run_id=self.run_id,
            agent="v3_scientist",
            world=self.world,
            episode_num=episode_num,
            hypothesis_text=self.current_hypothesis,
            reward_fn_code=self.current_reward_fn_code,
            boden_type=scores["boden_type"],
            boden_justification=scores["boden_justification"],
            embedding=scores["embedding"],
            embedding_distance=scores["embedding_distance"]
        )

        log_novelty_score(
            run_id=self.run_id,
            agent="v3_scientist",
            world=self.world,
            episode_num=episode_num,
            linguistic_surprise=scores["embedding_distance"],
            strategy_label=scores.get("strategy_label", "Unknown"),
            behavioral_hash=behavioral_hash
        )

        self.previous_embedding = scores["embedding"]
        self.hypothesis_history.append({
            "episode": episode_num,
            "hypothesis": self.current_hypothesis,
            "boden_type": scores["boden_type"],
            "distance": scores["embedding_distance"],
            "strategy_label": scores.get("strategy_label")
        })

        return scores


def run(world: str = "inverted",
        revision_interval: int = 50,
        n_revisions: int = 5,
        timesteps_per_phase: int = 50_000,
        run_id: str = None,
        verbose: bool = True) -> dict:
    """
    Run V3 Scientist on a world.

    Protocol:
    1. Collect telemetry → form hypothesis → write reward fn
    2. Train DQN for timesteps_per_phase steps
    3. Evaluate performance
    4. If not solved, revise hypothesis and repeat
    5. Log everything

    Args:
        world: Physics world to run in
        revision_interval: Episodes between performance checks
        n_revisions: Max number of hypothesis revisions
        timesteps_per_phase: RL training steps per hypothesis
        run_id: Unique ID (auto-generated if None)
    """
    init_db()
    run_id = run_id or f"v3_{world}_{uuid.uuid4().hex[:8]}"

    log_experiment(run_id, "v3_scientist", world, config={
        "revision_interval": revision_interval,
        "n_revisions": n_revisions,
        "timesteps_per_phase": timesteps_per_phase
    })

    agent = ScientistAgent(world=world, run_id=run_id)
    all_episode_rewards = []
    last_trajectory = []

    # Phase 0: Initial hypothesis
    base_env = make_world(world)
    response = agent.form_initial_hypothesis(base_env)
    base_env.close()

    episode_offset = 0
    
    # Create DQN model once - will be reused across all phases
    base_env = make_world(world)
    wrapped_env = CustomRewardWrapper(base_env, agent.current_reward_fn_code)
    model = DQN(
        "MlpPolicy", wrapped_env,
        learning_rate=1e-3, buffer_size=10_000,
        learning_starts=500, batch_size=64,
        gamma=0.99, exploration_fraction=0.3,
        exploration_final_eps=0.05,
        verbose=0, seed=42
    )
    wrapped_env.close()

    for phase in range(n_revisions + 1):
        if verbose:
            print(f"\n[V3] Phase {phase+1}/{n_revisions+1} | World: {world}")
            print(f"  Reward fn preview: {agent.current_reward_fn_code[:120]}...")

        # Set up environment with current custom reward
        base_env = make_world(world)
        wrapped_env = CustomRewardWrapper(base_env, agent.current_reward_fn_code)

        # Trajectory collector callback
        phase_rewards = []
        phase_trajectories = []
        current_traj = []

        class TrajectoryCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self._ep_reward = 0
                self._ep_steps = 0
                self._ep_num = episode_offset

            def _on_step(self):
                obs = self.locals["new_obs"][0]
                action = self.locals["actions"][0]
                reward = self.locals["rewards"][0]
                done = self.locals["dones"][0]

                current_traj.append((obs.copy(), int(action)))
                self._ep_reward += reward
                self._ep_steps += 1

                if done:
                    phase_rewards.append(self._ep_reward)
                    phase_trajectories.append(current_traj.copy())
                    log_episode(
                        run_id=run_id,
                        agent="v3_scientist",
                        world=world,
                        episode_num=self._ep_num,
                        total_reward=self._ep_reward,
                        steps=self._ep_steps,
                        terminated=True
                    )
                    current_traj.clear()
                    self._ep_reward = 0
                    self._ep_steps = 0
                    self._ep_num += 1

                return True

        cb = TrajectoryCallback()

        # Update environment with current reward function and set it on the model
        base_env = make_world(world)
        wrapped_env = CustomRewardWrapper(base_env, agent.current_reward_fn_code)
        model.set_env(wrapped_env)
        
        # Train on this phase without resetting timestep counter (accumulate learning across phases)
        model.learn(total_timesteps=timesteps_per_phase, callback=cb, reset_num_timesteps=False)

        wrapped_env.close()

        # Score and log this hypothesis
        if phase_trajectories:
            last_trajectory = phase_trajectories[-1]
        agent.score_and_log_hypothesis(episode_offset, last_trajectory)
        episode_offset += len(phase_rewards)
        all_episode_rewards.extend(phase_rewards)

        mean_r = np.mean(phase_rewards) if phase_rewards else 0
        if verbose:
            print(f"  Phase {phase+1} mean reward: {mean_r:.1f} over {len(phase_rewards)} episodes")

        # Check if solved (mean > 400 = near-perfect balance)
        if mean_r > 400:
            if verbose:
                print(f"  [V3] SOLVED in phase {phase+1}!")
            break

        # Revise if not last phase
        if phase < n_revisions:
            base_env = make_world(world)
            agent.revise_hypothesis(base_env, phase_rewards, last_trajectory)
            base_env.close()

    if verbose:
        print(f"\n[V3] Complete. Total episodes: {len(all_episode_rewards)}")
        print(f"  Hypotheses generated: {len(agent.hypothesis_history)}")
        print(f"  Final mean reward: {np.mean(all_episode_rewards[-50:]):.1f}")

    return {
        "run_id": run_id,
        "agent": "v3_scientist",
        "world": world,
        "total_episodes": len(all_episode_rewards),
        "final_mean_reward": float(np.mean(all_episode_rewards[-50:])) if all_episode_rewards else 0,
        "n_revisions": agent.revision_count,
        "hypothesis_history": agent.hypothesis_history
    }


if __name__ == "__main__":
    result = run(world="inverted", n_revisions=3, timesteps_per_phase=50_000)
    print("\nFinal result:", result)
