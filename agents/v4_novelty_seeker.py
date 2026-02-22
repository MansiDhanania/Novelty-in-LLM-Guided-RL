"""
v4_novelty_seeker.py
V4: The Novelty Seeker

V3 + ENFORCED novelty pressure via rejection sampling.

Key differences from V3:
1. After generating a hypothesis, we compute its embedding distance BEFORE accepting it.
2. Distance is checked against ALL previous hypotheses (not just the most recent one).
3. If distance < MIN_NOVELTY_THRESHOLD, the hypothesis is REJECTED and LLM must retry.
4. Each retry shows the LLM its novelty score and a scaled penalty so it understands
   WHY it was rejected and how far off it is.
5. Accepted hypotheses get a novelty bonus fed back into the prompt as positive reinforcement.

Novelty scoring system (updated based on empirical research with all-MiniLM-L6-v2):
  Δ >= 0.70  → Novelty score: 100/100  (transformational leap)
  Δ 0.50-0.70 → Novelty score: 80/100  (very different)
  Δ 0.30-0.50 → Novelty score: 60/100  (quite different) ← minimum to pass
  Δ 0.15-0.30 → Novelty score: 30/100  (somewhat similar) ← REJECTED
  Δ 0.00-0.15 → Novelty score:  0/100  (almost identical) ← REJECTED

Research benchmarks:
  - Distance ~0.10 = nearly identical meaning (~90% similar)
  - Distance ~0.33 = related/similar topics (~67% similar)
  - Distance ~0.60 = somewhat different concepts (~40% similar)
  - Distance ~0.80+ = completely unrelated domains (<20% similar)

This makes V4 architecturally distinct from V3, not just prompt-distinct.
"""

import uuid
import textwrap
import json
import re
import os
import numpy as np
from typing import Optional
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
from measurement.novelty_scorer import (
    score_hypothesis, fingerprint_trajectory,
    embed_hypothesis, compute_embedding_distance
)

load_dotenv()

AGENT_NAME = "v4_novelty_seeker"

# ── Novelty thresholds ───────────────────────────────────────────────────────
# Based on sentence-transformers research with all-MiniLM-L6-v2:
#   Distance ~0.10 (sim 0.90) = nearly identical meaning
#   Distance ~0.33 (sim 0.67) = related/similar topics  
#   Distance ~0.60 (sim 0.40) = somewhat different
#   Distance ~0.80+ (sim 0.20) = completely unrelated
MIN_NOVELTY_DISTANCE = 0.30   # below this → rejected, must retry
MAX_RETRIES = 3               # max attempts before accepting best available


def novelty_score_100(distance: float) -> int:
    """Convert embedding distance to 0-100 novelty score shown to LLM."""
    if distance >= 0.70:        # similarity < 0.30 = truly different domains
        return 100
    elif distance >= 0.50:      # similarity < 0.50 = different approaches
        return 80
    elif distance >= 0.30:      # similarity < 0.70 = moderately different
        return 60
    elif distance >= 0.15:      # similarity < 0.85 = similar but distinct
        return 30
    else:                       # similarity >= 0.85 = too similar
        return 0


def novelty_label(distance: float) -> str:
    if distance >= 0.70:
        return "transformational leap ✓✓✓"
    elif distance >= 0.50:
        return "very different ✓✓"
    elif distance >= 0.30:
        return "quite different ✓ (minimum passing score)"
    elif distance >= 0.15:
        return "somewhat similar ✗ REJECTED"
    else:
        return "almost identical ✗✗ REJECTED"


# ── Prompts ──────────────────────────────────────────────────────────────────

INITIAL_HYPOTHESIS_PROMPT = """You control a cart-pole system.
Observations: [cart_pos, cart_vel, pole_angle, pole_vel]

Telemetry from random episodes:
{telemetry}

Form a brief physics hypothesis (1-2 sentences). Generate ONE reward function.

REWARD DESIGN TIPS:
- Reward progress toward goals, not just survival.
- Try: reward if pole angle is DECREASING, or cart moves toward center.
- Use abs(pole_angle), max(), min() for smooth rewards.

Use \\n for newlines.

RESPOND WITH ONLY THIS JSON:
{{"hypothesis": "Your theory", "reward_function": "def custom_reward(obs, action, base_reward, terminated):\\n    cart_pos, cart_vel, pole_angle, pole_vel = obs\\n    if terminated: return -5\\n    return 0.5 - abs(pole_angle)"}}"""


NOVELTY_REVISION_PROMPT = """Your theory failed.
Performance: mean={mean_reward:.1f}, best={best_reward:.1f}
Problem: {diagnosis}

━━━ NOVELTY SCORING SYSTEM ━━━
Your hypothesis will be scored 0-100 for novelty before being accepted.
Scoring rules:
  Δ ≥ 0.70 → 100/100  (transformational — completely new framing)
  Δ ≥ 0.50 →  80/100  (very different — new causal mechanism)
  Δ ≥ 0.30 →  60/100  (quite different — MINIMUM TO PASS)
  Δ ≥ 0.15 →  30/100  (somewhat similar — REJECTED, will retry)
  Δ < 0.15 →   0/100  (almost identical — REJECTED, will retry)

You will NOT advance until your novelty score is ≥ 60/100.

━━━ YOUR PREVIOUS THEORIES (DO NOT REPEAT) ━━━
{previous_hypotheses}

━━━ YOUR NOVELTY SCORES SO FAR ━━━
{novelty_scores}

━━━ YOUR TASK ━━━
Generate a hypothesis with novelty score ≥ 60 (Δ ≥ 0.30 from all previous).

To get a high score:
- Pick a DIFFERENT physical variable as your focus (if you used angle → use velocity or momentum)
- Try a completely different causal mechanism
- Consider: energy conservation, phase relationships, oscillation frequency, damping
- If you used penalties → try only positive shaping rewards
- If you focused on pole → try focusing on cart dynamics instead

Respond with 1-2 sentences and a new reward function.
Use \\n for newlines. Use abs(), max(), min().

RESPOND WITH ONLY THIS JSON:
{{"hypothesis": "Your fundamentally different theory", "reward_function": "def custom_reward(obs, action, base_reward, terminated):\\n    cart_pos, cart_vel, pole_angle, pole_vel = obs\\n    if terminated: return -5\\n    return 0.5 - abs(pole_angle)"}}"""


REJECTION_PROMPT = """NOVELTY SCORE: {score}/100 — {label}

Your hypothesis: "{hypothesis}"
Was too similar to previous theories (Δ={distance:.3f}, need Δ≥{threshold:.2f}).

You must try again with a MORE DIFFERENT approach.

━━━ WHAT YOU ALREADY TRIED ━━━
{previous_hypotheses}

━━━ HINT ━━━
{hint}

Generate a hypothesis that focuses on a COMPLETELY DIFFERENT aspect of the physics.
Use \\n for newlines.

RESPOND WITH ONLY THIS JSON:
{{"hypothesis": "Your new theory", "reward_function": "def custom_reward(obs, action, base_reward, terminated):\\n    cart_pos, cart_vel, pole_angle, pole_vel = obs\\n    if terminated: return -5\\n    return 0.5 - abs(pole_angle)"}}"""


REJECTION_HINTS = [
    "Try focusing on cart VELOCITY and MOMENTUM instead of pole angle.",
    "Try using energy: kinetic energy = 0.5 * velocity^2. Reward minimizing total system energy.",
    "Try focusing on the RATE OF CHANGE: reward when abs(pole_vel) is decreasing over time.",
    "Ignore the pole entirely. Design a reward purely based on cart position staying centered.",
    "Try rewarding OSCILLATION DAMPING: reward when the system's total motion is decreasing.",
]


class HypothesisResponse(BaseModel):
    hypothesis: str = Field(default="Unable to determine")
    reward_function: str = Field(
        default="def custom_reward(obs, action, base_reward, terminated):\n    return base_reward"
    )


def _diagnose_failure(mean_reward: float, best_reward: float) -> str:
    if mean_reward < 15:
        return "completely ineffective — pole falls almost immediately"
    elif mean_reward < 50:
        return "poor — agent makes progress but fails early"
    elif mean_reward < 150:
        return "mediocre — short bursts but can't sustain"
    elif best_reward > 400:
        return "inconsistent — sometimes excellent but not reliably stable"
    else:
        return "suboptimal — better than random but not solving"


# ── Reward wrapper ────────────────────────────────────────────────────────────

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
            print(f"  [V4 Compile Error] {e}. Using base reward.")
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
            except Exception:
                reward = base_reward
                self._error_count += 1
                if not self._warned and self._error_count >= 3:
                    print("  [V4 WARNING] Reward fn crashing repeatedly.")
                    self._warned = True
        else:
            reward = base_reward
        return obs, reward, terminated, truncated, info


# ── V4 Agent ──────────────────────────────────────────────────────────────────

class NoveltySeeker:

    def __init__(self, world: str, run_id: str,
                 groq_model: str = "llama-3.1-8b-instant"):
        self.world = world
        self.run_id = run_id
        self.groq_model = groq_model
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        self.current_hypothesis: Optional[str] = None
        self.current_reward_fn_code: Optional[str] = None
        self.previous_embedding: Optional[list] = None
        self.all_embeddings: list = []  # Store ALL previous embeddings for strict novelty check
        self.hypothesis_history: list = []  # {"text": ..., "distance": ..., "score": ...}
        self.revision_count: int = 0
        self._hint_index: int = 0

    def _call_llm(self, prompt: str) -> dict:
        try:
            resp = self.client.chat.completions.create(
                model=self.groq_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=800
            )
            raw = resp.choices[0].message.content
            if raw is None:
                raise ValueError("LLM returned None content")
            raw = raw.strip()
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                raw = json_match.group()
            parsed = json.loads(raw)
            validated = HypothesisResponse(**parsed)
            return validated.model_dump()
        except Exception as e:
            print(f"  [V4 LLM Error] {e}")
            return {
                "hypothesis": "Unable to parse response",
                "reward_function": "def custom_reward(obs, action, base_reward, terminated):\n    return base_reward"
            }

    def _score_candidate(self, hypothesis: str) -> tuple[float, dict]:
        """
        Score a candidate hypothesis against ALL previous hypotheses.
        Returns (min_distance_to_any_previous, full_scores_dict).
        This ensures novelty relative to the ENTIRE history, not just the last one.
        """
        current_embedding = embed_hypothesis(hypothesis)
        
        if not self.all_embeddings:
            # First hypothesis - nothing to compare against
            return 0.0, {"embedding": current_embedding, "embedding_distance": 0.0}
        
        # Check against ALL previous embeddings, take the minimum (most similar)
        distances = [
            compute_embedding_distance(current_embedding, prev_emb)
            for prev_emb in self.all_embeddings
        ]
        min_distance = min(distances)
        
        return min_distance, {"embedding": current_embedding, "embedding_distance": min_distance}

    def _format_history(self) -> tuple[str, str]:
        """Format previous hypotheses and novelty scores for prompt."""
        prev_lines = []
        score_lines = []
        for i, h in enumerate(self.hypothesis_history):
            prev_lines.append(f"  Theory {i+1}: \"{h['text'][:100]}\"")
            score = h.get("score", 0)
            dist = h.get("distance", 0.0)
            label = novelty_label(dist)
            score_lines.append(f"  Theory {i+1}: {score}/100 (Δ={dist:.3f}) — {label}")
        return (
            "\n".join(prev_lines) or "  (none yet)",
            "\n".join(score_lines) or "  (none yet)"
        )

    def form_initial_hypothesis(self, env) -> dict:
        """Collect telemetry, form initial hypothesis. No novelty check on first one."""
        print(f"\n[V4] Collecting telemetry from {self.world}...")
        telemetry_lines = []
        for ep in range(5):
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
                f"  Ep {ep+1}: reward={sum(ep_rewards):.1f}, steps={len(ep_rewards)}, "
                f"mean_obs=[{', '.join(f'{v:.3f}' for v in mean_obs)}]"
            )

        prompt = INITIAL_HYPOTHESIS_PROMPT.format(telemetry="\n".join(telemetry_lines))
        response = self._call_llm(prompt)
        self.current_hypothesis = response["hypothesis"]
        self.current_reward_fn_code = response["reward_function"]
        print(f"  [V4] Initial hypothesis: {(self.current_hypothesis or '')[:100]}...")
        return response

    def revise_with_enforced_novelty(self, phase_rewards: list) -> dict:
        """
        Generate a revision hypothesis, enforce novelty via rejection sampling.
        
        Loop:
          1. Generate candidate hypothesis
          2. Score it (embedding distance from previous)
          3. If score < MIN_NOVELTY_DISTANCE → show penalty score → retry
          4. Accept after MAX_RETRIES even if not novel enough (log warning)
        """
        self.revision_count += 1
        mean_r = float(np.mean(phase_rewards)) if phase_rewards else 0.0
        best_r = float(max(phase_rewards)) if phase_rewards else 0.0
        diagnosis = _diagnose_failure(mean_r, best_r)

        prev_hypotheses, novelty_scores = self._format_history()

        # First attempt — full revision prompt
        prompt = NOVELTY_REVISION_PROMPT.format(
            mean_reward=mean_r,
            best_reward=best_r,
            diagnosis=diagnosis,
            previous_hypotheses=prev_hypotheses,
            novelty_scores=novelty_scores
        )

        best_response = None
        best_distance = -1.0
        accepted = False
        response = None  # Initialize to avoid unbound variable

        for attempt in range(MAX_RETRIES):
            response = self._call_llm(prompt)
            hypothesis = response["hypothesis"]

            if hypothesis.startswith("Unable to parse"):
                print(f"  [V4] LLM parse failed on attempt {attempt+1}, retrying...")
                # Still track this as best if we have nothing else
                if best_response is None:
                    best_response = response
                    best_distance = 0.0
                continue

            # Score the candidate
            distance, _ = self._score_candidate(hypothesis)
            score = novelty_score_100(distance)
            label = novelty_label(distance)

            print(f"  [V4] Attempt {attempt+1}: Δ={distance:.3f} → {score}/100 — {label}")

            # Track best candidate in case all are rejected
            if distance > best_distance:
                best_distance = distance
                best_response = response

            if distance >= MIN_NOVELTY_DISTANCE:
                # Accepted — tell user and break
                print(f"  [V4] ✓ Accepted (novelty score {score}/100)")
                accepted = True
                break
            else:
                # Rejected — build rejection prompt with penalty shown explicitly
                if attempt < MAX_RETRIES - 1:
                    hint = REJECTION_HINTS[self._hint_index % len(REJECTION_HINTS)]
                    self._hint_index += 1
                    prompt = REJECTION_PROMPT.format(
                        score=score,
                        label=label,
                        hypothesis=hypothesis[:120],
                        distance=distance,
                        threshold=MIN_NOVELTY_DISTANCE,
                        previous_hypotheses=prev_hypotheses,
                        hint=hint
                    )
                    print(f"  [V4] ✗ Rejected — retrying with penalty feedback...")

        # Ensure we have a valid response
        if best_response is None:
            # All attempts failed catastrophically
            print(f"  [V4] ⚠ All {MAX_RETRIES} attempts failed. Using fallback hypothesis.")
            best_response = {
                "hypothesis": "Unable to determine physics model after multiple attempts",
                "reward_function": "def custom_reward(obs, action, base_reward, terminated):\n    return base_reward"
            }
            best_distance = 0.0
        elif not accepted:
            print(f"  [V4] ⚠ All {MAX_RETRIES} attempts below threshold. "
                  f"Accepting best (Δ={best_distance:.3f}).")

        final = best_response
        self.current_hypothesis = final["hypothesis"]
        self.current_reward_fn_code = final["reward_function"]
        print(f"  [V4] Revision {self.revision_count}: {(self.current_hypothesis or '')[:100]}...")
        return final

    def score_and_log(self, episode_num: int, trajectory: list):
        """Full Boden scoring + logging for accepted hypothesis."""
        # Ensure hypothesis was generated (should always be true if called after form/revise)
        assert self.current_hypothesis is not None, "current_hypothesis must be set before scoring"
        assert self.current_reward_fn_code is not None, "current_reward_fn_code must be set before scoring"
        
        scores = score_hypothesis(
            self.current_hypothesis,
            previous_embedding=self.previous_embedding,
            run_boden=True
        )
        behavioral_hash = fingerprint_trajectory(trajectory)
        distance = scores["embedding_distance"]
        score = novelty_score_100(distance)

        log_hypothesis(
            run_id=self.run_id,
            agent=AGENT_NAME,
            world=self.world,
            episode_num=episode_num,
            hypothesis_text=self.current_hypothesis,
            reward_fn_code=self.current_reward_fn_code,
            boden_type=scores["boden_type"],
            boden_justification=scores["boden_justification"],
            embedding=scores["embedding"],
            embedding_distance=distance
        )
        log_novelty_score(
            run_id=self.run_id,
            agent=AGENT_NAME,
            world=self.world,
            episode_num=episode_num,
            linguistic_surprise=distance,
            strategy_label=scores.get("strategy_label", "Unknown"),
            behavioral_hash=behavioral_hash
        )

        self.hypothesis_history.append({
            "text": self.current_hypothesis,
            "distance": distance,
            "score": score
        })
        self.previous_embedding = scores["embedding"]
        self.all_embeddings.append(scores["embedding"])  # Track ALL embeddings for strict novelty
        return scores


# ── Main run function ─────────────────────────────────────────────────────────

def run(world: str = "inverted",
        n_revisions: int = 4,
        timesteps_per_phase: int = 50_000,
        run_id: Optional[str] = None,
        verbose: bool = True) -> dict:
    """
    Train V4 with enforced novelty pressure.
    Same training protocol as V3 for fair comparison.
    """
    init_db()
    run_id = run_id or f"v4_{world}_{uuid.uuid4().hex[:8]}"

    log_experiment(run_id, AGENT_NAME, world, config={
        "n_revisions": n_revisions,
        "timesteps_per_phase": timesteps_per_phase,
        "novelty_pressure": "enforced",
        "min_novelty_distance": MIN_NOVELTY_DISTANCE,
        "max_retries": MAX_RETRIES,
        "temperature": 0.7
    })

    agent = NoveltySeeker(world=world, run_id=run_id)
    all_episode_rewards = []
    last_trajectory = []
    episode_offset = 0

    # Initial hypothesis (no novelty check)
    base_env = make_world(world)
    agent.form_initial_hypothesis(base_env)
    base_env.close()
    
    # Ensure hypothesis was successfully generated
    assert agent.current_hypothesis is not None, "Initial hypothesis generation failed"
    assert agent.current_reward_fn_code is not None, "Initial reward function generation failed"

    # Build DQN model once, reuse across phases
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
            print(f"\n[V4] Phase {phase+1}/{n_revisions+1} | World: {world}")

        base_env = make_world(world)
        wrapped_env = CustomRewardWrapper(base_env, agent.current_reward_fn_code)

        phase_rewards = []
        current_traj = []

        class TrajectoryCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self._ep_reward = 0.0
                self._ep_steps = 0
                self._ep_num = episode_offset

            def _on_step(self):
                obs = self.locals["new_obs"][0]
                action = self.locals["actions"][0]
                done = self.locals["dones"][0]

                current_traj.append((obs.copy(), int(self.locals["actions"][0])))
                self._ep_reward += self.locals["rewards"][0]
                self._ep_steps += 1

                if done:
                    phase_rewards.append(self._ep_reward)
                    nonlocal last_trajectory
                    last_trajectory = current_traj.copy()
                    log_episode(
                        run_id=run_id, agent=AGENT_NAME, world=world,
                        episode_num=self._ep_num, total_reward=self._ep_reward,
                        steps=self._ep_steps, terminated=True
                    )
                    current_traj.clear()
                    self._ep_reward = 0.0
                    self._ep_steps = 0
                    self._ep_num += 1
                return True

        cb = TrajectoryCallback()
        model.set_env(wrapped_env)
        model.learn(total_timesteps=timesteps_per_phase, callback=cb,
                    reset_num_timesteps=False)
        wrapped_env.close()

        agent.score_and_log(episode_offset, last_trajectory)
        episode_offset += len(phase_rewards)
        all_episode_rewards.extend(phase_rewards)

        mean_r = np.mean(phase_rewards) if phase_rewards else 0
        if verbose:
            print(f"  Phase {phase+1} mean reward: {mean_r:.1f} over {len(phase_rewards)} episodes")

        if mean_r > 400:
            if verbose:
                print(f"  [V4] SOLVED in phase {phase+1}!")
            break

        if phase < n_revisions:
            agent.revise_with_enforced_novelty(phase_rewards)

    final_mean = float(np.mean(all_episode_rewards[-50:])) if all_episode_rewards else 0.0

    if verbose:
        print(f"\n[V4] Complete. Total episodes: {len(all_episode_rewards)}")
        print(f"  Revisions: {agent.revision_count}")
        print(f"  Final mean reward: {final_mean:.1f}")
        novelty_summary = [(h['distance'], h['score']) for h in agent.hypothesis_history]
        print(f"  Novelty (Δ, score/100): {novelty_summary}")

    return {
        "run_id": run_id,
        "agent": AGENT_NAME,
        "world": world,
        "total_episodes": len(all_episode_rewards),
        "final_mean_reward": final_mean,
        "n_revisions": agent.revision_count,
        "hypothesis_history": agent.hypothesis_history
    }


if __name__ == "__main__":
    result = run(world="inverted", n_revisions=4, timesteps_per_phase=50_000)
    print("\nFinal result:", result)
