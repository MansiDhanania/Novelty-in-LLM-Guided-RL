"""
novelty_scorer.py
The measurement heart of the project.

Three novelty signals:
1. Linguistic Surprise    — embedding cosine distance between consecutive hypotheses
2. Boden Classification   — LLM classifies its own strategy into Boden's taxonomy
3. Behavioral Fingerprint — UMAP clustering of state-action trajectories
"""

import json
import hashlib
import numpy as np
from typing import Optional
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity

# Lazy imports (heavy, only load when needed)
_sentence_model = None
_groq_client = None


def _get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        # all-MiniLM-L6-v2: tiny (80MB), fast on CPU, good semantic quality
        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sentence_model


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client


# ── 1. Linguistic Surprise ─────────────────────────────────────────────────

def embed_hypothesis(hypothesis_text: str) -> list:
    """Convert a hypothesis string to a dense embedding vector."""
    model = _get_sentence_model()
    embedding = model.encode(hypothesis_text, normalize_embeddings=True)
    return embedding.tolist()


def compute_embedding_distance(embedding_a: list, embedding_b: list) -> float:
    """
    Cosine distance between two embeddings.
    0.0 = identical theories, 1.0 = completely different theories.
    Returns 0.0 if either embedding is None (first hypothesis has no prior).
    """
    if embedding_a is None or embedding_b is None:
        return 0.0
    a = np.array(embedding_a).reshape(1, -1)
    b = np.array(embedding_b).reshape(1, -1)
    similarity = cosine_similarity(a, b)[0][0]
    return float(1.0 - similarity)


# ── 2. Boden Classification ────────────────────────────────────────────────

BODEN_PROMPT = """You are a creativity researcher analyzing an AI agent's strategy.

The agent is trying to balance a pole in a physics-deformed environment.
Here is the strategy/hypothesis the agent just generated:

---
{hypothesis}
---

Classify this strategy using Margaret Boden's three types of creativity:

1. COMBINATIONAL: Combines existing known strategies in a new way. 
   Example: "Try the standard balancing approach but inverted."

2. EXPLORATORY: Explores the edges of an existing conceptual space.
   Example: "Discovered that extreme oscillation stabilizes better than micro-corrections."

3. TRANSFORMATIONAL: Genuinely reframes the problem or invents a new conceptual framework.
   Example: "Realized the goal is not balance but controlled falling in a new direction."

Respond ONLY with valid JSON in this exact format:
{{
  "boden_type": "combinational" | "exploratory" | "transformational",
  "confidence": 0.0-1.0,
  "justification": "One sentence explaining why.",
  "strategy_label": "A short memorable name for this strategy (e.g. 'Inverse Pendulum Flip')"
}}"""


def classify_boden(hypothesis_text: str) -> dict:
    """
    Ask the LLM to classify a hypothesis into Boden's taxonomy.
    Returns dict with boden_type, confidence, justification, strategy_label.
    """
    client = _get_groq_client()

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "user", "content": BODEN_PROMPT.format(hypothesis=hypothesis_text)}
            ],
            temperature=0.2,
            max_tokens=300
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        result = json.loads(raw)
        return result

    except (json.JSONDecodeError, KeyError, Exception) as e:
        # Graceful fallback — never crash the experiment
        return {
            "boden_type": "combinational",
            "confidence": 0.0,
            "justification": f"Classification failed: {str(e)}",
            "strategy_label": "Unknown"
        }


# ── 3. Behavioral Fingerprinting ───────────────────────────────────────────

def fingerprint_trajectory(trajectory: list) -> str:
    """
    Convert a list of (obs, action) tuples into a compact behavioral hash.
    Used to detect when two agents with different reasoning arrived at
    structurally similar behaviors.

    trajectory: list of (obs_array, action_int) tuples from one episode
    """
    if not trajectory:
        return "empty"

    # Discretize observations into bins to reduce noise sensitivity
    binned = []
    for obs, action in trajectory:
        obs_arr = np.array(obs)
        # Bin each dimension into 10 buckets between [-3, 3]
        binned_obs = np.digitize(np.clip(obs_arr, -3, 3), bins=np.linspace(-3, 3, 10))
        binned.append((*binned_obs.tolist(), action))

    # Hash the binned sequence
    raw = json.dumps(binned, separators=(",", ":"))
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def compute_trajectory_embeddings_for_umap(all_trajectories: list) -> np.ndarray:
    """
    Convert a list of trajectories into fixed-length feature vectors
    suitable for UMAP dimensionality reduction.

    all_trajectories: list of lists of (obs, action) tuples
    Returns: np.ndarray of shape (n_trajectories, n_features)
    """
    features = []
    for trajectory in all_trajectories:
        if not trajectory:
            features.append(np.zeros(12))
            continue

        obs_list = [np.array(obs) for obs, _ in trajectory]
        actions = [a for _, a in trajectory]
        obs_arr = np.stack(obs_list)  # (steps, 4)

        feat = np.concatenate([
            obs_arr.mean(axis=0),          # mean state (4)
            obs_arr.std(axis=0),           # state variance (4)
            obs_arr.max(axis=0),           # max excursion (4)
            [np.mean(actions)],            # action bias (1: 0=left-heavy, 1=right-heavy)
            [len(trajectory) / 500.0],     # episode length normalized (1)
            [np.std(actions)],             # action diversity (1)
        ])
        features.append(feat)

    return np.array(features, dtype=np.float32)


# ── Combined scoring pipeline ──────────────────────────────────────────────

def score_hypothesis(
    hypothesis_text: str,
    previous_embedding: Optional[list] = None,
    run_boden: bool = True
) -> dict:
    """
    Full scoring pipeline for a single hypothesis.
    Returns everything needed for database logging.
    """
    embedding = embed_hypothesis(hypothesis_text)
    distance = compute_embedding_distance(previous_embedding, embedding)

    result = {
        "embedding": embedding,
        "embedding_distance": distance,
        "boden_type": None,
        "boden_confidence": None,
        "boden_justification": None,
        "strategy_label": None,
    }

    if run_boden:
        boden = classify_boden(hypothesis_text)
        result["boden_type"] = boden.get("boden_type")
        result["boden_confidence"] = boden.get("confidence")
        result["boden_justification"] = boden.get("justification")
        result["strategy_label"] = boden.get("strategy_label")

    return result


if __name__ == "__main__":
    # Quick test — doesn't need Groq key for embedding distance
    h1 = "The pole falls to the right, so I should push right to counteract."
    h2 = "The physics seem inverted. I must move opposite to the pole's direction."
    h3 = "I will use rapid oscillation to create a gyroscopic stabilization effect."

    e1 = embed_hypothesis(h1)
    e2 = embed_hypothesis(h2)
    e3 = embed_hypothesis(h3)

    print(f"H1→H2 distance (should be moderate): {compute_embedding_distance(e1, e2):.4f}")
    print(f"H1→H3 distance (should be higher):   {compute_embedding_distance(e1, e3):.4f}")
    print(f"H1→H1 distance (should be 0.0):      {compute_embedding_distance(e1, e1):.4f}")
