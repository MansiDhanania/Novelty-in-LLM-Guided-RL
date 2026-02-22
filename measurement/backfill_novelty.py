# backfill_novelty.py
# Run once to compute missing novelty scores from existing hypotheses
import sys, json
sys.path.insert(0, '.')
from measurement.database import get_hypotheses_df, log_novelty_score, init_db
from measurement.novelty_scorer import score_hypothesis, fingerprint_trajectory

init_db()
df = get_hypotheses_df()

# Find hypotheses that don't have novelty scores yet
from measurement.database import get_novelty_df
scored = get_novelty_df()
scored_ids = set(zip(scored['run_id'], scored['episode_num'])) if not scored.empty else set()

prev_embedding = {}  # keyed by (agent, world)

for _, row in df.sort_values(['agent','world','episode_num']).iterrows():
    key = (row['agent'], row['world'])
    prev_emb = prev_embedding.get(key)
    
    if row['embedding']:
        curr_emb = json.loads(row['embedding'])
        from measurement.novelty_scorer import compute_embedding_distance
        dist = compute_embedding_distance(prev_emb, curr_emb)
        
        log_novelty_score(
            run_id=row['run_id'],
            agent=row['agent'],
            world=row['world'],
            episode_num=row['episode_num'],
            linguistic_surprise=dist,
            strategy_label=row.get('boden_type') or 'unknown'
        )
        prev_embedding[key] = curr_emb

print("Backfill complete.")