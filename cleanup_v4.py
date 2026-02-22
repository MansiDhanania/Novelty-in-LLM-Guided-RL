"""
Cleanup script to remove old v4_novelty_seeker data from the database.
Run this before re-running v4 with the new novelty feedback implementation.
"""

import sqlite3
from pathlib import Path

db_path = Path("logs/experiment.db")

if not db_path.exists():
    print(f"Database not found at {db_path}")
    exit(1)

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# Delete all v4_novelty_seeker data
cursor.execute('DELETE FROM episodes WHERE agent="v4_novelty_seeker"')
episodes_deleted = cursor.rowcount

cursor.execute('DELETE FROM hypotheses WHERE agent="v4_novelty_seeker"')
hypotheses_deleted = cursor.rowcount

cursor.execute('DELETE FROM novelty_scores WHERE agent="v4_novelty_seeker"')
novelty_deleted = cursor.rowcount

cursor.execute('DELETE FROM experiments WHERE agent="v4_novelty_seeker"')
experiments_deleted = cursor.rowcount

conn.commit()
conn.close()

print(f"✓ V4 data cleared from database:")
print(f"  - {experiments_deleted} experiments")
print(f"  - {episodes_deleted} episodes")
print(f"  - {hypotheses_deleted} hypotheses")
print(f"  - {novelty_deleted} novelty scores")
print(f"\nYou can now run v4 again with the new feedback loop code.")
