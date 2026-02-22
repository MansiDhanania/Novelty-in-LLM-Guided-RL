"""
Quick script to extract metrics from database for README
"""
import sqlite3
from pathlib import Path

db_path = Path("logs/experiment.db")
if not db_path.exists():
    print("No database found")
    exit(1)

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

print("\n" + "="*80)
print("AGENT PERFORMANCE SUMMARY")
print("="*80)

# Overall performance by agent and world
cursor.execute("""
    SELECT agent, world, 
           COUNT(*) as episodes, 
           AVG(total_reward) as avg_reward,
           MAX(total_reward) as max_reward
    FROM episodes 
    GROUP BY agent, world 
    ORDER BY agent, world
""")

results = cursor.fetchall()
print(f"\n{'Agent':<25} {'World':<12} {'Episodes':>8} {'Avg Reward':>12} {'Max Reward':>12}")
print("-"*80)
for row in results:
    print(f"{row[0]:<25} {row[1]:<12} {row[2]:>8} {row[3]:>12.2f} {row[4]:>12.2f}")

# Novelty scores
print("\n" + "="*80)
print("NOVELTY/DIVERSITY METRICS")
print("="*80)

cursor.execute("""
    SELECT agent, 
           COUNT(*) as total_hypotheses,
           AVG(linguistic_surprise) as avg_novelty,
           MAX(linguistic_surprise) as max_novelty
    FROM novelty_scores 
    GROUP BY agent
    ORDER BY agent
""")

results = cursor.fetchall()
print(f"\n{'Agent':<25} {'Hypotheses':>12} {'Avg Distance':>15} {'Max Distance':>15}")
print("-"*80)
for row in results:
    print(f"{row[0]:<25} {row[1]:>12} {row[2]:>15.4f} {row[3]:>15.4f}")

# Boden type distribution
print("\n" + "="*80)
print("BODEN CREATIVITY TYPES")
print("="*80)

cursor.execute("""
    SELECT agent, boden_type, COUNT(*) as count
    FROM hypotheses
    WHERE boden_type IS NOT NULL
    GROUP BY agent, boden_type
    ORDER BY agent, boden_type
""")

results = cursor.fetchall()
print(f"\n{'Agent':<25} {'Boden Type':<20} {'Count':>10}")
print("-"*80)
for row in results:
    print(f"{row[0]:<25} {row[1]:<20} {row[2]:>10}")

conn.close()
