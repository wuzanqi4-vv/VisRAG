import sqlite3
db_path = "/home/wuzanqi/VisRAG/artifacts/baseline_eval/runs/nation_standard_bench/visrag/top5_23ae66dd7ad8da2a4acf/state.sqlite3"
conn = sqlite3.connect(db_path)
total, done = conn.execute("SELECT COUNT(1), COUNT(result) FROM questions").fetchone()
attempts = conn.execute("SELECT COUNT(1) FROM attempts").fetchone()[0]
latest = conn.execute("SELECT key FROM questions WHERE result IS NOT NULL ORDER BY rowid DESC LIMIT 1").fetchone()
latest_key = latest[0] if latest else "None"
print(f"Total: {total} | Completed: {done} | Attempts: {attempts} | Latest: {latest_key}")
