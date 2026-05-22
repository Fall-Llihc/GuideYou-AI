"""Smoke test: simulates running notebook 01 cells 1, 4, 5, 7, 9, 15
without network (Overpass) and with reduced RL episodes.

Run from project root:  python3 scripts/smoke_test.py
"""
import json
import sys
import io
import os
from pathlib import Path

# Move CWD to project root regardless of where it's launched from
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

NB_PATH = ROOT / "notebooks" / "01_recommendation_engine.ipynb"
nb = json.loads(NB_PATH.read_text())

# Pick the cells we want to execute (by index from the README spec)
TARGET_CELLS = [1, 4, 5, 7, 9, 12, 15]

# Patch: shrink RL episodes for the smoke test
def patch_source(src: str) -> str:
    src = src.replace("N_EPISODES = 3000", "N_EPISODES = 150")
    src = src.replace("rewards_log = train_rl_agent(env, rl_agent, n_episodes=N_EPISODES, log_interval=500)",
                       "rewards_log = train_rl_agent(env, rl_agent, n_episodes=N_EPISODES, log_interval=50)")
    # Strip Jupyter magics/shell
    keep = []
    for line in src.split("\n"):
        if line.lstrip().startswith(("!", "%")):
            continue
        keep.append(line)
    return "\n".join(keep)

# Also: stub out cell 3 (Overpass) by injecting an empty df_osm
stub_overpass = """
import pandas as pd
df_osm = pd.DataFrame(columns=['id','name','category','lat','lng','source'])
print('ℹ️  (smoke-test) skipped Overpass — using empty df_osm')
"""

# Disable matplotlib show + skip OSRM network
stub_pre = """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None

# Speed up RouteOptimizer: avoid live OSRM
import os as _os
_os.environ['SMOKE_TEST'] = '1'
"""

ns = {}
exec(stub_pre, ns)
exec(stub_overpass, ns)

for idx in TARGET_CELLS:
    cell = nb["cells"][idx]
    if cell["cell_type"] != "code":
        continue
    src = patch_source("".join(cell["source"]))
    # Disable OSRM in smoke test by setting use_osrm=False
    if "RouteOptimizer(use_osrm=True)" in src:
        src = src.replace("RouteOptimizer(use_osrm=True)", "RouteOptimizer(use_osrm=False)")
    print(f"\n{'='*60}\n▶ Executing cell {idx}\n{'='*60}")
    try:
        exec(src, ns)
    except Exception as e:
        print(f"❌ Cell {idx} runtime error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\n🎉 Smoke test PASSED")
