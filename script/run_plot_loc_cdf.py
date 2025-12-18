import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
plot_script = ROOT / "plot" / "plot_loc_cdf.py"

cmd = [sys.executable, str(plot_script)]

print("Running:", " ".join(cmd))
result = subprocess.run(cmd, check=False)
raise SystemExit(result.returncode)
