from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from kv_combo_swap import run_variants  # noqa: E402

VARIANTS = {
    "V_L7-20": {"k": [], "v": list(range(7, 21))},
}

if __name__ == "__main__":
    run_variants(VARIANTS, "step1_v_l7_20")
