from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from kv_combo_swap import run_variants  # noqa: E402

VARIANTS = {
    "V_L7-10": {"k": [], "v": list(range(7, 11))},
    "V_L11-13": {"k": [], "v": list(range(11, 14))},
    "V_L14-17": {"k": [], "v": list(range(14, 18))},
    "V_L18-20": {"k": [], "v": list(range(18, 21))},
}

if __name__ == "__main__":
    run_variants(VARIANTS, "step2_v_split4")
