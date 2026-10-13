from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from kv_combo_swap import run_variants  # noqa: E402

VARIANTS = {
    "V_L11-17": {"k": [], "v": list(range(11, 18))},
}

if __name__ == "__main__":
    run_variants(VARIANTS, "step3_v_l11_17")
