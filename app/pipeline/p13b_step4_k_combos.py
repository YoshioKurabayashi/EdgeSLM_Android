from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from kv_combo_swap import run_variants  # noqa: E402

VARIANTS = {
    "K_L0-6+L7-13": {"k": list(range(0, 7)) + list(range(7, 14)), "v": []},
    "K_L0-6+L14-20": {"k": list(range(0, 7)) + list(range(14, 21)), "v": []},
    "K_L0-6+L21-27": {"k": list(range(0, 7)) + list(range(21, 28)), "v": []},
}

if __name__ == "__main__":
    run_variants(VARIANTS, "step4_k_combos")
