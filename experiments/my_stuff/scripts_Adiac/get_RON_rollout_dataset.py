import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rollout_dataset_utils import collect_rollout_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Adiac RON trajectories.")
    parser.add_argument("--n-hid", type=int, choices=(6, 9, 12, 15), default=6)
    parser.add_argument("--checkpoint-index", type=int, default=5)
    parser.add_argument("--n-trajectories", type=int, default=2048)
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    collect_rollout_dataset(
        task="Adiac",
        n_hid=args.n_hid,
        n_inp=1,
        dt=0.02,
        input_range=(-2.1, 2.7),
        y_range=(-0.3, 0.3),
        yd_range=(-0.25, 0.25),
        checkpoint_index=args.checkpoint_index,
        n_trajectories=args.n_trajectories,
        n_steps=args.n_steps,
        seed=args.seed,
        overwrite=args.overwrite,
    )
