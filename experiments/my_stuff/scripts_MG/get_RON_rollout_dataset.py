import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rollout_dataset_utils import collect_rollout_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Mackey-Glass RON trajectories.")
    parser.add_argument("--n-hid", type=int, choices=(6, 9, 12, 15), default=6)
    parser.add_argument("--checkpoint-index", type=int, default=0)
    parser.add_argument("--n-trajectories", type=int, default=2048)
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    collect_rollout_dataset(
        task="MG",
        n_hid=args.n_hid,
        n_inp=1,
        dt=0.1,
        input_range=(-0.6, 0.4),
        y_range=(-0.4, 0.4),
        yd_range=(-0.8, 0.8),
        checkpoint_index=args.checkpoint_index,
        n_trajectories=args.n_trajectories,
        n_steps=args.n_steps,
        seed=args.seed,
        overwrite=args.overwrite,
    )
