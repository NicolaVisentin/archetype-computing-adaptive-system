import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rollout_dataset_utils import collect_rollout_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sMNIST RON trajectories.")
    parser.add_argument("--n-hid", type=int, choices=(6, 9, 12, 15), default=15)
    parser.add_argument("--checkpoint-index", type=int, default=4)
    parser.add_argument("--n-trajectories", type=int, default=2048)
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    collect_rollout_dataset(
        task="sMNIST",
        n_hid=args.n_hid,
        n_inp=1,
        dt=0.01,
        input_range=(0.0, 1.0),
        y_range=(-2.5, 2.5),
        yd_range=(-3.5, 3.5),
        checkpoint_index=args.checkpoint_index,
        n_trajectories=args.n_trajectories,
        n_steps=args.n_steps,
        seed=args.seed,
        overwrite=args.overwrite,
    )
