from pathlib import Path

import numpy as np
import torch
from acds.archetypes import RandomizedOscillatorsNetwork


def collect_rollout_dataset(
    *,
    task: str,
    n_hid: int,
    n_inp: int,
    dt: float,
    input_range: tuple[float, float],
    y_range: tuple[float, float],
    yd_range: tuple[float, float],
    checkpoint_index: int,
    n_trajectories: int = 2048,
    n_steps: int = 64,
    input_knot_stride: int = 8,
    batch_size: int = 128,
    seed: int = 123,
    overwrite: bool = False,
) -> Path:
    """Generate RON input/state trajectories and save the rollout dataset."""
    if n_hid < 1 or n_inp < 1:
        raise ValueError("n_hid and n_inp must be positive.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if n_trajectories < 1:
        raise ValueError("n_trajectories must be positive.")
    if n_steps < 2:
        raise ValueError("n_steps must be at least two.")
    if input_knot_stride < 1:
        raise ValueError("input_knot_stride must be positive.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    script_dir = Path(__file__).parent
    experiment_dir = script_dir.parent
    model_dir = experiment_dir / "my_stuff" / "trained_architectures"

    architecture = f"{task}_RON_full_{n_hid}hidden"
    checkpoint_path = (
        model_dir
        / architecture
        / f"{architecture}_model_{checkpoint_index}.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"RON checkpoint not found: {checkpoint_path}")

    task_script_dir = {
        "Adiac": "scripts_Adiac",
        "MG": "scripts_MG",
        "lorenz": "scripts_Lorenz96",
        "sMNIST": "scripts_sMNIST",
    }.get(task)
    if task_script_dir is None:
        raise ValueError(f"Unsupported RON task: {task}")
    output_path = (
        script_dir
        / "results"
        / task_script_dir
        / "get_RON_rollout_dataset"
        / architecture
        / f"trajectories_N{n_hid}.npz"
    )
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Dataset already exists: {output_path}. "
            "Set overwrite=True in this task script to replace it."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RandomizedOscillatorsNetwork(
        n_inp=n_inp,
        n_hid=n_hid,
        dt=dt,
        gamma=(2.7, 3.2),
        epsilon=(0.26, 0.76),
        diffusive_gamma=0.0,
        rho=0.99,
        input_scaling=1.0,
        topology="full",
        reservoir_scaler=1.0,
        sparsity=0.0,
        device=device,
    ).to(device)
    model_params = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(model_params)
    model.eval()

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    y_min, y_max = y_range
    yd_min, yd_max = yd_range
    u_min, u_max = input_range
    times = np.arange(n_steps, dtype=np.float64) * dt
    y_data = np.empty((n_trajectories, n_steps, n_hid), dtype=np.float32)
    yd_data = np.empty_like(y_data)
    u_data = np.empty((n_trajectories, n_steps, n_inp), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n_trajectories, batch_size):
            stop = min(start + batch_size, n_trajectories)
            current_batch_size = stop - start
            n_input_knots = (
                (n_steps - 1 + input_knot_stride - 1) // input_knot_stride + 1
            )
            u_knots = u_min + (u_max - u_min) * torch.rand(
                (current_batch_size, n_input_knots, n_inp),
                device=device,
                generator=generator,
            )
            u_batch = torch.nn.functional.interpolate(
                u_knots.transpose(1, 2),
                size=n_steps,
                mode="linear",
                align_corners=True,
            ).transpose(1, 2)
            y_state = y_min + (y_max - y_min) * torch.rand(
                (current_batch_size, n_hid),
                device=device,
                generator=generator,
            )
            yd_state = yd_min + (yd_max - yd_min) * torch.rand(
                (current_batch_size, n_hid),
                device=device,
                generator=generator,
            )
            y_batch = torch.empty(
                (current_batch_size, n_steps, n_hid), device=device
            )
            yd_batch = torch.empty_like(y_batch)

            for step in range(n_steps):
                y_batch[:, step] = y_state
                yd_batch[:, step] = yd_state
                if step + 1 < n_steps:
                    y_state, yd_state = model.cell(
                        u_batch[:, step], y_state, yd_state
                    )

            y_data[start:stop] = y_batch.cpu().numpy()
            yd_data[start:stop] = yd_batch.cpu().numpy()
            u_data[start:stop] = u_batch.cpu().numpy()

            print(f"Generated trajectories {stop}/{n_trajectories}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        time=times,
        u=u_data,
        y=y_data,
        yd=yd_data,
        task=np.asarray(task),
        architecture=np.asarray(architecture),
        checkpoint_index=np.asarray(checkpoint_index),
        dt=np.asarray(dt),
        seed=np.asarray(seed),
        input_range=np.asarray(input_range, dtype=np.float32),
        y_range=np.asarray(y_range, dtype=np.float32),
        yd_range=np.asarray(yd_range, dtype=np.float32),
    )
    print(f"Saved {n_trajectories} trajectories to {output_path}")
    return output_path
