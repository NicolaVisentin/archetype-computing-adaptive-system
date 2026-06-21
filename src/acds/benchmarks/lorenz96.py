import torch
import numpy as np
from scipy.integrate import odeint

def get_lorenz(dim, F, num_batch=128, lag=25, washout=200, window_size=0):
    # https://en.wikipedia.org/wiki/Lorenz_96_model
    def L96(x, t):
        """Lorenz 96 model with constant forcing"""
        # Setting up vector
        d = np.zeros(dim)
        # Loops over indices (with operations and Python underflow indexing handling edge cases)
        for i in range(dim):
            d[i] = (x[(i + 1) % dim] - x[i - 2]) * x[i - 1] - x[i] + F
        return d

    dt = 0.01
    t = np.arange(0.0, 20+(lag*dt)+(washout*dt), dt) # from t=0 to t=20 s + lag + whashout; from k=0 to k=N-1
    dataset = []
    for i in range(num_batch):
        x0 = np.random.rand(dim) + F - 0.5 # [F-0.5, F+0.5]
        x = odeint(L96, x0, t)
        dataset.append(x)
    dataset = np.stack(dataset, axis=0)
    dataset = torch.from_numpy(dataset).float()

    if window_size > 0:
        windows, targets = [], []
        for i in range(dataset.shape[0]):
            w, trg = get_fixed_length_windows(dataset[i], window_size, prediction_lag=lag)
        windows.append(w)
        targets.append(trg)
        return torch.utils.data.TensorDataset(torch.cat(windows, dim=0), torch.cat(targets, dim=0))
    else:
        return dataset
    

def get_fixed_length_windows(tensor, length, prediction_lag=1):
    assert len(tensor.shape) <= 2
    if len(tensor.shape) == 1:
        tensor = tensor.unsqueeze(-1)

    windows = tensor[:-prediction_lag].unfold(0, length, 1)
    windows = windows.permute(0, 2, 1)

    targets = tensor[length+prediction_lag-1:]
    return windows, targets  # input (B, L, I), target, (B, I)