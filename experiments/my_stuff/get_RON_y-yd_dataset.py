# =========================================================
# Setup
# =========================================================

# Imports
import numpy as np
import torch
import time
import joblib
from acds.archetypes import (
    DeepReservoir,
    RandomizedOscillatorsNetwork,
    PhysicallyImplementableRandomizedOscillatorsNetwork,
    MultistablePhysicallyImplementableRandomizedOscillatorsNetwork,
)
from torchvision import transforms, datasets
from sklearn import preprocessing
import matplotlib.pyplot as plt
from pathlib import Path

# Choose device (CPU/GPU)
device = (torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

# Get relevant paths
curr_dir = Path(__file__).parent                           # current folder
model_dir = Path(curr_dir/'results/trained_architectures') # folder with the trained architectures
imgs_dir = Path('src/acds/benchmarks/raw')                 # folder with data
save_dataset_dir = Path(curr_dir/'results/other')          # folder to save the created dataset
plots_dir = curr_dir/'plots'/Path(__file__).stem           # folder to save plots
plots_dir.mkdir(parents=True, exist_ok=True)

# Function to compute forward dynamics
def forw_dynamics(u, y, yd, gamma, epsilon, W, V, b):
    """
    Forward dynamics of the RON reservoir. 

    Args
    ----
        u : shape (batch_size, n_input)
            Input of the model.
        y : shape (batch_size, n_hid)
            Hidden state of the model.
        yd : shape (batch_size, n_hid)
            Hidden state derivative.
        gamma : shape (n_hid,)
            Stifness.
        epsilon : shape (n_hid,)
            Damping.
        W : shape (n_hid, n_hid)
            Hidden-to-hidden matrix.
        V : shape (n_hid, n_input)
            Input-to-hidden matrix.
        b : shape (n_hid,)
            Bias vector.
    Returns
    -------
        ydd : shape (batch_size, n_hid)
            Hidden state second derivative.
    """
    ydd = - gamma * y - epsilon * yd + torch.tanh(y @ W + u @ V + b)
    return ydd


# =========================================================
# Create dataset
# =========================================================

# Parameters
n_inp = 1   # input dimension
n_hid = 6   # hidden states
m = 1000000 # dimension of the dataset

# Sample m random configurations (y, yd). To have an idea about the ranges, take a look at get_RON_dynamics.py
u = torch.zeros((m, 1), device=device)        # input (null in our case)
y = -2 + 4*torch.rand((m, 6), device=device)  # m samples y = [y1, ..., yN]^T
yd = -2 + 4*torch.rand((m, 6), device=device) # m samples yd = [yd1, ..., ydN]^T

# Extract saved model parameters
model_params = torch.load(model_dir/"sMNIST_RON_full_6hidden/sMNIST_RON_full_6hidden_model_5.pt", map_location=device)
gamma = model_params["gamma"]
epsilon= model_params["epsilon"]
W = model_params["h2h"]
V = model_params["x2h"]
b = model_params["bias"]

# Compute labels ydd
start = time.perf_counter()
ydd = forw_dynamics(u, y, yd, gamma, epsilon, W, V, b)
end = time.perf_counter()
print(f'Dataset generated in {(end-start):.6f} s')

# Save everything as numpy
np.savez(
    save_dataset_dir/'dataset_y_yd_ydd.npz', 
    y = y.cpu().numpy(), 
    yd = yd.cpu().numpy(), 
    ydd = ydd.cpu().numpy(),
)

###############################################################
########## !! Check: compare with built-in solver !! ##########

# Create an object of the reservoir
dt = 0.042
gamma = (2.7 - 1 / 2.0, 2.7 + 1 / 2.0)
epsilon = (0.51 - 0.5 / 2.0, 0.51 + 0.5 / 2.0)

model = RandomizedOscillatorsNetwork(
    n_inp=n_inp,
    n_hid=n_hid,
    dt=dt,
    gamma=gamma,
    epsilon=epsilon,
    diffusive_gamma=0.0,
    rho=9,
    input_scaling=1.0,
    topology='full',
    reservoir_scaler=1.0,
    sparsity=0.0,
    device=device,
).to(device)

# Assign saved parameters to the reservoir
model.load_state_dict(model_params)
model.eval()

# Evaluate forward pass
y_next, yd_next = model.cell(u, y, yd)

# Compute derivative
ydd_check = (yd_next - yd) / dt

# Check
assert torch.any(torch.abs(ydd_check - ydd) < 1e-14), 'Something wrong'

########## !! End check !! ####################################
###############################################################


# =========================================================
# Visualize dataset
# =========================================================

# Plot phase space (y, yd) to see distribution of the samples
if m < 1e6 + 1:
    fig, axs = plt.subplots(3,2, figsize=(12,9))
    for i, ax in enumerate(axs.flatten()):
        sc = ax.scatter(y.cpu().numpy()[:,i], yd.cpu().numpy()[:,i])
        ax.grid(True)
        ax.set_xlabel('y')
        ax.set_ylabel('yd')
        ax.set_title(f'samples hidden state {i+1}')
    plt.tight_layout()
    plt.savefig(plots_dir/'state_space', bbox_inches='tight')
    plt.show()