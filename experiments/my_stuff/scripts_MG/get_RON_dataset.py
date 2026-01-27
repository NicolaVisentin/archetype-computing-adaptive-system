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
curr_dir = Path(__file__).parent                                                     # current folder
model_dir = Path(curr_dir.parent/'trained_architectures')                            # folder with the trained architectures to test
imgs_dir = Path('src/acds/benchmarks/raw')                                           # folder with datasets
plots_dir = curr_dir.parent/'plots'/curr_dir.stem/Path(__file__).stem                # folder to save plots
save_results_dir = Path(curr_dir.parent/'results'/curr_dir.stem/Path(__file__).stem) # folder to save data


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
# Script settings
# =========================================================

n_hid = 6 # dimension of the hidden state (number of oscillators) (default RON: 1000)
architecture_to_test = 'MG_RON_full_6hidden_DT0.05_inpscal0.1' # path of the folder containing trained scaler, model and classifier
dt = 0.05 # dt of the RON reservoir (default RON: 0.17)
m = int(1e5) # dataset dimension (number of datapoints and labels)
y_range = [-0.2, 0.5] # range of positions to sample. To have an idea about the ranges, take a look at test_RON_model.py
yd_range = [-0.2, 0.4] # range of velocities to sample. To have an idea about the ranges, take a look at test_RON_model.py
# !!! remember to choose the best model when loading the model, scaler and classifier !!!

# ------------

plots_dir = plots_dir/architecture_to_test
save_results_dir = save_results_dir/architecture_to_test
plots_dir.mkdir(parents=True, exist_ok=True)
save_results_dir.mkdir(parents=True, exist_ok=True)


# =========================================================
# Create dataset
# =========================================================

# Parameters
n_inp = 1 # input dimension

# Sample m random configurations (y, yd, u). To have an idea about the ranges, take a look at test_RON_model.py
y_min, y_max = y_range
yd_min, yd_max = yd_range

u = -0.6 + (0.4 + 0.6) * torch.rand((m, n_inp), device=device) # m samples u. Shape (m, 1)
y = y_min + (y_max - y_min) * torch.rand((m, n_hid), device=device) # m samples y = [y1, ..., yN]^T. Shape (m, n_hidden)
yd = yd_min + (yd_max - yd_min) * torch.rand((m, n_hid), device=device) # m samples yd = [yd1, ..., ydN]^T. Shape (m, n_hidden)

# Extract saved model parameters
model_params = torch.load(model_dir/architecture_to_test/f"{architecture_to_test}_model_4.pt", map_location=device)
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
    save_results_dir/'dataset_y_yd_u_ydd.npz', 
    y = y.cpu().numpy(), 
    yd = yd.cpu().numpy(), 
    ydd = ydd.cpu().numpy(),
    u = u.cpu().numpy()
)

###############################################################
########## !! Check: compare with built-in solver !! ##########

# Create an object of the reservoir
gamma = (2.7 - 1 / 2.0, 2.7 + 1 / 2.0) # dummy
epsilon = (0.51 - 0.5 / 2.0, 0.51 + 0.5 / 2.0) # dummy

model = RandomizedOscillatorsNetwork(
    n_inp=n_inp,
    n_hid=n_hid,
    dt=dt,
    gamma=gamma,
    epsilon=epsilon,
    diffusive_gamma=0.0,
    rho=0.9,
    input_scaling=10.0,
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
    n_cols = min(3, n_hid)
    n_rows = int(np.ceil(n_hid / n_cols))
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 9))
    if n_hid == 1:
        axs = np.array([axs])
    else:
        axs = axs.flatten()
    
    for i in range(n_hid):
        sc = axs[i].scatter(y.cpu().numpy()[:, i], yd.cpu().numpy()[:, i], s=10, alpha=0.6)
        axs[i].grid(True)
        axs[i].set_xlabel('y')
        axs[i].set_ylabel('yd')
        axs[i].set_title(f'Samples hidden state {i+1}')
    
    for i in range(n_hid, len(axs)):
        axs[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(plots_dir/'state_space', bbox_inches='tight')
    plt.show()
