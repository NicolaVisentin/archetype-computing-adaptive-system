# =========================================================
# Setup
# =========================================================

# Imports
import numpy as np
import torch
import joblib
from tqdm import tqdm
from acds.archetypes import (
    DeepReservoir,
    RandomizedOscillatorsNetwork,
    PhysicallyImplementableRandomizedOscillatorsNetwork,
    MultistablePhysicallyImplementableRandomizedOscillatorsNetwork,
)
from acds.benchmarks import get_mackey_glass
from PIL import Image
from torchvision import transforms, datasets
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path

# Choose device (CPU/GPU)
device = (torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

# Get relevant paths
curr_dir = Path(__file__).parent                                                     # current folder
model_dir = Path(curr_dir.parent/'trained_architectures')                            # folder with the trained architectures to test
dataset_dir = Path('src/acds/benchmarks/raw')                                        # folder with datasets
plots_dir = curr_dir.parent/'plots'/curr_dir.stem/Path(__file__).stem                # folder to save plots
save_results_dir = Path(curr_dir.parent/'results'/curr_dir.stem/Path(__file__).stem) # folder to save data


# =========================================================
# Script settings
# =========================================================

n_hid = 12 # dimension of the hidden state (number of oscillators) (default RON: 1000)
architecture_to_test = 'MG_RON_full_12hidden_DT0.15' # path of the folder containing trained scaler, model and classifier
dt = 0.15 # dt of the RON reservoir (default RON: 0.17)
rho = 0.9 # spectral radius of the hidden-to-hidden weight matrix (defaul RON: 0.9)
inp_scaling = 10.0 # scaling for the input matrix (default RON: 10.0)
lag = 84
washout = 200
# !!! remember to choose the best model when loading the model, scaler and classifier !!!

# ------------

plots_dir = plots_dir/architecture_to_test
save_results_dir = save_results_dir/architecture_to_test
plots_dir.mkdir(parents=True, exist_ok=True)
save_results_dir.mkdir(parents=True, exist_ok=True)


# =========================================================
# Re-create the full, saved network
# =========================================================

# Create an object of the reservoir
n_inp = 1
gamma = (2.0 - 1 / 2.0, 2.0 + 1 / 2.0) # dummy
epsilon = (2.0 - 0.5 / 2.0, 2.0 + 0.5 / 2.0) # dummy

model = RandomizedOscillatorsNetwork(
    n_inp=n_inp,
    n_hid=n_hid,
    dt=dt,
    gamma=gamma,
    epsilon=epsilon,
    diffusive_gamma=0.0,
    rho=rho,
    input_scaling=inp_scaling,
    topology='full',
    reservoir_scaler=1.0,
    sparsity=0.0,
    device=device,
).to(device)

# Load and assign saved parameters to the reservoir (! this assignes only epsilon, gamma, h2h, x2h, bias. Other
# parameters must be initialized correctly !)
model_params = torch.load(model_dir/architecture_to_test/f"{architecture_to_test}_model_9.pt", map_location=device)
model.load_state_dict(model_params)
model.eval()

# Load saved scaler and classifier
scaler = joblib.load(model_dir/architecture_to_test/f"{architecture_to_test}_scaler_9.pkl")
classifier = joblib.load(model_dir/architecture_to_test/f"{architecture_to_test}_classifier_9.pkl")


# =========================================================
# Test on the test set
# =========================================================

# Function to test the trained classifier
@torch.no_grad()
def test(dataset, target, classifier, scaler):
    dataset = dataset.reshape(1, -1, 1).to(device)
    target = target.reshape(-1, 1).numpy()
    states_hist = model(dataset)[0].cpu().numpy() # reservoir's states evolution from k=0 to k=N-Nl-1. Shape (1, N-Nl, n_hid) array
    activations = states_hist[:, washout:] # remove the initial washout steps: reservoir's states evolution from k=Nw to k=N-Nl-1. Shape (1, N-Nl-Nw, n_hid)
    activations = activations.reshape(-1, n_hid) # shape (1, N-Nl-Nw, n_hid) -> (N-Nl-Nw, n_hid)
    activations_scaled = scaler.transform(activations)
    predictions = classifier.predict(activations_scaled) # predicted time sequence. Shape (N-Nl-Nw,)

    predictions = torch.from_numpy(predictions).float()
    target = torch.from_numpy(target.squeeze()).float()
    rmse = torch.sqrt(torch.mean((predictions - target) ** 2))
    rms_target = torch.sqrt(torch.mean(target ** 2))
    nrmse = (rmse / rms_target).item()
    return (
        nrmse, # nmrse(predictions, target)
        states_hist[0], # states evolution from k=0 to k=N-Nl-1. Shape (N-Nl, n_hid)
        activations, # states evolutions from k=Nw to k=N-Nl-1. Shape (N-Nl-Nw, n_hid)
        predictions, # predicted time sequence from k=Nw+Nl to k=N-1. Shape (N-Nl-Nw,)
        target.squeeze() # target time sequence from k=Nw+Nl to k=N-1. Shape (N-Nl-Nw,)
    )

# Build test dataset
_, _, (dataset_sequence, test_target) = get_mackey_glass(csvfolder=dataset_dir, lag=lag, washout=washout)

# Test on test set
test_nrmse, states_histories, activations, prediction, target = test(dataset_sequence, test_target, classifier, scaler)
print(f'NMRSE on the test set: {test_nrmse}')

# Visualize time sequences
Nw = washout
Nl = lag
N = len(target) + Nl + Nw

time = dt * np.arange(0, N)
full_sequence = np.concatenate([dataset_sequence.numpy(), target[-Nl:]])

plt.figure(figsize=(12,3))
plt.plot(time, full_sequence, 'k--', label='full sequence')
plt.plot(time[Nw:N-Nl], full_sequence[Nw:N-Nl], 'k', label='test sequence')
plt.plot(time[Nw+Nl:], prediction, 'r', label='predicted sequence')
plt.grid(True)
plt.xlabel('t [s]')
plt.ylabel('u')
plt.title('Mackey-Glass')
plt.legend()
plt.tight_layout()
plt.savefig(plots_dir/'prediction', bbox_inches='tight')
#plt.show()

# Show dynamics of the reservoir: states, velocities, accelerations and input in time (!! MAX FIRST 15 STATES !!)
if n_hid > 15:
    n_hid_show = 15
else:
    n_hid_show = n_hid

velocities_histories = np.diff(states_histories, axis=0) / dt # from k=0 to k=N-Nl-1-1. shape (N-Nl-1, n_hid)
accelerations_histories = np.diff(velocities_histories, axis=0) / dt # from k=0 to k=N-Nl-1-2. shape (N-Nl-2, n_hid)
input_history = dataset_sequence # from k=0 to k=N-Nl-1. shape (N-Nl,)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12,12))
for i in range(n_hid_show):
    ax1.plot(time[:N-Nl], states_histories[:,i], label=f'y{i+1}(t)')
    ax2.plot(time[:N-Nl-1], velocities_histories[:,i], label=f'yd{i+1}(t)')
    ax3.plot(time[:N-Nl-2], accelerations_histories[:,i], label=f'ydd{i+1}(t)')
ax4.plot(time[:N-Nl], input_history)
for ax in [ax1, ax2, ax3, ax4]:
    ax.grid(True)
    ax.set_xlabel('t [s]')
ax1.set_ylabel('y')
ax2.set_ylabel('yd')
ax3.set_ylabel('ydd')
ax4.set_ylabel('u')
ax1.set_title('Hidden states positions')
ax2.set_title('Hidden states velocities')
ax3.set_title('Hidden states accelerations')
ax4.set_title('Input')
ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
ax3.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
plt.tight_layout()
plt.savefig(plots_dir/'states_evolution', bbox_inches='tight')
#plt.show()

# Show dynamics of the reservoir: (y, yd) in y,yd plane
n_cols = min(3, n_hid_show)
n_rows = int(np.ceil(n_hid_show / n_cols))

fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 9))
if n_hid_show == 1:
    axs = np.array([axs])
else:
    axs = axs.flatten()

for i in range(n_hid_show):
    sc = axs[i].scatter(states_histories[:-1, i], velocities_histories[:, i], 
                        c=time[:N-Nl-1], cmap='viridis', s=10, alpha=0.6, label='t=0')
    axs[i].grid(True)
    axs[i].set_xlabel('y')
    axs[i].set_ylabel('yd')
    axs[i].set_title(f'Hidden state {i+1}')
    axs[i].legend(loc='upper left')

for i in range(n_hid, len(axs)):
    axs[i].set_visible(False)

plt.tight_layout()
plt.savefig(plots_dir/'state_space', bbox_inches='tight')
#plt.show()

# Show dynamics of the reservoir: y(t) (in separate plots)
fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 9))
if n_hid_show == 1:
    axs = np.array([axs])
else:
    axs = axs.flatten()

for i in range(n_hid_show):
    axs[i].plot(time[:N-Nl], states_histories[:,i])
    axs[i].grid(True)
    axs[i].set_xlabel('t [s]')
    axs[i].set_ylabel(r'$y_{i+1}$')
    axs[i].set_title(f'Component {i+1}')

for i in range(n_hid_show, len(axs)):
    axs[i].set_visible(False)

plt.tight_layout()
plt.savefig(plots_dir/'y_evoluation', bbox_inches='tight')
plt.show()

# Save dynamics of the reservoir
np.savez(
    save_results_dir/'RON_evolution.npz', 
    time = time[:N-Nl-2],
    y = states_histories[:-2], 
    yd = velocities_histories[:-1], 
    ydd = accelerations_histories[:],
    u = input_history[:-2]
)
