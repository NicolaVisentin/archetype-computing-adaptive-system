# =========================================================
# Setup
# =========================================================

# Imports
import numpy as np
import torch
import joblib
from acds.archetypes import RandomizedOscillatorsNetwork
from acds.benchmarks import get_lorenz
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
plots_dir = curr_dir.parent/'plots'/curr_dir.stem/Path(__file__).stem                # folder to save plots
save_results_dir = Path(curr_dir.parent/'results'/curr_dir.stem/Path(__file__).stem) # folder to save data


# =========================================================
# Script settings
# =========================================================

n_hid = 6 # dimension of the hidden state (number of oscillators)
architecture_to_test = 'lorenz_RON_full_6hidden' # path of the folder containing trained scaler, model and predictor
dt = 0.05 # dt of the RON reservoir (default RON: 0.17 s)
rho = 0.99 # spectral radius of the hidden-to-hidden weight matrix (defaul RON: 0.99)
inp_scaling = 0.01 # scaling for the input matrix (default RON: 0.1)
lag = 25
washout = 200
# !!! remember to choose the best model when loading the model, scaler and predictor !!!

# ------------

plots_dir = plots_dir/architecture_to_test
save_results_dir = save_results_dir/architecture_to_test
plots_dir.mkdir(parents=True, exist_ok=True)
save_results_dir.mkdir(parents=True, exist_ok=True)


# =========================================================
# Re-create the full, saved network
# =========================================================

# Create an object of the reservoir
n_inp = 5
gamma = (2.0 - 1 / 2.0, 2.0 + 1 / 2.0) # dummy
epsilon = (2.0 - 0.5 / 2.0, 2.0 + 0.5 / 2.0) # dummy

model = RandomizedOscillatorsNetwork(
    n_inp=n_inp,
    n_hid=n_hid,
    dt=dt,
    gamma=gamma,
    epsilon=epsilon,
    rho=rho,
    input_scaling=inp_scaling,
    device=device,
).to(device)

# Load and assign saved parameters to the reservoir (! this assignes only epsilon, gamma, h2h, x2h, bias. Other
# parameters must be initialized correctly !)
model_params = torch.load(model_dir/architecture_to_test/f"{architecture_to_test}_model_6.pt", map_location=device)
model.load_state_dict(model_params)
model.eval()

# Load saved scaler and predictor
scaler = joblib.load(model_dir/architecture_to_test/f"{architecture_to_test}_scaler_6.pkl")
predictor = joblib.load(model_dir/architecture_to_test/f"{architecture_to_test}_predictor_6.pkl")


# =========================================================
# Test on the test set
# =========================================================

# Function to test the trained output layer
@torch.no_grad()
def test(dataset, predictor, scaler):
    datapoints = dataset[:, :(2000+washout)].to(device) # datapoints, from k=0 to k=N-Nl-1. Shape (B, N-Nl, n_inp)
    target_batched = dataset[:, (lag+washout):].cpu().numpy() # from k=Nw+Nl to k=N-1. Shape (B, N-Nw-Nl, n_inp)
    target = target_batched.reshape(-1, n_inp) # shape (B, N-Nw-Nl, n_inp) -> (B*(N-Nw-Nl), n_inp)
    states_hist = model(datapoints)[0].cpu().numpy() # reservoir's states evolution from k=0 to k=N-Nl-1. Shape (B, N-Nl, n_hid)
    activations_batched = states_hist[:, washout:] # remove the initial washout steps: reservoir's states evolution from k=Nw to k=N-Nl-1. Shape (B, N-Nl-Nw, n_hid)
    activations = activations_batched.reshape(-1, n_hid) # shape (B, N-Nl-Nw, n_hid) -> (B*(N-Nl-Nw), n_hid)
    activations_scaled = scaler.transform(activations)
    predictions = predictor.predict(activations_scaled)

    predictions = torch.from_numpy(predictions).float()
    target = torch.from_numpy(target.squeeze()).float()
    rmse = torch.sqrt(torch.mean((predictions - target) ** 2))
    rms_target = torch.sqrt(torch.mean(target ** 2))
    nrmse = (rmse / rms_target).item()

    B, T = activations_batched.shape[:2]  # B and N-Nl-Nw
    predictions_batched = predictions.reshape(B, T, -1)  # shape (B, N-Nl-Nw, n_inp)
    
    return (
        nrmse, # nmrse(predictions, target)
        states_hist, # states evolution from k=0 to k=N-Nl-1. Shape (B, N-Nl, n_hid)
        activations_batched, # states evolutions from k=Nw to k=N-Nl-1. Shape (N-Nl-Nw, n_hid)
        predictions_batched, # predicted time sequence from k=Nw+Nl to k=N-1. Shape (B, N-Nl-Nw, n_inp)
        target_batched # target time sequence from k=Nw+Nl to k=N-1. Shape (B, N-Nl-Nw, n_inp)
    )

# Build test dataset
dataset = get_lorenz(dim=5, F=8, lag=lag, washout=washout) # from k=0 to k=N-1. Shape (B, N, n_inp)

# Test on test set
test_nrmse, states_histories, activations, predictions, targets = test(dataset, predictor, scaler)
print(f'NMRSE on the test set: {test_nrmse}')

# Visualize time sequences (just the first one from the batch)
Nw = washout
Nl = lag
N = dataset.shape[1]

time = dt * np.arange(0, N)
full_sequence = dataset[0].numpy() # first n_inp-dimensional sequence of the batch. Shape (N, n_inp)
prediction = predictions[0] # first n_inp-dimensional sequence of the batch. Shape (N-Nw-Nl, n_inp)
states_histories = states_histories[0] # first n_hid-dimensional sequence of the batch. Shape (N-Nl, n_hid)

fig, axs = plt.subplots(n_inp, 1, figsize=(12,12))
for i, ax in enumerate(axs):
    ax.plot(time, full_sequence[:, i], 'k--', label='full sequence')
    ax.plot(time[Nw:N-Nl], full_sequence[Nw:N-Nl, i], 'k', label='test sequence')
    ax.plot(time[Nw+Nl:], prediction[:, i], 'r', label='predicted sequence')
    ax.grid(True)
    ax.set_xlabel('t [s]') if i==n_inp-1 else ax.set_xlabel('')
    ax.set_ylabel(rf'$u_{{{i+1}}}$')
    ax.set_title(f'Component {i+1}')
    ax.legend()
plt.tight_layout()
plt.savefig(plots_dir/'prediction', bbox_inches='tight')
#plt.show()

# Show dynamics of the reservoir: states, velocities, accelerations and input in time
velocities_histories = np.diff(states_histories, axis=0) / dt # from k=0 to k=N-Nl-1-1. shape (N-Nl-1, n_hid)
accelerations_histories = np.diff(velocities_histories, axis=0) / dt # from k=0 to k=N-Nl-1-2. shape (N-Nl-2, n_hid)
input_history = dataset[0, :-Nl] # from k=0 to k=N-Nl-1. shape (N-Nl, n_inp)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12,12))
for i in range(n_hid):
    ax1.plot(time[:N-Nl], states_histories[:,i], label=rf'$y_{{{i+1}}}(t)$')
    ax2.plot(time[:N-Nl-1], velocities_histories[:,i], label=rf'$yd_{{{i+1}}}(t)$')
    ax3.plot(time[:N-Nl-2], accelerations_histories[:,i], label=rf'$ydd_{{{i+1}}}(t)$')
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
if n_hid < 16:
    ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    ax3.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
plt.tight_layout()
plt.savefig(plots_dir/'states_evolution', bbox_inches='tight')
#plt.show()

# Show dynamics of the reservoir: (y, yd) in y,yd plane (!! MAX FIRST 15 STATES !!)
if n_hid > 15:
    n_hid_show = 15
else:
    n_hid_show = n_hid

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

# Show dynamics of the reservoir: y(t) (in separate plots, !! MAX FIRST 15 STATES !!)
fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 9))
if n_hid_show == 1:
    axs = np.array([axs])
else:
    axs = axs.flatten()

for i in range(n_hid_show):
    axs[i].plot(time[:N-Nl], states_histories[:,i])
    axs[i].grid(True)
    axs[i].set_xlabel('t [s]')
    axs[i].set_ylabel(rf'$y_{{{i+1}}}$')
    axs[i].set_title(f'Component {i+1}')

for i in range(n_hid_show, len(axs)):
    axs[i].set_visible(False)

plt.tight_layout()
plt.savefig(plots_dir/'y_evolution', bbox_inches='tight')
plt.show()

# Save dynamics of the reservoir
np.savez(
    save_results_dir/'RON_evolution.npz', 
    time = time[:N-Nl-2],
    y = states_histories[:-2], 
    yd = velocities_histories[:-1], 
    ydd = accelerations_histories[:],
    u = input_history[:-2, None]
)
